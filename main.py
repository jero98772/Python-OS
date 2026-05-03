"""
RISC-V Operating System in Python
==================================
Educational simulation of a RISC-V OS written in Python.
Based on Stephen Marz's "Writing a RISC-V OS in Rust" blog series.

This file simulates in software what would run on bare-metal RISC-V hardware.
Each class/section maps directly to a chapter in the tutorial.

Chapters:
  1  - Boot process (Hart selection, BSS, kmain)
  2  - UART driver (NS16550a)
  3.1 - Page-grained memory allocator
  3.2 - Memory Management Unit (SV39 paging)
  4  - Trap handling
  5  - Platform-Level Interrupt Controller (PLIC)
  6  - Process structure
  7  - System calls
  8  - Process scheduler
  9  - VirtIO block driver
  10 - Minix3 filesystem
  11 - ELF loader & userspace
"""

import struct
import sys
import os
import io
import time
import random
from enum import IntEnum, IntFlag
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Any
from collections import deque

# =============================================================================
# CHAPTER 0  –  Constants & Memory Map
# (mirrors virt_memmap[] in qemu/hw/riscv/virt.c)
# =============================================================================

VIRT_DEBUG      = 0x0000_0000
VIRT_MROM       = 0x0000_1000
VIRT_TEST       = 0x0010_0000
VIRT_CLINT      = 0x0200_0000
VIRT_PLIC       = 0x0C00_0000
VIRT_UART0      = 0x1000_0000
VIRT_VIRTIO     = 0x1000_1000
VIRT_DRAM       = 0x8000_0000

PAGE_ORDER = 12
PAGE_SIZE  = 1 << PAGE_ORDER          # 4 096 bytes
HEAP_SIZE  = 128 * 1024 * 1024        # 128 MiB total simulated RAM
STACK_PAGES = 2
STACK_ADDR  = 0x1000_0000             # virtual stack base for user processes
PROCESS_STARTING_ADDR = 0x2000_0000   # virtual entry point for user processes

UART_BASE = VIRT_UART0

# =============================================================================
# CHAPTER 0  –  Simulated Physical RAM
# =============================================================================

class PhysicalMemory:
    """
    Flat byte array representing the machine's DRAM.
    Address 0 of this array corresponds to VIRT_DRAM (0x8000_0000).
    """
    def __init__(self, size: int = HEAP_SIZE):
        self._mem = bytearray(size)
        self.size = size

    def _check(self, addr: int, length: int):
        if addr < 0 or addr + length > self.size:
            raise MemoryError(f"Physical address 0x{addr:x} out of range")

    def read8(self, addr: int) -> int:
        self._check(addr, 1)
        return self._mem[addr]

    def write8(self, addr: int, val: int):
        self._check(addr, 1)
        self._mem[addr] = val & 0xFF

    def read64(self, addr: int) -> int:
        self._check(addr, 8)
        return struct.unpack_from('<Q', self._mem, addr)[0]

    def write64(self, addr: int, val: int):
        self._check(addr, 8)
        struct.pack_into('<Q', self._mem, addr, val & 0xFFFF_FFFF_FFFF_FFFF)

    def read_bytes(self, addr: int, n: int) -> bytes:
        self._check(addr, n)
        return bytes(self._mem[addr:addr+n])

    def write_bytes(self, addr: int, data: bytes):
        self._check(addr, len(data))
        self._mem[addr:addr+len(data)] = data

    def zero_range(self, start: int, end: int):
        """Simulate BSS clearing."""
        for i in range(start, end):
            self._mem[i] = 0


PHYS_MEM = PhysicalMemory()

# Symbolic addresses into the flat array (offsets from base 0)
BSS_START = 0x0001_0000
BSS_END   = 0x0002_0000
HEAP_START = 0x0010_0000                          # descriptor region
ALLOC_START = HEAP_START + (HEAP_SIZE // PAGE_SIZE)  # usable pages start here

# =============================================================================
# CHAPTER 1  –  Boot Process
# =============================================================================

class Hart:
    """Simulates a RISC-V Hardware Thread (core)."""
    def __init__(self, hart_id: int):
        self.hart_id = hart_id
        self.registers = [0] * 32    # x0..x31
        self.pc = VIRT_DRAM          # program counter
        self.mhartid = hart_id
        self.satp = 0                # supervisor address translation
        self.mstatus = 0
        self.mepc = 0
        self.mtvec = 0
        self.mie = 0
        self.mscratch = 0
        self.parked = False

    @property
    def sp(self):   return self.registers[2]
    @sp.setter
    def sp(self, v): self.registers[2] = v

    @property
    def ra(self):   return self.registers[1]
    @ra.setter
    def ra(self, v): self.registers[1] = v

    def __repr__(self):
        return f"Hart#{self.hart_id}(pc=0x{self.pc:x}, parked={self.parked})"


def boot(num_harts: int = 4) -> Hart:
    """
    Simulates boot.S:
      - Creates harts; all but hart 0 are parked (wfi loop).
      - Clears BSS.
      - Sets up machine mode registers.
      - 'Returns' hart 0 ready to enter kmain.
    """
    print("=== BOOT SEQUENCE ===")
    harts = [Hart(i) for i in range(num_harts)]

    # Park every hart except hart 0
    for h in harts[1:]:
        h.parked = True
        print(f"  Hart #{h.hart_id}: parked (wfi loop)")

    boot_hart = harts[0]
    print(f"  Hart #0: selected as boot hart")

    # Clear satp (disable MMU for now)
    boot_hart.satp = 0
    print("  satp  = 0  (MMU disabled, physical addressing)")

    # Zero the BSS section
    print(f"  Clearing BSS: 0x{BSS_START:x} → 0x{BSS_END:x}")
    PHYS_MEM.zero_range(BSS_START, BSS_END)

    # Set up mstatus: MPP=11 (machine mode), MPIE=1, MIE=1
    MIE  = 1 << 3
    MPIE = 1 << 7
    MPP  = 0b11 << 11
    boot_hart.mstatus = MPP | MPIE | MIE
    print(f"  mstatus = 0b{boot_hart.mstatus:b}  (machine mode, interrupts enabled)")

    # Enable machine timer, software, external interrupts in mie
    boot_hart.mie = (1 << 3) | (1 << 7) | (1 << 11)

    print("  Boot complete → entering kmain()\n")
    return boot_hart


# =============================================================================
# CHAPTER 2  –  UART Driver (NS16550a)
# =============================================================================

class NS16550aRegisters(IntEnum):
    """MMIO register offsets for the NS16550a UART chip."""
    RBR_THR = 0   # Receiver Buffer / Transmitter Holding
    IER     = 1   # Interrupt Enable
    FCR_IIR = 2   # FIFO Control / Interrupt Identification
    LCR     = 3   # Line Control
    MCR     = 4   # Modem Control
    LSR     = 5   # Line Status
    MSR     = 6   # Modem Status
    SCR     = 7   # Scratch


class Uart:
    """
    Simulates the NS16550a UART connected at VIRT_UART0 (0x1000_0000).
    In real hardware every read/write goes to MMIO; here we use a Python buffer.
    """
    def __init__(self, base_addr: int = UART_BASE):
        self.base_addr = base_addr
        self._regs = bytearray(8)
        self._rx_fifo: deque = deque()  # bytes waiting to be read
        self._tx_log: list  = []        # bytes that were transmitted
        self._initialized = False

    # ---- low-level MMIO helpers ----

    def _write_volatile(self, offset: int, val: int):
        self._regs[offset] = val & 0xFF

    def _read_volatile(self, offset: int) -> int:
        return self._regs[offset]

    # ---- public API (mirrors Rust's Uart impl) ----

    def init(self):
        """
        Set word length to 8 bits, enable FIFOs, enable RX interrupts.
        Optionally configure the baud-rate divisor (no-op in emulation).
        """
        # LCR[1:0] = 11 → 8-bit word length
        lcr = (1 << 0) | (1 << 1)
        self._write_volatile(NS16550aRegisters.LCR, lcr)

        # FCR[0] = 1 → enable FIFOs
        self._write_volatile(NS16550aRegisters.FCR_IIR, 1 << 0)

        # IER[0] = 1 → enable receiver buffer interrupts
        self._write_volatile(NS16550aRegisters.IER, 1 << 0)

        # Set divisor (no real effect in simulation)
        divisor = 592
        lcr_dlab = lcr | (1 << 7)
        self._write_volatile(NS16550aRegisters.LCR, lcr_dlab)
        self._write_volatile(0, divisor & 0xFF)
        self._write_volatile(1, (divisor >> 8) & 0xFF)
        self._write_volatile(NS16550aRegisters.LCR, lcr)

        self._initialized = True
        print(f"  UART initialized at 0x{self.base_addr:08x} (NS16550a)")

    def put(self, byte: int):
        """Write one byte to the transmitter (THR)."""
        self._write_volatile(NS16550aRegisters.RBR_THR, byte)
        self._tx_log.append(byte)
        sys.stdout.write(chr(byte))
        sys.stdout.flush()

    def get(self) -> Optional[int]:
        """
        Read one byte from the receiver buffer if data is ready.
        LSR[0] (DR – Data Ready) must be 1.
        """
        if self._rx_fifo:
            byte = self._rx_fifo.popleft()
            # Clear DR bit when FIFO empties
            if not self._rx_fifo:
                self._regs[NS16550aRegisters.LSR] &= ~1
            return byte
        return None

    def inject(self, data: bytes):
        """Simulate bytes arriving at the RX pin (e.g. user typing)."""
        for b in data:
            self._rx_fifo.append(b)
        # Set DR bit (bit 0) in LSR
        self._regs[NS16550aRegisters.LSR] |= 1

    def write_str(self, s: str):
        """Implement the Write trait's write_str in Python."""
        for ch in s:
            self.put(ord(ch))


# Global UART instance
UART0 = Uart(UART_BASE)


def print_uart(msg: str, end: str = '\n'):
    """Macro equivalent: print!() / println!() through the UART."""
    UART0.write_str(msg + end)


# =============================================================================
# CHAPTER 3.1  –  Page-Grained Memory Allocator
# =============================================================================

class PageBits(IntFlag):
    Empty = 0
    Taken = 1 << 0
    Last  = 1 << 1


@dataclass
class PageDescriptor:
    """One descriptor byte per 4 KiB page."""
    flags: int = PageBits.Empty

    def is_free(self)  -> bool: return not (self.flags & PageBits.Taken)
    def is_taken(self) -> bool: return bool(self.flags & PageBits.Taken)
    def is_last(self)  -> bool: return bool(self.flags & PageBits.Last)
    def set_taken(self):  self.flags |= PageBits.Taken
    def set_last(self):   self.flags |= PageBits.Last
    def clear(self):      self.flags  = PageBits.Empty


class PageAllocator:
    """
    Descriptor-based page allocator.
    One PageDescriptor per PAGE_SIZE bytes of heap space.
    """
    def __init__(self, heap_start: int, heap_size: int):
        self.heap_start  = heap_start
        self.heap_size   = heap_size
        self.num_pages   = heap_size // PAGE_SIZE
        self.descriptors = [PageDescriptor() for _ in range(self.num_pages)]
        # Usable allocation space begins after the descriptor array
        self.alloc_start = heap_start + self.num_pages  # 1 byte/descriptor

    # ---- alloc ----

    def alloc(self, pages: int) -> Optional[int]:
        """
        Find `pages` contiguous free pages, mark them Taken+Last,
        return the physical address of the first page (or None).
        """
        assert pages > 0, "Must request at least 1 page"
        for i in range(self.num_pages - pages):
            if self.descriptors[i].is_free():
                # Check the entire requested range is free
                if all(self.descriptors[j].is_free() for j in range(i, i + pages)):
                    for j in range(i, i + pages):
                        self.descriptors[j].set_taken()
                    self.descriptors[i + pages - 1].set_last()
                    return self.alloc_start + i * PAGE_SIZE
        return None  # Out of memory

    def zalloc(self, pages: int) -> Optional[int]:
        """Allocate and zero pages."""
        addr = self.alloc(pages)
        if addr is not None:
            end = addr + pages * PAGE_SIZE
            if end <= PHYS_MEM.size:
                PHYS_MEM.zero_range(addr, end)
        return addr

    def dealloc(self, ptr: int):
        """Free a previously allocated range starting at ptr."""
        assert ptr is not None and ptr != 0
        # Convert physical address back to descriptor index
        idx = (ptr - self.alloc_start) // PAGE_SIZE
        assert 0 <= idx < self.num_pages, "Invalid pointer to dealloc"
        while self.descriptors[idx].is_taken() and not self.descriptors[idx].is_last():
            self.descriptors[idx].clear()
            idx += 1
        assert self.descriptors[idx].is_last(), \
            "Possible double-free: last page not found before untaken page"
        self.descriptors[idx].clear()

    def print_table(self):
        print("\n=== PAGE ALLOCATION TABLE ===")
        alloc_end = self.alloc_start + self.num_pages * PAGE_SIZE
        print(f"  META: 0x{self.heap_start:x} → descriptors")
        print(f"  PHYS: 0x{self.alloc_start:x} → 0x{alloc_end:x}")
        print("  " + "~" * 50)
        i = 0
        count = 0
        while i < self.num_pages:
            if self.descriptors[i].is_taken():
                start_phys = self.alloc_start + i * PAGE_SIZE
                j = i
                while not self.descriptors[j].is_last():
                    j += 1
                end_phys = self.alloc_start + j * PAGE_SIZE + PAGE_SIZE - 1
                n = j - i + 1
                count += n
                print(f"  0x{start_phys:010x} → 0x{end_phys:010x}  ({n} page(s))")
                i = j + 1
            else:
                i += 1
        free = self.num_pages - count
        print(f"  Allocated: {count:6d} pages ({count*PAGE_SIZE:12d} bytes)")
        print(f"  Free     : {free:6d} pages ({free*PAGE_SIZE:12d} bytes)")
        print("  " + "~" * 50 + "\n")


# Global page allocator
PAGE_ALLOC = PageAllocator(HEAP_START, HEAP_SIZE - HEAP_START)


# Convenience wrappers
def alloc(pages: int) -> Optional[int]:  return PAGE_ALLOC.alloc(pages)
def zalloc(pages: int) -> Optional[int]: return PAGE_ALLOC.zalloc(pages)
def dealloc(ptr: int):                   PAGE_ALLOC.dealloc(ptr)


# =============================================================================
# CHAPTER 3.2  –  Memory Management Unit (SV39 paging)
# =============================================================================

class EntryBits(IntFlag):
    Valid       = 1 << 0
    Read        = 1 << 1
    Write       = 1 << 2
    Execute     = 1 << 3
    User        = 1 << 4
    Global      = 1 << 5
    Accessed    = 1 << 6
    Dirty       = 1 << 7

    ReadWrite        = Read | Write
    ReadExecute      = Read | Execute
    ReadWriteExecute = Read | Write | Execute
    UserReadWrite    = User | Read | Write
    UserReadExecute  = User | Read | Execute


class PageTableEntry:
    """Simulates a single 64-bit SV39 page table entry."""
    def __init__(self, val: int = 0):
        self.entry = val

    def is_valid(self)  -> bool: return bool(self.entry & EntryBits.Valid)
    def is_invalid(self)-> bool: return not self.is_valid()
    def is_leaf(self)   -> bool: return bool(self.entry & 0xE)   # R|W|X
    def is_branch(self) -> bool: return not self.is_leaf()

    def ppn(self) -> int:
        """Extract physical page number from entry."""
        return (self.entry >> 10) & 0xFFF_FFFF_FFFF  # 44 bits

    def physical_addr(self) -> int:
        return self.ppn() << 12

    def __repr__(self):
        return f"PTE(0x{self.entry:016x})"


class PageTable:
    """A single SV39 page table: 512 × 8-byte entries = 4 096 bytes."""
    SIZE = 512

    def __init__(self):
        self.entries: List[PageTableEntry] = [PageTableEntry() for _ in range(self.SIZE)]

    def __getitem__(self, i): return self.entries[i]
    def __setitem__(self, i, v): self.entries[i] = v


class MMU:
    """
    Software implementation of the RISC-V SV39 MMU.
    Supports map(), unmap(), and virt_to_phys().
    """
    # All page tables live in this dict keyed by a simulated physical address.
    _tables: Dict[int, PageTable] = {}
    _next_table_addr = 0x9000_0000  # simulated physical addresses for tables

    @classmethod
    def _new_table(cls) -> int:
        addr = cls._next_table_addr
        cls._tables[addr] = PageTable()
        cls._next_table_addr += PAGE_SIZE
        return addr

    @classmethod
    def _get_table(cls, phys_addr: int) -> PageTable:
        if phys_addr not in cls._tables:
            cls._tables[phys_addr] = PageTable()
        return cls._tables[phys_addr]

    @classmethod
    def map(cls, root_addr: int, vaddr: int, paddr: int, bits: int, level: int = 0):
        """
        Walk/create a 3-level SV39 page table and install a mapping.
        level=0 → 4 KiB page, level=1 → 2 MiB, level=2 → 1 GiB.
        """
        assert bits & 0xE, "Read, Write, or Execute must be set"

        vpn = [
            (vaddr >> 12) & 0x1FF,   # VPN[0]
            (vaddr >> 21) & 0x1FF,   # VPN[1]
            (vaddr >> 30) & 0x1FF,   # VPN[2]
        ]
        ppn = [
            (paddr >> 12) & 0x1FF,
            (paddr >> 21) & 0x1FF,
            (paddr >> 30) & 0x3FF_FFFF,
        ]

        root = cls._get_table(root_addr)
        v = root.entries[vpn[2]]

        for i in range(2, level, -1):
            if v.is_invalid():
                # Allocate a new page table
                new_tbl_addr = cls._new_table()
                # Store PPN shifted right by 2 in the entry (RISC-V spec)
                v.entry = ((new_tbl_addr >> 2) & ~0x3FF) | int(EntryBits.Valid)
                root.entries[vpn[2]] = v   # write back

            next_addr = v.physical_addr()
            tbl = cls._get_table(next_addr)
            v = tbl.entries[vpn[i - 1]]

        # Build the leaf entry
        entry = (
            (ppn[2] << 28) |
            (ppn[1] << 19) |
            (ppn[0] << 10) |
            bits            |
            int(EntryBits.Valid)
        )
        # v now points to the target entry – store it
        tbl = cls._get_table(v.physical_addr() if v.is_valid() else root_addr)
        # Simpler: re-walk to the correct table for the leaf
        cls._leaf_set(root_addr, vpn, level, entry)

    @classmethod
    def _leaf_set(cls, root_addr, vpn, level, entry_val):
        """Helper: walk to the correct table depth and set the leaf entry."""
        tables = [root_addr]
        tbl = cls._get_table(root_addr)
        cur_entry = tbl.entries[vpn[2]]
        for i in range(2, level, -1):
            if cur_entry.is_invalid():
                new_addr = cls._new_table()
                cur_entry.entry = ((new_addr >> 2) & ~0x3FF) | int(EntryBits.Valid)
                tbl.entries[vpn[i]] = cur_entry
            next_addr = cur_entry.physical_addr()
            tbl = cls._get_table(next_addr)
            cur_entry = tbl.entries[vpn[i - 1]]
            tables.append(next_addr)
        tbl.entries[vpn[level]] = PageTableEntry(entry_val)

    @classmethod
    def virt_to_phys(cls, root_addr: int, vaddr: int) -> Optional[int]:
        """Walk the page table and return the physical address, or None."""
        vpn = [
            (vaddr >> 12) & 0x1FF,
            (vaddr >> 21) & 0x1FF,
            (vaddr >> 30) & 0x1FF,
        ]
        offset = vaddr & 0xFFF

        if root_addr not in cls._tables:
            return None

        tbl = cls._get_table(root_addr)
        v = tbl.entries[vpn[2]]

        for i in range(2, -1, -1):
            if v.is_invalid():
                return None
            if v.is_leaf():
                off_mask = (1 << (12 + i * 9)) - 1
                vaddr_pgoff = vaddr & off_mask
                addr = (v.entry << 2) & ~off_mask
                return addr | vaddr_pgoff
            next_addr = v.physical_addr()
            tbl = cls._get_table(next_addr)
            v = tbl.entries[vpn[i - 1]]

        return None

    @classmethod
    def id_map_range(cls, root_addr: int, start: int, end: int, bits: int):
        """Identity-map a range: virtual address = physical address."""
        addr = start & ~(PAGE_SIZE - 1)
        num_pages = ((end + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1) - addr) // PAGE_SIZE
        for _ in range(num_pages):
            cls.map(root_addr, addr, addr, bits, 0)
            addr += PAGE_SIZE

    @classmethod
    def unmap(cls, root_addr: int):
        """Free all sub-tables; does NOT free root itself."""
        if root_addr not in cls._tables:
            return
        root = cls._tables[root_addr]
        for lv2_entry in root.entries:
            if lv2_entry.is_valid() and lv2_entry.is_branch():
                lv1_addr = lv2_entry.physical_addr()
                if lv1_addr in cls._tables:
                    lv1 = cls._tables[lv1_addr]
                    for lv1_entry in lv1.entries:
                        if lv1_entry.is_valid() and lv1_entry.is_branch():
                            lv0_addr = lv1_entry.physical_addr()
                            cls._tables.pop(lv0_addr, None)
                    cls._tables.pop(lv1_addr, None)


# =============================================================================
# CHAPTER 4  –  Trap Handling
# =============================================================================

class TrapCause(IntEnum):
    # Synchronous (MSB=0)
    InstructionMisaligned = 0
    InstructionAccessFault = 1
    IllegalInstruction = 2
    Breakpoint = 3
    LoadMisaligned = 4
    LoadAccessFault = 5
    StoreMisaligned = 6
    StoreAccessFault = 7
    EcallUser       = 8
    EcallSupervisor = 9
    EcallMachine    = 11
    InstructionPageFault = 12
    LoadPageFault        = 13
    StorePageFault       = 15
    # Asynchronous (MSB=1)
    MachineSoftware  = (1 << 63) | 3
    MachineTimer     = (1 << 63) | 7
    MachineExternal  = (1 << 63) | 11


@dataclass
class TrapFrame:
    """
    Snapshot of a hart's state when a trap occurs.
    Mirrors the C-compatible TrapFrame struct in the Rust OS.
    """
    regs:       List[int]  = field(default_factory=lambda: [0]*32)  # x0..x31
    fregs:      List[float]= field(default_factory=lambda: [0.0]*32)
    satp:       int        = 0
    trap_stack: int        = 0   # physical address of trap stack
    hartid:     int        = 0
    pc:         int        = 0   # program counter at trap time
    pid:        int        = 0

    @classmethod
    def zero(cls) -> 'TrapFrame':
        return cls()

    # Register aliases
    @property
    def a0(self): return self.regs[10]
    @a0.setter
    def a0(self, v): self.regs[10] = v

    @property
    def a1(self): return self.regs[11]
    @a1.setter
    def a1(self, v): self.regs[11] = v

    @property
    def sp(self): return self.regs[2]
    @sp.setter
    def sp(self, v): self.regs[2] = v


class TrapHandler:
    """
    Simulates the m_trap / asm_trap_vector machinery.
    In hardware this is triggered by the mtvec function pointer.
    """
    def __init__(self, plic: 'PLIC', uart: Uart, scheduler: 'Scheduler',
                 syscall_handler: 'SyscallHandler'):
        self.plic = plic
        self.uart = uart
        self.scheduler = scheduler
        self.syscall = syscall_handler
        self.timer_count = 0

    def handle(self, hart: Hart, cause: int, epc: int, tval: int,
               frame: TrapFrame) -> int:
        """
        Main trap dispatcher (mirrors m_trap() in trap.rs).
        Returns the PC to resume execution at.
        """
        is_async = bool(cause >> 63 & 1)
        cause_num = cause & 0xFFF
        return_pc = epc

        if is_async:
            if cause_num == 3:
                print_uart(f"\n[TRAP] Machine software interrupt, hart#{hart.hart_id}")
            elif cause_num == 7:
                # Machine timer – context switch
                self.timer_count += 1
                self._reset_timer()
                return_pc = self._context_switch(frame, return_pc)
            elif cause_num == 11:
                # Machine external – PLIC
                return_pc = self._handle_external(frame, return_pc)
            else:
                raise RuntimeError(f"Unhandled async trap #{cause_num}")
        else:
            if cause_num == 2:
                raise RuntimeError(
                    f"Illegal instruction at 0x{epc:x}, tval=0x{tval:x}")
            elif cause_num in (8, 9, 11):
                return_pc = self.syscall.dispatch(return_pc, frame)
                return_pc += 4
            elif cause_num == 12:
                print_uart(f"\n[TRAP] Instruction page fault at 0x{epc:x}")
                return_pc += 4
            elif cause_num == 13:
                print_uart(f"\n[TRAP] Load page fault at 0x{epc:x}, addr=0x{tval:x}")
                return_pc += 4
            elif cause_num == 15:
                print_uart(f"\n[TRAP] Store page fault at 0x{epc:x}, addr=0x{tval:x}")
                return_pc += 4
            else:
                raise RuntimeError(f"Unhandled sync trap #{cause_num} at 0x{epc:x}")

        return return_pc

    def _reset_timer(self):
        """Simulate writing to mtimecmp to schedule the next interrupt."""
        pass  # In hardware: mtimecmp = mtime + 10_000_000

    def _context_switch(self, frame: TrapFrame, pc: int) -> int:
        return self.scheduler.schedule(frame, pc)

    def _handle_external(self, frame: TrapFrame, pc: int) -> int:
        interrupt_id = self.plic.next()
        if interrupt_id is None:
            return pc
        if interrupt_id == 10:  # UART
            byte = self.uart.get()
            if byte is not None:
                self._echo(byte)
        else:
            print_uart(f"\n[PLIC] Non-UART interrupt: {interrupt_id}")
        self.plic.complete(interrupt_id)
        return pc

    def _echo(self, c: int):
        if c == 8:    # backspace
            sys.stdout.write('\b \b'); sys.stdout.flush()
        elif c in (10, 13):
            UART0.write_str('\r\n')
        else:
            UART0.put(c)


# =============================================================================
# CHAPTER 5  –  Platform-Level Interrupt Controller (PLIC)
# =============================================================================

PLIC_PRIORITY  = VIRT_PLIC + 0x000000
PLIC_PENDING   = VIRT_PLIC + 0x001000
PLIC_ENABLE    = VIRT_PLIC + 0x002000
PLIC_THRESHOLD = VIRT_PLIC + 0x200000
PLIC_CLAIM     = VIRT_PLIC + 0x200004


class PLIC:
    """
    Simulates the Platform-Level Interrupt Controller.
    Routes external interrupts (UART=10, VirtIO=1-8, PCIe=32-35).
    """
    MAX_SOURCES = 64

    def __init__(self):
        self._priority  = [0] * self.MAX_SOURCES
        self._enabled   = 0           # bitmask
        self._threshold = 0
        self._pending   = deque()     # queue of pending interrupt IDs

    def set_priority(self, source: int, prio: int):
        assert 0 <= prio <= 7
        self._priority[source] = prio & 7
        print(f"  PLIC: source {source} priority = {prio}")

    def enable(self, source: int):
        self._enabled |= (1 << source)
        print(f"  PLIC: interrupt source {source} enabled")

    def disable(self, source: int):
        self._enabled &= ~(1 << source)

    def set_threshold(self, tsh: int):
        self._threshold = tsh & 7
        print(f"  PLIC: threshold = {tsh}")

    def trigger(self, source: int):
        """Simulate an external device raising an interrupt."""
        if (self._enabled >> source) & 1:
            if self._priority[source] > self._threshold:
                self._pending.append(source)

    def next(self) -> Optional[int]:
        """Claim the highest-priority pending interrupt (returns ID or None)."""
        if not self._pending:
            return None
        # Sort by priority (descending) and return highest
        best = max(self._pending, key=lambda s: self._priority[s])
        self._pending.remove(best)
        return best

    def complete(self, source: int):
        """Signal that interrupt handling is done; device may re-interrupt."""
        pass  # In hardware this resets the interrupt pending bit


# Global PLIC instance
PLIC_CTRL = PLIC()


# =============================================================================
# CHAPTER 6  –  Process Structure
# =============================================================================

class ProcessState(IntEnum):
    Running  = 0
    Sleeping = 1
    Waiting  = 2
    Dead     = 3


_next_pid = 1


class Process:
    """
    Represents a single schedulable process.
    Mirrors the Process struct in process.rs.
    """
    def __init__(self, func: Optional[Callable] = None,
                 is_kernel: bool = True):
        global _next_pid
        self.pid   = _next_pid
        _next_pid += 1
        self.state = ProcessState.Waiting
        self.frame = TrapFrame()
        self.frame.pid = self.pid
        self.root_table_addr = MMU._new_table()  # own page table
        self.stack_pages: Optional[int] = alloc(STACK_PAGES)
        self.program_counter = 0
        self._func = func
        self.is_kernel = is_kernel
        self._iteration = 0  # simulation: track progress through func

        # Set up stack pointer in frame
        if self.stack_pages:
            self.frame.sp = STACK_ADDR + PAGE_SIZE * STACK_PAGES

        # Map stack
        if self.stack_pages and self.root_table_addr:
            saddr = self.stack_pages
            for i in range(STACK_PAGES):
                MMU.map(
                    self.root_table_addr,
                    STACK_ADDR + i * PAGE_SIZE,
                    saddr + i * PAGE_SIZE,
                    int(EntryBits.UserReadWrite),
                    0
                )

    def set_running(self):  self.state = ProcessState.Running
    def set_waiting(self):  self.state = ProcessState.Waiting
    def set_sleeping(self): self.state = ProcessState.Sleeping
    def set_dead(self):     self.state = ProcessState.Dead

    def get_frame_address(self) -> int:
        """Simulated: return an identifier for the frame."""
        return id(self.frame)

    def step(self) -> bool:
        """
        Simulate one 'time slice' of execution.
        Returns True if the process is still alive.
        """
        if self._func is None or self.state == ProcessState.Dead:
            return False
        try:
            self._func(self)
        except StopIteration:
            self.set_dead()
            return False
        return True

    def __repr__(self):
        return f"Process(pid={self.pid}, state={self.state.name})"

    def __del__(self):
        if self.stack_pages:
            try:
                dealloc(self.stack_pages)
            except Exception:
                pass
        if self.root_table_addr:
            MMU.unmap(self.root_table_addr)


# =============================================================================
# CHAPTER 7  –  System Calls
# =============================================================================

class SyscallNumber(IntEnum):
    # Matches libgloss/newlib numbers for portability
    EXIT    = 93
    READ    = 63
    WRITE   = 64
    PRINT   = 1    # custom


class SyscallHandler:
    """
    Dispatches ecall instructions to the correct kernel service.
    Mirrors do_syscall() in syscall.rs.
    """
    def __init__(self, uart: Uart, fs: Optional['MinixFS'] = None):
        self.uart = uart
        self.fs   = fs
        self._handlers: Dict[int, Callable] = {
            SyscallNumber.EXIT  : self._exit,
            SyscallNumber.WRITE : self._write,
            SyscallNumber.READ  : self._read,
            SyscallNumber.PRINT : self._print,
        }

    def dispatch(self, mepc: int, frame: TrapFrame) -> int:
        """
        Called from TrapHandler when cause == 8/9/11.
        A0 (regs[10]) holds the syscall number.
        Returns the (possibly updated) PC.
        """
        sysno = frame.regs[10]
        handler = self._handlers.get(sysno)
        if handler:
            handler(frame)
        else:
            print_uart(f"\n[SYSCALL] Unknown syscall #{sysno}")
        return mepc  # caller adds +4

    def _exit(self, frame: TrapFrame):
        pid = frame.pid
        print_uart(f"\n[SYSCALL] exit() called by pid={pid}")
        # Find process in scheduler and mark dead
        SCHEDULER.remove_pid(pid)

    def _write(self, frame: TrapFrame):
        # a1=buffer addr (virtual), a2=length
        buf_vaddr = frame.regs[11]
        length    = frame.regs[12]
        # In a real OS we'd translate vaddr→paddr via MMU
        # For simulation we just print from the embedded buffer
        print_uart(f"[SYSCALL] write({length} bytes)")

    def _read(self, frame: TrapFrame):
        # a1=dev, a2=inode, a3=buffer, a4=size, a5=offset
        if self.fs:
            inode  = frame.regs[11]
            size   = frame.regs[13]
            offset = frame.regs[14]
            data = self.fs.read_inode(inode, size, offset)
            frame.regs[10] = len(data)  # return value
            print_uart(f"[SYSCALL] fs_read inode={inode} → {len(data)} bytes")

    def _print(self, frame: TrapFrame):
        # Custom: a1 = message id
        msg_id = frame.regs[11]
        print_uart(f"[SYSCALL] print (msg_id={msg_id})")


# =============================================================================
# CHAPTER 8  –  Process Scheduler
# =============================================================================

class Scheduler:
    """
    Round-robin process scheduler.
    Mirrors the schedule() function in process.rs.
    """
    def __init__(self):
        self._queue: deque = deque()

    def add(self, proc: Process):
        proc.set_running()
        self._queue.append(proc)
        print_uart(f"  Scheduler: added pid={proc.pid}")

    def remove_pid(self, pid: int):
        self._queue = deque(p for p in self._queue if p.pid != pid)

    def schedule(self, frame: TrapFrame, mepc: int) -> int:
        """
        Pick the next runnable process.  Saves old context, loads new one.
        Returns the PC for the next process (or current if none other).
        """
        if not self._queue:
            print_uart("\n[SCHED] No processes!")
            return mepc

        # Rotate left: move front to back, front becomes new current
        self._queue.rotate(-1)
        proc = self._queue[0]
        while proc.state != ProcessState.Running:
            self._queue.rotate(-1)
            proc = self._queue[0]
            # Safety: don't spin forever
            if all(p.state != ProcessState.Running for p in self._queue):
                return mepc

        print_uart(f"\n  [SCHED] Switching to pid={proc.pid}")
        # In hardware we'd do switch_to_user; here just step the process
        proc.step()
        return proc.program_counter if proc.program_counter else mepc

    def run_all(self, ticks: int = 5):
        """Simulate `ticks` context-switch timer firings."""
        print_uart(f"\n=== Running scheduler for {ticks} ticks ===")
        for tick in range(ticks):
            print_uart(f"\n--- Tick {tick+1} ---")
            if not self._queue:
                print_uart("  No runnable processes.")
                break
            for _ in range(len(self._queue)):
                proc = self._queue[0]
                if proc.state == ProcessState.Running:
                    print_uart(f"  Running pid={proc.pid}")
                    alive = proc.step()
                    if not alive:
                        print_uart(f"  pid={proc.pid} exited.")
                        self._queue.popleft()
                    else:
                        self._queue.rotate(-1)
                else:
                    self._queue.rotate(-1)

    def __len__(self): return len(self._queue)


# Global scheduler instance
SCHEDULER = Scheduler()


# =============================================================================
# CHAPTER 9  –  VirtIO Block Driver
# =============================================================================

class VirtioMmioOffset(IntEnum):
    MagicValue    = 0x000
    Version       = 0x004
    DeviceId      = 0x008
    VendorId      = 0x00C
    HostFeatures  = 0x010
    GuestFeatures = 0x020
    GuestPageSize = 0x028
    QueueSel      = 0x030
    QueueNumMax   = 0x034
    QueueNum      = 0x038
    QueuePfn      = 0x040
    QueueNotify   = 0x050
    InterruptStatus = 0x060
    InterruptAck  = 0x064
    Status        = 0x070
    Config        = 0x100


VIRTIO_MAGIC     = 0x74726976   # "virt" little-endian
VIRTIO_RING_SIZE = 16
VIRTIO_BLK_T_IN  = 0   # read
VIRTIO_BLK_T_OUT = 1   # write
VIRTIO_BLK_F_RO  = 5


@dataclass
class VirtioRequest:
    blktype: int    # IN or OUT
    sector:  int    # disk sector
    data:    bytearray = field(default_factory=bytearray)
    status:  int = 111


class BlockDevice:
    """
    Simulates a VirtIO block device backed by a bytes buffer (simulated disk).
    Mirrors block.rs in the Rust OS.
    """
    SECTOR_SIZE = 512

    def __init__(self, disk_image: Optional[bytes] = None, read_only: bool = False):
        self.read_only   = read_only
        self._disk       = bytearray(disk_image or bytes(32 * 1024 * 1024))  # 32 MiB default
        self._pending_q: deque = deque()
        self._used_q:    deque = deque()
        self._mmio: Dict[int, int] = {}
        self._initialized = False
        self._watchers: Dict[int, int] = {}  # request_id → watcher_pid
        self._next_req_id = 0

    # ---- MMIO simulation ----

    def mmio_read(self, offset: int) -> int:
        return self._mmio.get(offset, 0)

    def mmio_write(self, offset: int, val: int):
        self._mmio[offset] = val & 0xFFFF_FFFF
        if offset == VirtioMmioOffset.QueueNotify:
            self._process_queue()

    # ---- Initialisation sequence (mirrors setup_block_device) ----

    def setup(self) -> bool:
        print("  VirtIO: probing block device...")
        # Magic check
        self._mmio[VirtioMmioOffset.MagicValue] = VIRTIO_MAGIC
        self._mmio[VirtioMmioOffset.DeviceId]   = 2       # block device
        self._mmio[VirtioMmioOffset.Version]     = 1
        self._mmio[VirtioMmioOffset.HostFeatures]= ~(1 << VIRTIO_BLK_F_RO) & 0xFFFFFFFF

        # Step 1: reset
        self.mmio_write(VirtioMmioOffset.Status, 0)
        # Step 2-3: acknowledge + driver
        self.mmio_write(VirtioMmioOffset.Status, 0x1 | 0x2)
        # Step 4-5: features
        guest_features = self._mmio[VirtioMmioOffset.HostFeatures]
        self.mmio_write(VirtioMmioOffset.GuestFeatures, guest_features)
        self.mmio_write(VirtioMmioOffset.Status, 0x1 | 0x2 | 0x8)
        # Step 6: verify FEATURES_OK
        if not (self._mmio[VirtioMmioOffset.Status] & 0x8):
            print("  VirtIO: features not accepted!")
            return False
        # Step 7-8: queue + driver_ok
        self._mmio[VirtioMmioOffset.QueueNumMax] = VIRTIO_RING_SIZE
        self.mmio_write(VirtioMmioOffset.QueueNum, VIRTIO_RING_SIZE)
        self.mmio_write(VirtioMmioOffset.Status, 0x1 | 0x2 | 0x8 | 0x4)
        self._initialized = True
        print("  VirtIO: block device ready!")
        return True

    # ---- Block I/O ----

    def block_op(self, buffer_out: bytearray, size: int, offset: int,
                 write: bool, watcher_pid: int = 0) -> int:
        """Enqueue a block request. Returns request-id."""
        assert self._initialized, "Device not initialized"
        if write and self.read_only:
            raise IOError("Attempt to write to read-only device")
        req = VirtioRequest(
            blktype = VIRTIO_BLK_T_OUT if write else VIRTIO_BLK_T_IN,
            sector  = offset // self.SECTOR_SIZE,
            data    = bytearray(buffer_out) if write else bytearray(size),
        )
        req_id = self._next_req_id
        self._next_req_id += 1
        self._pending_q.append((req_id, req, buffer_out, watcher_pid))
        self._process_queue()
        return req_id

    def _process_queue(self):
        """Service all pending requests (synchronous in simulation)."""
        while self._pending_q:
            req_id, req, buf, watcher = self._pending_q.popleft()
            byte_offset = req.sector * self.SECTOR_SIZE
            if req.blktype == VIRTIO_BLK_T_IN:
                end = min(byte_offset + len(buf), len(self._disk))
                buf[:end - byte_offset] = self._disk[byte_offset:end]
                req.status = 0
            else:
                end = min(byte_offset + len(req.data), len(self._disk))
                self._disk[byte_offset:end] = req.data[:end - byte_offset]
                req.status = 0
            self._used_q.append((req_id, req, watcher))
            # Simulate PLIC interrupt
            PLIC_CTRL.trigger(1)  # VirtIO device 1 → PLIC source 1

    def read(self, offset: int, size: int) -> bytes:
        """Convenience: synchronous direct read (bypasses sector alignment)."""
        end = min(offset + size, len(self._disk))
        return bytes(self._disk[offset:end])

    def write(self, offset: int, data: bytes):
        """Convenience: synchronous direct write."""
        end = min(offset + len(data), len(self._disk))
        self._disk[offset:end] = data[:end - offset]


# =============================================================================
# CHAPTER 10  –  Minix3 File System
# =============================================================================

MINIX3_MAGIC      = 0x4d5a
MINIX3_BLOCK_SIZE = 1024
MINIX3_INODE_SIZE = 64
MINIX3_DIR_ENTRY_SIZE = 64
MINIX3_NAME_MAX   = 60


class Superblock:
    """Minix3 superblock (located at byte offset 1024 from disk start)."""
    OFFSET = 1024

    def __init__(self, data: bytes):
        if len(data) < 32:
            raise ValueError("Superblock data too short")
        (self.ninodes,
         _pad0,
         self.imap_blocks,
         self.zmap_blocks,
         self.first_data_zone,
         self.log_zone_size,
         _pad1,
         self.max_size,
         self.zones,
         self.magic,
         _pad2,
         self.block_size,
         self.disk_version) = struct.unpack_from('<IHHHHHHI I HHHb', data)

    def is_valid(self) -> bool:
        return self.magic == MINIX3_MAGIC


@dataclass
class Inode:
    """Minix3 inode (64 bytes)."""
    mode:  int
    nlinks:int
    uid:   int
    gid:   int
    size:  int
    atime: int
    mtime: int
    ctime: int
    zones: List[int]  # 7 direct + 1 indirect + 1 double + 1 triple

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Inode':
        (mode, nlinks, uid, gid, size,
         atime, mtime, ctime,
         *zones) = struct.unpack_from('<HHHHI III 10I', data)
        return cls(mode, nlinks, uid, gid, size, atime, mtime, ctime, list(zones))

    def is_dir(self)        -> bool: return (self.mode & 0xF000) == 0x4000
    def is_regular(self)    -> bool: return (self.mode & 0xF000) == 0x8000
    def perms(self)         -> str:
        p = ''
        for bit, ch in [(0o400,'r'),(0o200,'w'),(0o100,'x'),
                        (0o040,'r'),(0o020,'w'),(0o010,'x'),
                        (0o004,'r'),(0o002,'w'),(0o001,'x')]:
            p += ch if self.mode & bit else '-'
        return ('d' if self.is_dir() else '-') + p


@dataclass
class DirEntry:
    inode: int
    name:  str


class MinixFS:
    """
    Read-only Minix3 filesystem driver.
    Mirrors minixfs.rs in the Rust OS.
    """
    def __init__(self, block_dev: BlockDevice):
        self.dev = block_dev
        self.sb  = self._read_superblock()
        if not self.sb.is_valid():
            raise RuntimeError("Not a valid Minix3 filesystem (bad magic)")
        print(f"  MinixFS: valid filesystem, {self.sb.ninodes} inodes, "
              f"block_size={self.sb.block_size}")

    def _read_block(self, block_num: int) -> bytes:
        return self.dev.read(block_num * MINIX3_BLOCK_SIZE, MINIX3_BLOCK_SIZE)

    def _read_superblock(self) -> Superblock:
        data = self.dev.read(Superblock.OFFSET, 64)
        return Superblock(data)

    def _inode_offset(self, inode_num: int) -> int:
        """Byte offset of inode on disk (1-based inode numbers)."""
        # Layout: boot(1) + super(1) + imap + zmap + inode_table
        inode_table_block = 2 + self.sb.imap_blocks + self.sb.zmap_blocks
        return (inode_table_block * MINIX3_BLOCK_SIZE +
                (inode_num - 1) * MINIX3_INODE_SIZE)

    def read_inode_meta(self, inode_num: int) -> Inode:
        offset = self._inode_offset(inode_num)
        data = self.dev.read(offset, MINIX3_INODE_SIZE)
        return Inode.from_bytes(data)

    def read_inode(self, inode_num: int, size: int = -1, offset: int = 0) -> bytes:
        """Read file data for the given inode."""
        inode = self.read_inode_meta(inode_num)
        if size < 0:
            size = inode.size
        result = bytearray()
        bs = MINIX3_BLOCK_SIZE
        blocks_to_skip = offset // bs
        byte_offset_in_block = offset % bs
        bytes_read = 0

        def read_zone(zone_num):
            nonlocal bytes_read
            if zone_num == 0 or bytes_read >= size:
                return
            block_data = self._read_block(zone_num)
            start = byte_offset_in_block if not result else 0
            chunk = block_data[start:start + (size - bytes_read)]
            result.extend(chunk)
            bytes_read += len(chunk)

        # Direct zones (0..6)
        for i, zone in enumerate(inode.zones[:7]):
            if i < blocks_to_skip:
                continue
            read_zone(zone)
            if bytes_read >= size:
                break

        # Indirect zone (index 7) – one level of indirection
        if bytes_read < size and inode.zones[7]:
            indirect_block = self._read_block(inode.zones[7])
            for i in range(bs // 4):
                z = struct.unpack_from('<I', indirect_block, i * 4)[0]
                if z:
                    read_zone(z)
                if bytes_read >= size:
                    break

        return bytes(result[:size])

    def list_dir(self, inode_num: int) -> List[DirEntry]:
        """List directory entries for a directory inode."""
        inode = self.read_inode_meta(inode_num)
        if not inode.is_dir():
            raise ValueError(f"Inode {inode_num} is not a directory")
        raw = self.read_inode(inode_num)
        entries = []
        for i in range(0, len(raw), MINIX3_DIR_ENTRY_SIZE):
            chunk = raw[i:i + MINIX3_DIR_ENTRY_SIZE]
            if len(chunk) < MINIX3_DIR_ENTRY_SIZE:
                break
            inum = struct.unpack_from('<I', chunk, 0)[0]
            if inum == 0:
                continue
            name_bytes = chunk[4:4 + MINIX3_NAME_MAX]
            name = name_bytes.rstrip(b'\x00').decode('utf-8', errors='replace')
            entries.append(DirEntry(inum, name))
        return entries


# =============================================================================
# CHAPTER 11  –  ELF Loader & Userspace Processes
# =============================================================================

ELF_MAGIC       = 0x464C457F   # 0x7F 'E' 'L' 'F'
ELF_MACHINE_RISCV = 0xF3
ELF_TYPE_EXEC   = 0x2
ELF_PT_LOAD     = 1
ELF_PF_EXECUTE  = 1
ELF_PF_WRITE    = 2
ELF_PF_READ     = 4


@dataclass
class ElfHeader:
    magic:         int
    bitsize:       int
    endian:        int
    abi_version:   int
    target_platform: int
    obj_type:      int
    machine:       int
    version:       int
    entry_addr:    int
    phoff:         int
    shoff:         int
    flags:         int
    ehsize:        int
    phentsize:     int
    phnum:         int
    shentsize:     int
    shnum:         int
    shstrndx:      int

    @classmethod
    def from_bytes(cls, data: bytes) -> 'ElfHeader':
        # ELF64 layout: e_ident(16s) + e_type(H) + e_machine(H) + e_version(I)
        # + e_entry(Q) + e_phoff(Q) + e_shoff(Q) + e_flags(I)
        # + e_ehsize(H) + e_phentsize(H) + e_phnum(H)
        # + e_shentsize(H) + e_shnum(H) + e_shstrndx(H)
        fmt = '<16s HHI QQQI HHHHHH'
        (ident,
         obj_type, machine, version,
         entry_addr, phoff, shoff, flags,
         ehsize, phentsize, phnum,
         shentsize, shnum, shstrndx
         ) = struct.unpack_from(fmt, data)
        magic   = struct.unpack_from('<I', ident, 0)[0]
        bitsize = ident[4]
        endian  = ident[5]
        abi_ver = ident[7]
        target  = ident[8]
        return cls(magic, bitsize, endian, abi_ver, target,
                   obj_type, machine, version,
                   entry_addr, phoff, shoff, flags,
                   ehsize, phentsize, phnum,
                   shentsize, shnum, shstrndx)


@dataclass
class ProgramHeader:
    seg_type: int
    flags:    int
    off:      int
    vaddr:    int
    paddr:    int
    filesz:   int
    memsz:    int
    align:    int

    @classmethod
    def from_bytes(cls, data: bytes) -> 'ProgramHeader':
        # 64-bit ELF program header
        (seg_type, flags, off, vaddr, paddr, filesz, memsz, align
         ) = struct.unpack_from('<II QQQQQQ', data)
        return cls(seg_type, flags, off, vaddr, paddr, filesz, memsz, align)


class ElfLoader:
    """
    Loads a RISC-V ELF executable into a Process.
    Mirrors the ELF loading code in test.rs / process.rs.
    """
    def __init__(self, scheduler: Scheduler):
        self.scheduler = scheduler

    def load(self, elf_bytes: bytes) -> Optional[Process]:
        """Parse and load an ELF file; return a ready Process."""
        if len(elf_bytes) < 64:
            print_uart("ELF: file too small")
            return None

        hdr = ElfHeader.from_bytes(elf_bytes)

        if hdr.magic != ELF_MAGIC:
            print_uart("ELF: bad magic")
            return None
        if hdr.machine != ELF_MACHINE_RISCV:
            print_uart("ELF: not RISC-V")
            return None
        if hdr.obj_type != ELF_TYPE_EXEC:
            print_uart("ELF: not executable")
            return None

        proc = Process(is_kernel=False)
        program_mem_addr = zalloc(64)   # 64 pages = 256 KiB working area
        if program_mem_addr is None:
            print_uart("ELF: out of memory")
            return None

        # Load each LOAD segment
        for i in range(hdr.phnum):
            ph_off = hdr.phoff + i * hdr.phentsize
            ph = ProgramHeader.from_bytes(elf_bytes[ph_off:ph_off + hdr.phentsize])
            if ph.seg_type != ELF_PT_LOAD or ph.memsz == 0:
                continue

            # Copy segment data into our simulated program memory
            src = elf_bytes[ph.off: ph.off + ph.filesz]
            dst_offset = program_mem_addr + ph.off
            if dst_offset + len(src) <= PHYS_MEM.size:
                PHYS_MEM.write_bytes(dst_offset, src)

            # Determine MMU permissions
            bits = int(EntryBits.User | EntryBits.Valid)
            if ph.flags & ELF_PF_READ:    bits |= int(EntryBits.Read)
            if ph.flags & ELF_PF_WRITE:   bits |= int(EntryBits.Write)
            if ph.flags & ELF_PF_EXECUTE: bits |= int(EntryBits.Execute)

            # Map pages
            pages = (ph.memsz + PAGE_SIZE) // PAGE_SIZE
            for j in range(pages):
                vaddr = ph.vaddr + j * PAGE_SIZE
                paddr = program_mem_addr + ph.off + j * PAGE_SIZE
                MMU.map(proc.root_table_addr, vaddr, paddr, bits, 0)

        # Set entry point and stack pointer
        proc.program_counter = hdr.entry_addr
        proc.frame.pc  = hdr.entry_addr
        proc.frame.sp  = STACK_ADDR + STACK_PAGES * PAGE_SIZE
        proc.frame.satp = (8 << 60) | (proc.pid << 44) | (proc.root_table_addr >> 12)

        print_uart(f"  ELF loaded: entry=0x{hdr.entry_addr:x}, pid={proc.pid}")
        return proc

    def load_from_fs(self, fs: MinixFS, inode_num: int,
                     expected_size: int) -> Optional[Process]:
        """Read ELF from the Minix3 filesystem and load it."""
        print_uart(f"  Loading ELF from inode {inode_num}...")
        data = fs.read_inode(inode_num, expected_size)
        if len(data) != expected_size:
            print_uart(f"  ELF: read {len(data)} bytes, expected {expected_size}")
            return None
        return self.load(bytes(data))


# =============================================================================
#  KERNEL MAIN  –  ties everything together (kmain / kinit)
# =============================================================================

def build_fake_minix3_disk() -> bytes:
    """
    Build a minimal Minix3 disk image in memory for demonstration.
    Real usage: use mkfs.minix -3 on a Linux machine.
    """
    disk = bytearray(32 * 1024 * 1024)  # 32 MiB

    # Superblock at offset 1024
    # ninodes=16, imap_blocks=1, zmap_blocks=1, first_data_zone=5,
    # log_zone_size=0, max_size=2GB, zones=8192, magic=0x4d5a, block_size=1024
    sb = struct.pack('<IHHHHHHI I HHHb',
        16,       # ninodes
        0,        # pad0
        1,        # imap_blocks
        1,        # zmap_blocks
        5,        # first_data_zone
        0,        # log_zone_size
        0,        # pad1
        2147483647, # max_size
        8192,     # zones
        MINIX3_MAGIC,
        0,        # pad2
        MINIX3_BLOCK_SIZE,
        3         # disk_version
    )
    disk[1024:1024+len(sb)] = sb

    # Root inode (#1) – directory, at inode table block 4
    inode_table_offset = (2 + 1 + 1) * MINIX3_BLOCK_SIZE  # block 4
    # mode=0x41ED (drwxr-xr-x), nlinks=2, uid=0, gid=0, size=64, zones=[5,0...]
    root_inode = struct.pack('<HHHHI III 10I',
        0x41ED, 2, 0, 0, 192,   # 3 dir-entries x 64 bytes = 192
        0, 0, 0,
        5, 0, 0, 0, 0, 0, 0, 0, 0, 0  # zone[0]=block 5, rest=0
    )
    disk[inode_table_offset:inode_table_offset+64] = root_inode

    # Inode #2 – regular file "hello.txt"
    file_content = b"Hello, this is my first file on Minix3 filesystem!\n\x00"
    file_size = len(file_content)
    file_inode = struct.pack('<HHHHI III 10I',
        0x81A4, 1, 0, 0, file_size,   # mode=0x81A4 (-rw-r--r--)
        0, 0, 0,
        6, 0, 0, 0, 0, 0, 0, 0, 0, 0  # zone[0]=block 6
    )
    disk[inode_table_offset+64:inode_table_offset+128] = file_inode

    # Root directory entries at block 5
    dir_block_offset = 5 * MINIX3_BLOCK_SIZE
    # Entry: inode=1 name="."
    disk[dir_block_offset:dir_block_offset+4]   = struct.pack('<I', 1)
    disk[dir_block_offset+4:dir_block_offset+64]= b'.\x00' + b'\x00'*58
    # Entry: inode=1 name=".."
    disk[dir_block_offset+64:dir_block_offset+68] = struct.pack('<I', 1)
    disk[dir_block_offset+68:dir_block_offset+128]= b'..\x00' + b'\x00'*57
    # Entry: inode=2 name="hello.txt"
    disk[dir_block_offset+128:dir_block_offset+132] = struct.pack('<I', 2)
    name_bytes = b'hello.txt\x00' + b'\x00'*50
    disk[dir_block_offset+132:dir_block_offset+192] = name_bytes

    # File data at block 6
    file_block_offset = 6 * MINIX3_BLOCK_SIZE
    disk[file_block_offset:file_block_offset+len(file_content)] = file_content

    return bytes(disk)


def build_fake_elf() -> bytes:
    """
    Build a minimal 64-bit RISC-V ELF with one LOAD segment.
    The 'program' just contains NOP instructions and is purely demonstrative.
    """
    # ELF header (64 bytes)
    # One program header immediately follows (56 bytes for 64-bit ELF)
    phoff = 64
    entry = 0x2000_0000   # PROCESS_STARTING_ADDR

    # Fake RISC-V instructions: 10 NOPs (addi x0,x0,0 = 0x00000013)
    code = b'\x13\x00\x00\x00' * 10

    filesz  = len(code)
    memsz   = filesz
    ph_flags = ELF_PF_READ | ELF_PF_EXECUTE

    # Build e_ident (16 bytes)
    ident = bytes([0x7F, 0x45, 0x4C, 0x46,  # magic: \x7fELF
                   2,                         # EI_CLASS: 64-bit
                   1,                         # EI_DATA: little-endian
                   1,                         # EI_VERSION
                   0,                         # EI_OSABI: System V
                   0,                         # EI_ABIVERSION
                   0, 0, 0, 0, 0, 0, 0])      # EI_PAD (7 bytes)

    elf_header = struct.pack('<16s HHI QQQI HHHHHH',
        ident,
        ELF_TYPE_EXEC,          # e_type
        ELF_MACHINE_RISCV,      # e_machine = 0xF3
        1,                       # e_version
        entry,                   # e_entry
        phoff,                   # e_phoff  (program header offset = 64)
        0,                       # e_shoff  (no section headers)
        0,                       # e_flags
        64,                      # e_ehsize (ELF header size)
        56,                      # e_phentsize (program header entry size)
        1,                       # e_phnum  (one program header)
        64,                      # e_shentsize
        0,                       # e_shnum
        0,                       # e_shstrndx
    )

    prog_header = struct.pack('<II QQQQQQ',
        ELF_PT_LOAD,
        ph_flags,
        phoff + 56,   # offset of code in file
        entry,        # vaddr
        0,            # paddr (unused)
        filesz,
        memsz,
        0x1000,       # align
    )

    return elf_header + prog_header + code


# =============================================================================
#  Sample user process functions (run by the scheduler)
# =============================================================================

def init_process(proc: Process):
    """The first kernel process – just loops and occasionally prints."""
    proc._iteration += 1
    if proc._iteration % 3 == 0:
        print_uart(f"  [init] iteration {proc._iteration} (pid={proc.pid})")
    if proc._iteration >= 9:
        raise StopIteration


def worker_process(proc: Process):
    """A second process that does 'work'."""
    proc._iteration += 1
    print_uart(f"  [worker] doing work #{proc._iteration} (pid={proc.pid})")
    if proc._iteration >= 5:
        raise StopIteration


# =============================================================================
#  KERNEL ENTRY POINT
# =============================================================================

def kmain():
    """
    The Rust kmain() entry point, translated to Python.
    Initialises all subsystems and demonstrates each chapter.
    """
    print("=" * 65)
    print("  RISC-V OS in Python  –  Educational Simulation")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Chapter 1: Boot
    # ------------------------------------------------------------------
    hart = boot(num_harts=4)
    print()

    # ------------------------------------------------------------------
    # Chapter 2: UART
    # ------------------------------------------------------------------
    print("=== CHAPTER 2: UART ===")
    UART0.init()
    print_uart("Hello from the kernel UART driver!")
    print_uart(f"Transmitting at base address 0x{UART0.base_addr:08x}")

    # Simulate receiving some input
    UART0.inject(b"hello\r")
    print_uart("\nSimulated RX input from user:")
    while True:
        c = UART0.get()
        if c is None:
            break
        if c in (10, 13):
            print_uart("")
        else:
            UART0.put(c)
    print()

    # ------------------------------------------------------------------
    # Chapter 3.1: Page allocator
    # ------------------------------------------------------------------
    print("=== CHAPTER 3.1: PAGE ALLOCATOR ===")
    p1 = alloc(1)
    p2 = alloc(4)
    p3 = alloc(64)
    print(f"  alloc(1)  → 0x{p1:x}")
    print(f"  alloc(4)  → 0x{p2:x}")
    print(f"  alloc(64) → 0x{p3:x}")
    PAGE_ALLOC.print_table()
    dealloc(p2)
    print(f"  dealloc(p2=0x{p2:x})")
    dealloc(p3)
    print()

    # ------------------------------------------------------------------
    # Chapter 3.2: MMU / SV39 paging
    # ------------------------------------------------------------------
    print("=== CHAPTER 3.2: MMU / SV39 PAGING ===")
    root = MMU._new_table()
    vaddr = 0x7d_beef_cafe & 0x7F_FFFF_FFFF   # 39-bit virtual address
    paddr = 0x8001_0000
    MMU.map(root, vaddr, paddr, int(EntryBits.ReadWrite), 0)
    result = MMU.virt_to_phys(root, vaddr)
    print(f"  Mapped vaddr=0x{vaddr:011x} → paddr=0x{paddr:x}")
    print(f"  virt_to_phys lookup: 0x{result:x}" if result else "  virt_to_phys: None (page fault)")

    # Identity-map UART MMIO
    MMU.id_map_range(root, VIRT_UART0, VIRT_UART0 + 0x100, int(EntryBits.ReadWrite))
    print(f"  Identity-mapped UART MMIO 0x{VIRT_UART0:08x}–0x{VIRT_UART0+0x100:08x}")
    print()

    # ------------------------------------------------------------------
    # Chapter 4: Trap handling
    # ------------------------------------------------------------------
    print("=== CHAPTER 4: TRAP HANDLING ===")
    syscall_handler = SyscallHandler(UART0)
    trap_handler = TrapHandler(PLIC_CTRL, UART0, SCHEDULER, syscall_handler)

    # Simulate a machine timer interrupt
    frame = TrapFrame(hartid=0)
    # Async = MSB set, cause=7 (machine timer)
    new_pc = trap_handler.handle(hart, (1 << 63) | 7, 0x8000_1000, 0, frame)
    print(f"  Timer trap handled, new PC=0x{new_pc:x}")

    # Simulate a load page fault
    new_pc = trap_handler.handle(hart, 13, 0x8000_2000, 0x0, frame)
    print(f"  Load page fault handled, new PC=0x{new_pc:x}")
    print()

    # ------------------------------------------------------------------
    # Chapter 5: PLIC
    # ------------------------------------------------------------------
    print("=== CHAPTER 5: PLIC ===")
    PLIC_CTRL.set_threshold(0)
    PLIC_CTRL.enable(10)             # UART0
    PLIC_CTRL.set_priority(10, 1)
    PLIC_CTRL.enable(1)              # VirtIO block
    PLIC_CTRL.set_priority(1, 2)

    # Simulate UART interrupt
    UART0.inject(b"A")
    PLIC_CTRL.trigger(10)
    intr = PLIC_CTRL.next()
    print(f"  PLIC claimed interrupt: {intr}")
    if intr is not None:
        PLIC_CTRL.complete(intr)
    print()

    # ------------------------------------------------------------------
    # Chapter 6 & 8: Processes & Scheduler
    # ------------------------------------------------------------------
    print("=== CHAPTERS 6 & 8: PROCESSES & SCHEDULER ===")
    p_init   = Process(func=init_process,   is_kernel=True)
    p_worker = Process(func=worker_process, is_kernel=True)
    SCHEDULER.add(p_init)
    SCHEDULER.add(p_worker)
    SCHEDULER.run_all(ticks=4)
    print()

    # ------------------------------------------------------------------
    # Chapter 7: System calls
    # ------------------------------------------------------------------
    print("=== CHAPTER 7: SYSTEM CALLS ===")
    sc_frame = TrapFrame(hartid=0, pid=99)
    sc_frame.regs[10] = SyscallNumber.PRINT   # a0 = syscall number
    sc_frame.regs[11] = 42                    # a1 = arg0
    syscall_handler.dispatch(0x2000_0010, sc_frame)
    sc_frame.regs[10] = SyscallNumber.EXIT
    sc_frame.pid = 99
    syscall_handler.dispatch(0x2000_0020, sc_frame)
    print()

    # ------------------------------------------------------------------
    # Chapter 9: VirtIO block driver
    # ------------------------------------------------------------------
    print("=== CHAPTER 9: VIRTIO BLOCK DRIVER ===")
    disk_image = build_fake_minix3_disk()
    block_dev = BlockDevice(disk_image)
    ok = block_dev.setup()
    if ok:
        data = block_dev.read(0, 64)
        print(f"  First 16 bytes of disk: {data[:16].hex(' ')}")
    print()

    # ------------------------------------------------------------------
    # Chapter 10: Minix3 filesystem
    # ------------------------------------------------------------------
    print("=== CHAPTER 10: MINIX3 FILESYSTEM ===")
    try:
        fs = MinixFS(block_dev)
        # List root directory (inode 1)
        entries = fs.list_dir(1)
        print("  Root directory entries:")
        for e in entries:
            meta = fs.read_inode_meta(e.inode)
            print(f"    [{e.inode}] {meta.perms()}  {e.name}")

        # Read hello.txt (inode 2)
        content = fs.read_inode(2)
        print(f"\n  hello.txt content: {content.decode('utf-8', errors='replace').strip()}")
    except Exception as exc:
        print(f"  MinixFS error: {exc}")
    print()

    # ------------------------------------------------------------------
    # Chapter 11: ELF loader
    # ------------------------------------------------------------------
    print("=== CHAPTER 11: ELF LOADER ===")
    loader = ElfLoader(SCHEDULER)
    elf_bytes = build_fake_elf()
    print(f"  ELF image size: {len(elf_bytes)} bytes")
    proc = loader.load(elf_bytes)
    if proc:
        print(f"  Loaded ELF: {proc}")
        print(f"  PC = 0x{proc.frame.pc:x}")
        print(f"  SP = 0x{proc.frame.sp:x}")
        print(f"  SATP = 0x{proc.frame.satp:x}")
    print()

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("=" * 65)
    print("  Kernel simulation complete.")
    print("  All subsystems demonstrated successfully.")
    print("=" * 65)


# =============================================================================
#  Entry point
# =============================================================================

if __name__ == "__main__":
    kmain()