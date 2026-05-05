#!/usr/bin/env python3
"""
riscv_os_ui.py  –  RISC-V OS Simulator with Web UI & QEMU Integration
======================================================================
Run:   python3 riscv_os_ui.py
Then open:  http://localhost:5000

Features
--------
  • Interactive architecture diagram (clickable subsystem nodes)
  • Per-chapter step-through simulation with live console output
  • QEMU virt-machine launcher (spawns real qemu-system-riscv64)
  • QEMU serial-port bridge: kernel output streams into the browser console
  • Animated data-flow arrows that light up as each subsystem activates
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import os
import sys
import time
import struct
import threading
import subprocess
import queue
import signal
import socket
import random
from enum import IntEnum, IntFlag
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, List, Dict, Callable

# ── web framework ────────────────────────────────────────────────────────────
from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit

# ============================================================================
# ── Paste the entire OS simulation here (trimmed / inline) ──────────────────
# ============================================================================
# (All classes from risc_v_os_python.py are reproduced below so this file is
#  fully self-contained.)

PAGE_SIZE   = 4096
HEAP_SIZE   = 32 * 1024 * 1024   # 32 MiB for the UI demo (faster)
VIRT_DRAM   = 0x8000_0000
VIRT_UART0  = 0x1000_0000
VIRT_PLIC   = 0x0C00_0000
VIRT_CLINT  = 0x0200_0000
VIRT_VIRTIO = 0x1000_1000
BSS_START   = 0x0001_0000
BSS_END     = 0x0002_0000
HEAP_START  = 0x0010_0000
STACK_PAGES = 2
STACK_ADDR  = 0x1000_0000
PROCESS_STARTING_ADDR = 0x2000_0000
MINIX3_MAGIC      = 0x4d5a
MINIX3_BLOCK_SIZE = 1024
MINIX3_INODE_SIZE = 64
MINIX3_DIR_ENTRY_SIZE = 64
ELF_MAGIC         = 0x464C457F
ELF_MACHINE_RISCV = 0xF3
ELF_TYPE_EXEC     = 0x2
ELF_PT_LOAD       = 1
ELF_PF_EXECUTE    = 1
ELF_PF_WRITE      = 2
ELF_PF_READ       = 4

# ── Event bus shared between simulation and Flask ────────────────────────────
_event_q: queue.Queue = queue.Queue()

def _emit(event: str, data: dict):
    """Thread-safe emit to browser via the queue."""
    _event_q.put((event, data))


# ── Physical memory ───────────────────────────────────────────────────────────
class PhysicalMemory:
    def __init__(self, size=HEAP_SIZE):
        self._mem = bytearray(size); self.size = size
    def read8(self, a):  return self._mem[a]
    def write8(self, a, v): self._mem[a] = v & 0xFF
    def read64(self, a): return struct.unpack_from('<Q', self._mem, a)[0]
    def write64(self, a, v): struct.pack_into('<Q', self._mem, a, v & 0xFFFF_FFFF_FFFF_FFFF)
    def read_bytes(self, a, n): return bytes(self._mem[a:a+n])
    def write_bytes(self, a, d): self._mem[a:a+len(d)] = d
    def zero_range(self, s, e): self._mem[s:e] = bytes(e - s)

PHYS_MEM = PhysicalMemory()
ALLOC_START = HEAP_START + (HEAP_SIZE // PAGE_SIZE)


# ── Page allocator ────────────────────────────────────────────────────────────
class PageBits(IntFlag):
    Empty = 0; Taken = 1; Last = 2

@dataclass
class PageDescriptor:
    flags: int = 0
    def is_free(self): return not (self.flags & PageBits.Taken)
    def is_taken(self): return bool(self.flags & PageBits.Taken)
    def is_last(self): return bool(self.flags & PageBits.Last)
    def set_taken(self): self.flags |= PageBits.Taken
    def set_last(self): self.flags |= PageBits.Last
    def clear(self): self.flags = 0

class PageAllocator:
    def __init__(self, hs, hsz):
        self.heap_start = hs; self.heap_size = hsz
        self.num_pages = hsz // PAGE_SIZE
        self.descriptors = [PageDescriptor() for _ in range(self.num_pages)]
        self.alloc_start = hs + self.num_pages

    def alloc(self, pages):
        for i in range(self.num_pages - pages):
            if self.descriptors[i].is_free():
                if all(self.descriptors[j].is_free() for j in range(i, i+pages)):
                    for j in range(i, i+pages): self.descriptors[j].set_taken()
                    self.descriptors[i+pages-1].set_last()
                    return self.alloc_start + i * PAGE_SIZE
        return None

    def zalloc(self, pages):
        addr = self.alloc(pages)
        if addr:
            end = addr + pages * PAGE_SIZE
            if end <= PHYS_MEM.size: PHYS_MEM.zero_range(addr, end)
        return addr

    def dealloc(self, ptr):
        if not ptr: return
        idx = (ptr - self.alloc_start) // PAGE_SIZE
        while self.descriptors[idx].is_taken() and not self.descriptors[idx].is_last():
            self.descriptors[idx].clear(); idx += 1
        self.descriptors[idx].clear()

    def snapshot(self):
        """Return compact allocation info for the UI."""
        rows = []
        i = 0
        while i < self.num_pages:
            if self.descriptors[i].is_taken():
                start = self.alloc_start + i * PAGE_SIZE
                j = i
                while not self.descriptors[j].is_last(): j += 1
                rows.append({"addr": f"0x{start:x}", "pages": j-i+1,
                             "bytes": (j-i+1)*PAGE_SIZE})
                i = j + 1
            else: i += 1
        used = sum(r["pages"] for r in rows)
        return {"allocations": rows, "used": used,
                "free": self.num_pages - used, "total": self.num_pages}

PAGE_ALLOC = PageAllocator(HEAP_START, HEAP_SIZE - HEAP_START)

def alloc(n): return PAGE_ALLOC.alloc(n)
def zalloc(n): return PAGE_ALLOC.zalloc(n)
def dealloc(p): PAGE_ALLOC.dealloc(p)


# ── MMU ───────────────────────────────────────────────────────────────────────
class EntryBits(IntFlag):
    Valid=1; Read=2; Write=4; Execute=8; User=16; Accessed=64; Dirty=128
    ReadWrite=6; ReadExecute=10; UserReadWrite=22; UserReadExecute=26

class PageTableEntry:
    def __init__(self, v=0): self.entry = v
    def is_valid(self): return bool(self.entry & EntryBits.Valid)
    def is_leaf(self): return bool(self.entry & 0xE)
    def is_branch(self): return not self.is_leaf()
    def physical_addr(self): return ((self.entry >> 10) & 0xFFF_FFFF_FFFF) << 12

class PageTable:
    def __init__(self): self.entries = [PageTableEntry() for _ in range(512)]
    def __getitem__(self, i): return self.entries[i]
    def __setitem__(self, i, v): self.entries[i] = v

class MMU:
    _tables: Dict[int, PageTable] = {}
    _next_addr = 0x9000_0000
    _mappings: List[dict] = []   # for UI display

    @classmethod
    def _new_table(cls):
        a = cls._next_addr; cls._tables[a] = PageTable(); cls._next_addr += PAGE_SIZE
        return a

    @classmethod
    def _get_table(cls, a):
        if a not in cls._tables: cls._tables[a] = PageTable()
        return cls._tables[a]

    @classmethod
    def map(cls, root, vaddr, paddr, bits, level=0):
        vpn = [(vaddr>>12)&0x1FF, (vaddr>>21)&0x1FF, (vaddr>>30)&0x1FF]
        ppn = [(paddr>>12)&0x1FF, (paddr>>21)&0x1FF, (paddr>>30)&0x3FFFFFF]
        cls._leaf_set(root, vpn, ppn, level, bits)
        cls._mappings.append({"vaddr": f"0x{vaddr:x}", "paddr": f"0x{paddr:x}",
                               "bits": int(bits)})
        if len(cls._mappings) > 50: cls._mappings.pop(0)

    @classmethod
    def _leaf_set(cls, root_addr, vpn, ppn, level, bits):
        tbl = cls._get_table(root_addr)
        cur = tbl.entries[vpn[2]]
        for i in range(2, level, -1):
            if not cur.is_valid():
                new_a = cls._new_table()
                cur.entry = ((new_a >> 2) & ~0x3FF) | int(EntryBits.Valid)
                tbl.entries[vpn[i]] = cur
            tbl = cls._get_table(cur.physical_addr())
            cur = tbl.entries[vpn[i-1]]
        entry = (ppn[2]<<28)|(ppn[1]<<19)|(ppn[0]<<10)|int(bits)|int(EntryBits.Valid)
        tbl.entries[vpn[level]] = PageTableEntry(entry)

    @classmethod
    def virt_to_phys(cls, root, vaddr):
        vpn = [(vaddr>>12)&0x1FF, (vaddr>>21)&0x1FF, (vaddr>>30)&0x1FF]
        if root not in cls._tables: return None
        tbl = cls._get_table(root); v = tbl.entries[vpn[2]]
        for i in range(2, -1, -1):
            if not v.is_valid(): return None
            if v.is_leaf():
                om = (1<<(12+i*9))-1
                return ((v.entry<<2) & ~om) | (vaddr & om)
            tbl = cls._get_table(v.physical_addr()); v = tbl.entries[vpn[i-1]]
        return None

    @classmethod
    def id_map_range(cls, root, start, end, bits):
        a = start & ~(PAGE_SIZE-1)
        n = ((end+PAGE_SIZE-1)&~(PAGE_SIZE-1)-a)//PAGE_SIZE
        for _ in range(n): cls.map(root, a, a, bits, 0); a += PAGE_SIZE

    @classmethod
    def snapshot(cls):
        return {"tables": len(cls._tables), "mappings": cls._mappings[-10:]}


# ── Hart ─────────────────────────────────────────────────────────────────────
class Hart:
    def __init__(self, hid):
        self.hart_id=hid; self.registers=[0]*32; self.pc=VIRT_DRAM
        self.satp=0; self.mstatus=0; self.mie=0; self.parked=(hid!=0)


# ── UART ──────────────────────────────────────────────────────────────────────
class Uart:
    def __init__(self, base=VIRT_UART0):
        self.base_addr=base; self._regs=bytearray(8)
        self._rx_fifo: deque = deque(); self._tx_buf=[]
        self._initialized=False

    def _wr(self, o, v): self._regs[o]=v&0xFF
    def _rd(self, o): return self._regs[o]

    def init(self):
        lcr=(1<<0)|(1<<1); self._wr(3,lcr); self._wr(2,1); self._wr(1,1)
        self._wr(3,lcr|(1<<7)); self._wr(0,592&0xFF); self._wr(1,592>>8)
        self._wr(3,lcr); self._initialized=True

    def put(self, b):
        """Send one byte through the simulated UART THR (0-255 only)."""
        if 0 <= b < 256:
            self._regs[0] = b & 0xFF
            self._tx_buf.append(b)
            _emit('uart_tx', {
                'char': chr(b) if 32 <= b < 127 else f'[{b:02x}]',
                'byte': b
            })
        # Unicode codepoints > 255 are silently dropped from the register;
        # kprint() emits them directly to the browser.

    def get(self):
        if self._rx_fifo:
            b = self._rx_fifo.popleft()
            if not self._rx_fifo:
                self._regs[5] &= ~1
            return b
        return None

    def inject(self, data: bytes):
        for b in data:
            self._rx_fifo.append(b)
        self._regs[5] |= 1

    def write_str(self, s: str):
        """Push only ASCII bytes through the simulated UART."""
        for b in s.encode('utf-8'):
            if b < 128:
                self.put(b)

UART0 = Uart()

def kprint(msg: str, end: str = '\n'):
    """Kernel print: send full Unicode to the browser console directly,
    and push ASCII-safe bytes through the simulated UART register."""
    full = msg + end
    # 1. Browser console - supports full Unicode including →, ✓, –, etc.
    _emit('console', {'text': full, 'level': 'info'})
    # 2. Simulated UART register - ASCII only (real UART is 8-bit anyway)
    for b in full.encode('utf-8'):
        if b < 128:
            UART0.put(b)


# ── PLIC ──────────────────────────────────────────────────────────────────────
class PLIC:
    def __init__(self):
        self._prio=[0]*64; self._enabled=0; self._threshold=0
        self._pending: deque = deque()
    def set_priority(self, s, p): self._prio[s]=p&7
    def enable(self, s): self._enabled|=(1<<s)
    def set_threshold(self, t): self._threshold=t&7
    def trigger(self, s):
        if (self._enabled>>s)&1 and self._prio[s]>self._threshold:
            self._pending.append(s)
    def next(self):
        if not self._pending: return None
        best=max(self._pending, key=lambda s: self._prio[s])
        self._pending.remove(best); return best
    def complete(self, s): pass

PLIC_CTRL = PLIC()


# ── TrapFrame & ProcessState ──────────────────────────────────────────────────
class ProcessState(IntEnum):
    Running=0; Sleeping=1; Waiting=2; Dead=3

@dataclass
class TrapFrame:
    regs: List[int] = field(default_factory=lambda:[0]*32)
    fregs: List[float] = field(default_factory=lambda:[0.0]*32)
    satp: int=0; trap_stack: int=0; hartid: int=0; pc: int=0; pid: int=0
    @property
    def a0(self): return self.regs[10]
    @a0.setter
    def a0(self,v): self.regs[10]=v
    @property
    def sp(self): return self.regs[2]
    @sp.setter
    def sp(self,v): self.regs[2]=v
    @classmethod
    def zero(cls): return cls()


# ── Processes ─────────────────────────────────────────────────────────────────
_next_pid=1
class Process:
    def __init__(self, func=None, is_kernel=True):
        global _next_pid
        self.pid=_next_pid; _next_pid+=1
        self.state=ProcessState.Waiting; self.frame=TrapFrame()
        self.frame.pid=self.pid; self.root_table_addr=MMU._new_table()
        self.stack_pages=alloc(STACK_PAGES); self.program_counter=0
        self._func=func; self.is_kernel=is_kernel; self._iteration=0
        if self.stack_pages: self.frame.sp=STACK_ADDR+PAGE_SIZE*STACK_PAGES
    def set_running(self): self.state=ProcessState.Running
    def set_dead(self): self.state=ProcessState.Dead
    def step(self):
        if not self._func or self.state==ProcessState.Dead: return False
        try: self._func(self)
        except StopIteration: self.set_dead(); return False
        return True
    def __repr__(self): return f"Process(pid={self.pid}, {self.state.name})"
    def __del__(self):
        try:
            if self.stack_pages: dealloc(self.stack_pages)
            if self.root_table_addr: MMU.unmap_root(self.root_table_addr)
        except: pass


# ── Scheduler ─────────────────────────────────────────────────────────────────
class Scheduler:
    def __init__(self): self._queue: deque = deque()
    def add(self, p): p.set_running(); self._queue.append(p)
    def remove_pid(self, pid): self._queue=deque(p for p in self._queue if p.pid!=pid)
    def tick(self):
        if not self._queue: return None
        self._queue.rotate(-1)
        for _ in range(len(self._queue)):
            proc=self._queue[0]
            if proc.state==ProcessState.Running: return proc
            self._queue.rotate(-1)
        return None
    def snapshot(self):
        return [{"pid":p.pid,"state":p.state.name,"kernel":p.is_kernel}
                for p in self._queue]

SCHEDULER = Scheduler()


# ── SyscallHandler ────────────────────────────────────────────────────────────
class SyscallHandler:
    def __init__(self): pass
    def dispatch(self, mepc, frame):
        sno=frame.regs[10]
        names={93:"exit",64:"write",63:"read",1:"print"}
        name=names.get(sno, f"unknown#{sno}")
        _emit('syscall', {'number': sno, 'name': name, 'pid': frame.pid})
        kprint(f"  [SYSCALL] {name}() from pid={frame.pid}")
        if sno==93: SCHEDULER.remove_pid(frame.pid)
        return mepc

SYSCALL = SyscallHandler()


# ── VirtIO Block ───────────────────────────────────────────────────────────────
VIRTIO_MAGIC=0x74726976
class BlockDevice:
    SECTOR=512
    def __init__(self, img=None):
        self._disk=bytearray(img or bytes(8*1024*1024))
        self._mmio={}; self._initialized=False
    def setup(self):
        self._mmio[0x000]=VIRTIO_MAGIC; self._mmio[0x008]=2
        self._mmio[0x010]=0xFFFFFFFF; self._mmio[0x034]=16
        self._initialized=True
        _emit('virtio', {'status':'ready','sectors':len(self._disk)//self.SECTOR})
    def read(self, offset, size):
        e=min(offset+size,len(self._disk)); return bytes(self._disk[offset:e])
    def write(self, offset, data):
        e=min(offset+len(data),len(self._disk)); self._disk[offset:e]=data[:e-offset]


# ── Minix3 FS ─────────────────────────────────────────────────────────────────
@dataclass
class Superblock:
    ninodes:int; imap_blocks:int; zmap_blocks:int; first_data_zone:int
    block_size:int; magic:int
    def is_valid(self): return self.magic==MINIX3_MAGIC

@dataclass
class Inode:
    mode:int; nlinks:int; uid:int; gid:int; size:int
    atime:int; mtime:int; ctime:int; zones:List[int]
    @classmethod
    def from_bytes(cls, d):
        m,nl,u,g,sz,at,mt,ct,*z=struct.unpack_from('<HHHHI III 10I',d)
        return cls(m,nl,u,g,sz,at,mt,ct,list(z))
    def is_dir(self): return (self.mode&0xF000)==0x4000
    def is_regular(self): return (self.mode&0xF000)==0x8000
    def perms(self):
        p=''.join(c if self.mode&b else '-' for b,c in
            [(0o400,'r'),(0o200,'w'),(0o100,'x'),(0o040,'r'),
             (0o020,'w'),(0o010,'x'),(0o004,'r'),(0o002,'w'),(0o001,'x')])
        return ('d' if self.is_dir() else '-')+p

@dataclass
class DirEntry: inode:int; name:str

class MinixFS:
    def __init__(self, dev):
        self.dev=dev; self.sb=self._read_sb()
    def _read_block(self, n): return self.dev.read(n*MINIX3_BLOCK_SIZE,MINIX3_BLOCK_SIZE)
    def _read_sb(self):
        d=self.dev.read(1024,64)
        if len(d)<32: raise ValueError("bad sb")
        t=struct.unpack_from('<IHHHHHHI I HHHb',d)
        return Superblock(t[0],t[2],t[3],t[4],t[11],t[9])
    def _inode_offset(self, n):
        tbl=(2+self.sb.imap_blocks+self.sb.zmap_blocks)*MINIX3_BLOCK_SIZE
        return tbl+(n-1)*MINIX3_INODE_SIZE
    def read_inode_meta(self, n):
        return Inode.from_bytes(self.dev.read(self._inode_offset(n),MINIX3_INODE_SIZE))
    def read_inode(self, n, size=-1, offset=0):
        ino=self.read_inode_meta(n); result=bytearray()
        if size<0: size=ino.size
        for z in ino.zones[:7]:
            if z and len(result)<size: result.extend(self._read_block(z))
        if ino.zones[7] and len(result)<size:
            ib=self._read_block(ino.zones[7])
            for i in range(MINIX3_BLOCK_SIZE//4):
                z=struct.unpack_from('<I',ib,i*4)[0]
                if z and len(result)<size: result.extend(self._read_block(z))
        return bytes(result[:size])
    def list_dir(self, n):
        ino=self.read_inode_meta(n)
        if not ino.is_dir(): raise ValueError(f"not a dir: {n}")
        raw=self.read_inode(n); entries=[]
        for i in range(0,len(raw),64):
            c=raw[i:i+64]
            if len(c)<64: break
            inum=struct.unpack_from('<I',c,0)[0]
            if inum==0: continue
            name=c[4:64].rstrip(b'\x00').decode('utf-8','replace')
            entries.append(DirEntry(inum,name))
        return entries


# ── Fake disk image ──────────────────────────────────────────────────────────
def make_fake_disk():
    d=bytearray(8*1024*1024)
    sb=struct.pack('<IHHHHHHI I HHHb',16,0,1,1,5,0,0,2147483647,8192,MINIX3_MAGIC,0,1024,3)
    d[1024:1024+len(sb)]=sb
    it=(2+1+1)*MINIX3_BLOCK_SIZE
    d[it:it+64]=struct.pack('<HHHHI III 10I',0x41ED,2,0,0,192,0,0,0,5,0,0,0,0,0,0,0,0,0)
    fc=b"Hello, this is my first file on Minix3 filesystem!\n\x00"
    d[it+64:it+128]=struct.pack('<HHHHI III 10I',0x81A4,1,0,0,len(fc),0,0,0,6,0,0,0,0,0,0,0,0,0)
    db=5*MINIX3_BLOCK_SIZE
    d[db:db+4]=struct.pack('<I',1);  d[db+4:db+64]=b'.\x00'+b'\x00'*58
    d[db+64:db+68]=struct.pack('<I',1); d[db+68:db+128]=b'..\x00'+b'\x00'*57
    d[db+128:db+132]=struct.pack('<I',2); d[db+132:db+192]=b'hello.txt\x00'+b'\x00'*50
    fb=6*MINIX3_BLOCK_SIZE; d[fb:fb+len(fc)]=fc
    return bytes(d)


# ============================================================================
# ── QEMU Manager ────────────────────────────────────────────────────────────
# ============================================================================

class QEMUManager:
    """
    Spawns qemu-system-riscv64 and streams its UART output to the browser.

    Strategy
    --------
    • Build a correct RISC-V flat binary at runtime using a tiny Python
      assembler (no cross-compiler required).
    • Use  -serial file:/tmp/qemu_uart_NNN.txt  so QEMU writes UART bytes
      to a file we can tail from a background thread.  This is the most
      portable and reliable approach — pipes buffer, sockets need timing.
    • Stream every new line from the file to all browser clients via SocketIO.
    """

    # ── Tiny RISC-V assembler ────────────────────────────────────────────
    @staticmethod
    def _lui(rd: int, imm20: int) -> int:
        """LUI rd, imm20  (imm20 is the upper 20-bit field, not pre-shifted)."""
        return ((imm20 & 0xFFFFF) << 12) | ((rd & 0x1F) << 7) | 0x37

    @staticmethod
    def _addi(rd: int, rs1: int, imm12: int) -> int:
        return ((imm12 & 0xFFF) << 20) | ((rs1 & 0x1F) << 15) | ((rd & 0x1F) << 7) | 0x13

    @staticmethod
    def _sb(rs2: int, rs1: int, imm: int = 0) -> int:
        """SB rs2, imm(rs1)"""
        return (((imm >> 5) & 0x7F) << 25) | ((rs2 & 0x1F) << 20) |                ((rs1 & 0x1F) << 15) | ((imm & 0x1F) << 7) | 0x23

    @staticmethod
    def _wfi() -> int: return 0x10500073

    @staticmethod
    def _j_minus4() -> int:
        """JAL x0, -4  (infinite loop back to wfi)"""
        return 0xFFFFF06F

    @classmethod
    def _build_stub(cls) -> bytes:
        """
        Assemble a small RISC-V machine-mode program that:
          1. Loads the UART base address (0x10000000) into a0
          2. Writes a boot message character by character to UART THR
          3. Loops forever with WFI (wait-for-interrupt)
        """
        a0, t0, x0 = 10, 5, 0
        UART_UPPER20 = 0x10000   # lui a0, 0x10000 → a0 = 0x10000000

        msg = (
            "\r\n"
            "=== qemu-system-riscv64  (RISC-V virt machine) ===\r\n"
            "\r\n"
            "Memory map (from QEMU virt.c):\r\n"
            "  CLINT   0x02000000\r\n"
            "  PLIC    0x0c000000\r\n"
            "  UART0   0x10000000  <-- you are here\r\n"
            "  VIRTIO  0x10001000\r\n"
            "  DRAM    0x80000000  (128 MiB)\r\n"
            "\r\n"
            "Hart #0 booting in machine mode...\r\n"
            "satp=0 (MMU off, physical addressing)\r\n"
            "mstatus.MPP=11 (machine mode)\r\n"
            "Entering kmain()...\r\n"
            "\r\n"
            "[RISC-V OS stub] All done. Hart parked in wfi loop.\r\n"
        )

        words = [cls._lui(a0, UART_UPPER20)]
        for ch in msg:
            words.append(cls._addi(t0, x0, ord(ch)))
            words.append(cls._sb(t0, a0, 0))
        words += [cls._wfi(), cls._j_minus4()]

        return b"".join(struct.pack("<I", w) for w in words)

    # ── Candidate QEMU binary paths ───────────────────────────────────────
    QEMU_CANDIDATES = [
        "/usr/bin/qemu-system-riscv64",
        "/usr/local/bin/qemu-system-riscv64",
        "/opt/homebrew/bin/qemu-system-riscv64",
        "/usr/local/homebrew/bin/qemu-system-riscv64",
    ]

    @classmethod
    def _find_qemu(cls) -> Optional[str]:
        import shutil
        for c in cls.QEMU_CANDIDATES:
            if os.path.isfile(c) and os.access(c, os.X_OK):
                return c
        return shutil.which("qemu-system-riscv64")

    # ── Instance ──────────────────────────────────────────────────────────
    def __init__(self):
        self._proc:   Optional[subprocess.Popen] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._stub_path  = "/tmp/riscv_stub.bin"
        self._uart_path  = "/tmp/riscv_uart.txt"   # UART serial output file

    def start(self):
        if self._proc and self._proc.poll() is None:
            _emit("qemu", {"status": "already_running"})
            return

        qemu_bin = self._find_qemu()
        if qemu_bin is None:
            _emit("qemu", {"status": "error",
                           "msg": "qemu-system-riscv64 not found — "
                                  "install with: sudo apt install qemu-system-misc"})
            return

        # Write stub binary
        stub = self._build_stub()
        with open(self._stub_path, "wb") as f:
            f.write(stub)

        # Wipe stale UART capture file
        try: os.remove(self._uart_path)
        except FileNotFoundError: pass
        open(self._uart_path, "w").close()   # create empty

        env = os.environ.copy()
        env["PATH"] = "/usr/bin:/usr/local/bin:/bin:" + env.get("PATH", "")

        cmd = [
            qemu_bin,
            "-machine", "virt",
            "-cpu",     "rv64",
            "-smp",     "1",
            "-m",       "128M",
            "-nographic",
            "-serial",  f"file:{self._uart_path}",  # UART → file (reliable)
            "-bios",    "none",
            "-kernel",  self._stub_path,
        ]

        _emit("qemu", {"status": "starting",
                       "cmd": f"{qemu_bin} -machine virt -cpu rv64 -m 128M "
                              f"-serial file:{self._uart_path} -bios none "
                              f"-kernel {self._stub_path}"})
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                env=env,
            )
            self._running = True
            self._thread = threading.Thread(target=self._tail_uart, daemon=True)
            self._thread.start()
            _emit("qemu", {"status": "running", "pid": self._proc.pid,
                           "binary": qemu_bin, "stub_bytes": len(stub)})
        except FileNotFoundError:
            _emit("qemu", {"status": "error", "msg": f"Cannot execute: {qemu_bin}"})
        except PermissionError:
            _emit("qemu", {"status": "error", "msg": f"Permission denied: {qemu_bin}"})
        except Exception as e:
            _emit("qemu", {"status": "error", "msg": str(e)})

    def _tail_uart(self):
        """Read new bytes from the UART output file and stream to the browser."""
        pos = 0
        buf = ""
        deadline = time.time() + 15          # give up after 15 s if no output
        while self._running:
            if self._proc and self._proc.poll() is not None:
                break                        # QEMU exited
            try:
                with open(self._uart_path, "r", errors="replace") as f:
                    f.seek(pos)
                    chunk = f.read(512)
                if chunk:
                    deadline = time.time() + 15   # reset watchdog
                    pos += len(chunk.encode())
                    buf += chunk
                    # Emit complete lines; hold partial lines in buf
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        _emit("qemu_output", {"text": line.rstrip("\r") + "\n"})
                else:
                    if time.time() > deadline:
                        _emit("qemu_output", {"text": "[watchdog] no UART output in 15 s\n"})
                        break
            except Exception:
                pass
            time.sleep(0.05)

        # Flush remainder
        if buf.strip():
            _emit("qemu_output", {"text": buf})
        _emit("qemu", {"status": "stopped"})

    def stop(self):
        self._running = False
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try: self._proc.wait(timeout=3)
            except: self._proc.kill()
        _emit("qemu", {"status": "stopped"})

    def info(self):
        running = bool(self._proc and self._proc.poll() is None)
        return {
            "available":  True,
            "running":    running,
            "pid":        self._proc.pid if running else None,
            "stub":       self._stub_path,
            "uart_log":   self._uart_path,
            "qemu_bin":   self._find_qemu(),
        }

QEMU_MGR = QEMUManager()


# ============================================================================
# ── OS Simulation runner ─────────────────────────────────────────────────────
# ============================================================================

SIM_STATE = {
    "chapter": 0,
    "running": False,
    "boot_done": False,
    "hart": None,
}

def _sleep(s): time.sleep(s)

def run_chapter(ch: int):
    """Run a single chapter and push events to the browser."""
    SIM_STATE["chapter"] = ch

    if ch == 1:
        _emit('chapter', {'n':1,'title':'Boot Process'})
        _emit('highlight_node', {'node':'hart'})
        kprint("=== CHAPTER 1: BOOT ===")
        hart = Hart(0)
        SIM_STATE["hart"] = hart
        for i in range(1,4):
            kprint(f"  Hart #{i}: parked (wfi)")
            _emit('hart_status', {'id':i,'parked':True})
            _sleep(0.2)
        hart.satp = 0
        kprint("  Hart #0: boot hart selected")
        kprint(f"  satp = 0  (MMU off)")
        kprint(f"  Clearing BSS 0x{BSS_START:x}–0x{BSS_END:x}")
        PHYS_MEM.zero_range(BSS_START, BSS_END)
        hart.mstatus = (0b11<<11)|(1<<7)|(1<<3)
        hart.mie = (1<<3)|(1<<7)|(1<<11)
        kprint(f"  mstatus = 0x{hart.mstatus:x}")
        SIM_STATE["boot_done"] = True
        _emit('boot_done', {})

    elif ch == 2:
        _emit('chapter', {'n':2,'title':'UART Driver'})
        _emit('highlight_node', {'node':'uart'})
        kprint("=== CHAPTER 2: UART ===")
        UART0.init()
        kprint("  UART0 init @ 0x10000000 (NS16550a)")
        kprint("  LCR: 8-bit word length set")
        kprint("  FCR: FIFO enabled")
        kprint("  IER: RX interrupt enabled")
        kprint("  Hello from the kernel UART driver!")
        UART0.inject(b"test\r")
        kprint("  [RX] Simulated input: 'test'")
        out = []
        while True:
            c = UART0.get()
            if c is None: break
            if c not in (10,13): out.append(chr(c))
        kprint(f"  [TX] Echo: {''.join(out)}")
        _emit('highlight_node', {'node':'uart'})

    elif ch == 3:
        _emit('chapter', {'n':3,'title':'Page Allocator + MMU'})
        _emit('highlight_node', {'node':'mmu'})
        kprint("=== CHAPTER 3.1: PAGE ALLOCATOR ===")
        p1 = alloc(1); p2 = alloc(4); p3 = alloc(16)
        kprint(f"  alloc(1)  → 0x{p1:x}")
        kprint(f"  alloc(4)  → 0x{p2:x}")
        kprint(f"  alloc(16) → 0x{p3:x}")
        _emit('page_table', PAGE_ALLOC.snapshot())
        dealloc(p2); kprint(f"  dealloc(p2) → freed 4 pages")
        _emit('page_table', PAGE_ALLOC.snapshot())
        kprint("=== CHAPTER 3.2: MMU / SV39 PAGING ===")
        root = MMU._new_table()
        vaddr = 0x2000_0000; paddr = 0x8010_0000
        MMU.map(root, vaddr, paddr, int(EntryBits.ReadWrite), 0)
        result = MMU.virt_to_phys(root, vaddr)
        kprint(f"  map 0x{vaddr:x} → 0x{paddr:x}")
        kprint(f"  virt_to_phys(0x{vaddr:x}) = 0x{result:x}" if result else "  page fault!")
        MMU.id_map_range(root, VIRT_UART0, VIRT_UART0+0x100, int(EntryBits.ReadWrite))
        kprint(f"  Identity-mapped UART MMIO")
        _emit('mmu_state', MMU.snapshot())

    elif ch == 4:
        _emit('chapter', {'n':4,'title':'Trap Handling'})
        _emit('highlight_node', {'node':'trap'})
        kprint("=== CHAPTER 4: TRAP HANDLING ===")
        frame = TrapFrame(hartid=0)
        causes = [
            ((1<<63)|7, "Machine Timer (async)"),
            (13,        "Load Page Fault  (sync)"),
            (8,         "User ecall       (sync)"),
        ]
        for cause, name in causes:
            is_async = bool(cause>>63&1)
            cnum = cause & 0xFFF
            kprint(f"  Trap: {name}  cause=0x{cause&0xFFFF:x}")
            _emit('trap_event', {'cause': cause & 0xFFFF, 'name': name,
                                 'async': is_async, 'num': cnum})
            _sleep(0.3)

    elif ch == 5:
        _emit('chapter', {'n':5,'title':'PLIC'})
        _emit('highlight_node', {'node':'plic'})
        kprint("=== CHAPTER 5: PLIC ===")
        PLIC_CTRL.set_threshold(0)
        PLIC_CTRL.enable(10); PLIC_CTRL.set_priority(10,1)
        PLIC_CTRL.enable(1);  PLIC_CTRL.set_priority(1,2)
        kprint("  PLIC threshold=0, UART(10) p=1, VirtIO(1) p=2")
        UART0.inject(b"A")
        PLIC_CTRL.trigger(10)
        intr = PLIC_CTRL.next()
        kprint(f"  Claimed interrupt: source={intr} (UART)")
        _emit('plic_event', {'source': intr, 'device': 'UART0'})
        PLIC_CTRL.complete(intr)
        kprint("  Completed interrupt")

    elif ch == 6:
        _emit('chapter', {'n':6,'title':'Processes'})
        _emit('highlight_node', {'node':'proc'})
        kprint("=== CHAPTERS 6+8: PROCESSES & SCHEDULER ===")

        def init_fn(p):
            p._iteration += 1
            kprint(f"  [init pid={p.pid}] tick #{p._iteration}")
            if p._iteration >= 4: raise StopIteration

        def worker_fn(p):
            p._iteration += 1
            kprint(f"  [worker pid={p.pid}] computing #{p._iteration}")
            if p._iteration >= 3: raise StopIteration

        p_init   = Process(func=init_fn,   is_kernel=True)
        p_worker = Process(func=worker_fn, is_kernel=True)
        SCHEDULER.add(p_init); SCHEDULER.add(p_worker)
        kprint(f"  Created pid={p_init.pid} (init), pid={p_worker.pid} (worker)")
        _emit('scheduler', SCHEDULER.snapshot())
        for tick in range(5):
            kprint(f"  --- Tick {tick+1} ---")
            for _ in range(len(SCHEDULER._queue)):
                proc = SCHEDULER.tick()
                if proc:
                    proc.step()
                    if proc.state == ProcessState.Dead:
                        kprint(f"  pid={proc.pid} exited")
                        SCHEDULER.remove_pid(proc.pid)
            _emit('scheduler', SCHEDULER.snapshot())
            _sleep(0.2)

    elif ch == 7:
        _emit('chapter', {'n':7,'title':'System Calls'})
        _emit('highlight_node', {'node':'syscall'})
        kprint("=== CHAPTER 7: SYSTEM CALLS ===")
        for sno, name in [(1,"print"),(64,"write"),(93,"exit")]:
            fr = TrapFrame(pid=42); fr.regs[10]=sno; fr.regs[11]=0
            SYSCALL.dispatch(0x2000_0000, fr)
            _sleep(0.15)

    elif ch == 8:
        _emit('chapter', {'n':8,'title':'VirtIO Block Driver'})
        _emit('highlight_node', {'node':'virtio'})
        kprint("=== CHAPTER 9: VIRTIO BLOCK DRIVER ===")
        disk = make_fake_disk()
        bd = BlockDevice(disk)
        bd.setup()
        data = bd.read(0,64)
        kprint(f"  Sector 0 (first 16 bytes): {data[:16].hex(' ')}")
        kprint(f"  Disk size: {len(disk)//1024} KiB")
        data2 = bd.read(5*MINIX3_BLOCK_SIZE, 64)
        kprint(f"  Block 5 (dir block) first 8: {data2[:8].hex(' ')}")

    elif ch == 9:
        _emit('chapter', {'n':9,'title':'Minix3 Filesystem'})
        _emit('highlight_node', {'node':'fs'})
        kprint("=== CHAPTER 10: MINIX3 FILESYSTEM ===")
        disk = make_fake_disk()
        bd = BlockDevice(disk); bd.setup()
        fs = MinixFS(bd)
        kprint(f"  Magic=0x{fs.sb.magic:x} ✓  ninodes={fs.sb.ninodes}  bsize={fs.sb.block_size}")
        entries = fs.list_dir(1)
        kprint("  Root directory:")
        dir_data = []
        for e in entries:
            meta = fs.read_inode_meta(e.inode)
            kprint(f"    [{e.inode}] {meta.perms()}  {e.name}  ({meta.size}B)")
            dir_data.append({"inode":e.inode,"name":e.name,"perms":meta.perms(),"size":meta.size})
        _emit('fs_dir', {'entries': dir_data})
        content = fs.read_inode(2)
        kprint(f"  hello.txt: {content.decode('utf-8','replace').strip()}")

    elif ch == 10:
        _emit('chapter', {'n':10,'title':'ELF Loader'})
        _emit('highlight_node', {'node':'elf'})
        kprint("=== CHAPTER 11: ELF LOADER ===")
        # Build minimal ELF
        ident=bytes([0x7F,0x45,0x4C,0x46,2,1,1,0,0,0,0,0,0,0,0,0])
        phoff=64; entry=PROCESS_STARTING_ADDR
        code=b'\x13\x00\x00\x00'*8  # 8 NOPs
        elf_hdr=struct.pack('<16s HHI QQQI HHHHHH',
            ident,ELF_TYPE_EXEC,ELF_MACHINE_RISCV,1,
            entry,phoff,0,0,64,56,1,64,0,0)
        ph=struct.pack('<II QQQQQQ',ELF_PT_LOAD,ELF_PF_READ|ELF_PF_EXECUTE,
            phoff+56,entry,0,len(code),len(code),0x1000)
        elf=elf_hdr+ph+code
        kprint(f"  ELF size: {len(elf)} bytes")
        kprint(f"  Magic: 0x{struct.unpack_from('<I',elf,0)[0]:08x}")
        kprint(f"  Machine: 0x{struct.unpack_from('<H',elf,18)[0]:02x} (RISC-V)")
        proc = Process(is_kernel=False)
        proc.frame.pc = entry
        proc.frame.sp = STACK_ADDR + STACK_PAGES * PAGE_SIZE
        proc.frame.satp = (8<<60)|(proc.pid<<44)|(proc.root_table_addr>>12)
        kprint(f"  Loaded: pid={proc.pid}")
        kprint(f"  PC=0x{proc.frame.pc:x}  SP=0x{proc.frame.sp:x}")
        kprint(f"  SATP=0x{proc.frame.satp:016x}")
        _emit('elf_loaded', {'pid': proc.pid, 'entry': f"0x{entry:x}",
                              'sp': f"0x{proc.frame.sp:x}"})

    _emit('chapter_done', {'n': ch})


def run_all_chapters():
    for ch in range(1, 11):
        if not SIM_STATE["running"]: break
        run_chapter(ch)
        _sleep(0.4)
    SIM_STATE["running"] = False
    kprint("\n=== All chapters complete! ===")
    _emit('sim_done', {})


# ============================================================================
# ── Flask + SocketIO App ──────────────────────────────────────────────────────
# ============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'riscv-os-edu'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

# Background thread: drains _event_q and emits to all connected clients
def _event_dispatcher():
    while True:
        try:
            event, data = _event_q.get(timeout=0.05)
            socketio.emit(event, data)
        except queue.Empty:
            pass
        except Exception:
            pass

threading.Thread(target=_event_dispatcher, daemon=True).start()

# ── HTML template ─────────────────────────────────────────────────────────────
HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RISC-V OS Simulator</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.min.js"></script>
<style>
  :root {
    --bg:#0d1117; --panel:#161b22; --border:#30363d; --accent:#58a6ff;
    --green:#3fb950; --yellow:#d29922; --red:#f85149; --purple:#bc8cff;
    --orange:#ffa657; --text:#c9d1d9; --muted:#8b949e;
    --node-idle:#21262d; --node-active:#1f3a5f; --node-border:#388bfd;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:'Consolas','Courier New',monospace;
       font-size:13px;height:100vh;display:flex;flex-direction:column;overflow:hidden}
  /* ── Header ── */
  header{background:var(--panel);border-bottom:1px solid var(--border);
         padding:8px 16px;display:flex;align-items:center;gap:16px;flex-shrink:0}
  header h1{color:var(--accent);font-size:16px;letter-spacing:1px}
  header .badge{background:#21262d;border:1px solid var(--border);
                border-radius:4px;padding:2px 8px;font-size:11px;color:var(--muted)}
  header .badge.on{border-color:var(--green);color:var(--green)}
  /* ── Main layout ── */
  .main{display:grid;grid-template-columns:1fr 340px;grid-template-rows:1fr;
        flex:1;overflow:hidden;gap:1px;background:var(--border)}
  /* ── Left: diagram + controls ── */
  .left{display:flex;flex-direction:column;background:var(--bg);overflow:hidden}
  /* ── Diagram ── */
  .diagram-wrap{flex:1;min-height:0;position:relative;overflow:hidden}
  svg.arch{width:100%;height:100%}
  /* nodes */
  .node rect{fill:var(--node-idle);stroke:var(--border);stroke-width:1.5;
             rx:6;transition:fill 0.3s,stroke 0.3s,filter 0.3s}
  .node.active rect{fill:var(--node-active);stroke:var(--node-border);
                    filter:drop-shadow(0 0 6px #388bfd88)}
  .node text{fill:var(--text);font-size:11px;font-family:monospace;
             pointer-events:none;dominant-baseline:middle;text-anchor:middle}
  .node .icon{font-size:18px}
  .node .label{font-size:11px;font-weight:bold}
  .node .sub{font-size:9px;fill:var(--muted)}
  /* arrows */
  .arrow{stroke:var(--border);stroke-width:1.5;fill:none;
         transition:stroke 0.3s,stroke-width 0.3s;marker-end:url(#arr)}
  .arrow.active{stroke:var(--accent);stroke-width:2.5;marker-end:url(#arr-active)}
  .arrow.pulse{stroke:var(--green);stroke-width:2.5;
               animation:pulse-line 1s ease-out forwards;marker-end:url(#arr-pulse)}
  @keyframes pulse-line{0%{stroke-opacity:1}100%{stroke-opacity:0.2}}
  /* data packet animation */
  .packet{fill:var(--accent);r:5;opacity:0}
  /* ── Chapter buttons ── */
  .controls{background:var(--panel);border-top:1px solid var(--border);
            padding:8px 12px;flex-shrink:0}
  .ch-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:4px;margin-bottom:6px}
  .ch-btn{background:var(--node-idle);border:1px solid var(--border);color:var(--muted);
          padding:4px 2px;border-radius:4px;cursor:pointer;font-size:10px;
          font-family:monospace;transition:all 0.2s;text-align:center}
  .ch-btn:hover{border-color:var(--accent);color:var(--text)}
  .ch-btn.done{border-color:var(--green);color:var(--green)}
  .ch-btn.running{border-color:var(--yellow);color:var(--yellow);
                  animation:blink 1s infinite}
  @keyframes blink{50%{opacity:0.5}}
  .run-row{display:flex;gap:6px}
  .btn{border:1px solid var(--border);background:var(--node-idle);color:var(--text);
       padding:5px 12px;border-radius:4px;cursor:pointer;font-family:monospace;
       font-size:11px;transition:all 0.2s}
  .btn:hover{border-color:var(--accent);color:var(--accent)}
  .btn.primary{border-color:var(--accent);color:var(--accent)}
  .btn.danger{border-color:var(--red);color:var(--red)}
  .btn.success{border-color:var(--green);color:var(--green)}
  /* ── Right: console + tabs ── */
  .right{background:var(--panel);display:flex;flex-direction:column;
         border-left:1px solid var(--border);overflow:hidden}
  .tabs{display:flex;border-bottom:1px solid var(--border);flex-shrink:0}
  .tab{padding:6px 12px;cursor:pointer;font-size:11px;color:var(--muted);
       border-bottom:2px solid transparent;transition:all 0.2s}
  .tab.active{color:var(--accent);border-bottom-color:var(--accent)}
  .tab-pane{display:none;flex:1;overflow-y:auto;padding:8px}
  .tab-pane.active{display:flex;flex-direction:column}
  /* console */
  #console-out{flex:1;overflow-y:auto;white-space:pre-wrap;word-break:break-all;
               font-size:11px;line-height:1.5;padding:4px}
  .log-info{color:var(--text)} .log-warn{color:var(--yellow)}
  .log-err{color:var(--red)}   .log-ok{color:var(--green)}
  .log-uart{color:var(--purple)} .log-sys{color:var(--orange)}
  /* status panel */
  .status-grid{display:grid;gap:6px}
  .stat-card{background:var(--node-idle);border:1px solid var(--border);
             border-radius:6px;padding:8px}
  .stat-title{color:var(--muted);font-size:10px;margin-bottom:4px;text-transform:uppercase}
  .stat-val{color:var(--text);font-size:11px}
  .stat-val span{color:var(--accent)}
  .badge-row{display:flex;flex-wrap:wrap;gap:4px;margin-top:4px}
  .chip{background:#21262d;border:1px solid var(--border);border-radius:10px;
        padding:1px 6px;font-size:10px}
  .chip.run{border-color:var(--green);color:var(--green)}
  .chip.wait{border-color:var(--yellow);color:var(--yellow)}
  .chip.dead{border-color:var(--red);color:var(--red)}
  /* QEMU panel */
  #qemu-out{flex:1;overflow-y:auto;white-space:pre-wrap;font-size:11px;
            line-height:1.5;color:var(--green);padding:4px;
            background:#0a1a0a;border-radius:4px}
  .qemu-controls{display:flex;gap:6px;margin-bottom:6px;flex-shrink:0}
  /* page table viz */
  .alloc-bar{display:flex;height:14px;border-radius:3px;overflow:hidden;
             margin:4px 0;background:#21262d;border:1px solid var(--border)}
  .alloc-used{background:var(--accent);transition:width 0.5s}
  .alloc-free{background:transparent}
  /* scroll */
  ::-webkit-scrollbar{width:6px;height:6px}
  ::-webkit-scrollbar-track{background:var(--bg)}
  ::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
  /* status bar */
  .statusbar{background:var(--panel);border-top:1px solid var(--border);
             padding:3px 12px;display:flex;gap:16px;font-size:10px;
             color:var(--muted);flex-shrink:0}
  .statusbar span{display:flex;align-items:center;gap:4px}
  .dot{width:6px;height:6px;border-radius:50%;background:var(--muted)}
  .dot.green{background:var(--green)} .dot.yellow{background:var(--yellow)}
  .dot.red{background:var(--red)}
</style>
</head>
<body>

<header>
  <h1>⚙ RISC-V OS Simulator</h1>
  <span class="badge" id="sim-badge">IDLE</span>
  <span class="badge" id="qemu-badge">QEMU: OFF</span>
  <span class="badge on">Python 3</span>
  <span class="badge on">QEMU 8.2</span>
</header>

<div class="main">

  <!-- ── LEFT: Architecture Diagram + Controls ── -->
  <div class="left">
    <div class="diagram-wrap">
      <svg class="arch" id="arch-svg" viewBox="0 0 700 420" preserveAspectRatio="xMidYMid meet">
        <defs>
          <marker id="arr" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0,8 3,0 6" fill="#30363d"/>
          </marker>
          <marker id="arr-active" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0,8 3,0 6" fill="#58a6ff"/>
          </marker>
          <marker id="arr-pulse" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0,8 3,0 6" fill="#3fb950"/>
          </marker>
          <filter id="glow">
            <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
            <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
          </filter>
        </defs>

        <!-- ── Arrows (drawn first, behind nodes) ── -->
        <!-- Hart → UART -->
        <path class="arrow" id="arr-hart-uart" d="M 175,90 L 280,90"/>
        <!-- Hart → MMU -->
        <path class="arrow" id="arr-hart-mmu" d="M 150,110 L 150,175"/>
        <!-- UART → PLIC -->
        <path class="arrow" id="arr-uart-plic" d="M 380,90 Q 420,90 420,130"/>
        <!-- MMU → Proc -->
        <path class="arrow" id="arr-mmu-proc" d="M 210,210 L 280,210"/>
        <!-- Proc → Syscall -->
        <path class="arrow" id="arr-proc-sys" d="M 380,210 L 450,210"/>
        <!-- PLIC → Trap -->
        <path class="arrow" id="arr-plic-trap" d="M 420,190 L 420,260"/>
        <!-- Trap → Proc -->
        <path class="arrow" id="arr-trap-proc" d="M 395,290 L 380,230"/>
        <!-- Proc → Sched -->
        <path class="arrow" id="arr-proc-sched" d="M 330,240 L 330,295"/>
        <!-- VirtIO → FS -->
        <path class="arrow" id="arr-virtio-fs" d="M 210,330 L 280,330"/>
        <!-- FS → ELF -->
        <path class="arrow" id="arr-fs-elf" d="M 380,330 L 450,330"/>
        <!-- ELF → Proc -->
        <path class="arrow" id="arr-elf-proc" d="M 510,300 L 430,240"/>
        <!-- Sched → Hart -->
        <path class="arrow" id="arr-sched-hart" d="M 290,320 Q 140,350 140,130"/>
        <!-- CLINT → Trap -->
        <path class="arrow" id="arr-clint-trap" d="M 560,210 L 480,285"/>

        <!-- ── QEMU boundary box ── -->
        <rect x="10" y="10" width="680" height="395" rx="8"
              fill="none" stroke="#1f3a1a" stroke-width="1.5" stroke-dasharray="6,3"/>
        <text x="20" y="26" fill="#2a5c1a" font-size="10" font-family="monospace">
          qemu-system-riscv64  -machine virt  -m 128M
        </text>

        <!-- ── Nodes ── -->
        <!-- Hart -->
        <g class="node" id="node-hart" transform="translate(60,60)"
           onclick="stepChapter(1)" style="cursor:pointer">
          <rect x="0" y="0" width="110" height="60"/>
          <text class="icon" x="55" y="20">🧠</text>
          <text class="label" x="55" y="38">HART #0</text>
          <text class="sub" x="55" y="52">mhartid=0  M-mode</text>
        </g>
        <!-- UART -->
        <g class="node" id="node-uart" transform="translate(280,60)"
           onclick="stepChapter(2)" style="cursor:pointer">
          <rect x="0" y="0" width="110" height="60"/>
          <text class="icon" x="55" y="20">📡</text>
          <text class="label" x="55" y="38">UART0</text>
          <text class="sub" x="55" y="52">NS16550a  0x10000000</text>
        </g>
        <!-- MMU -->
        <g class="node" id="node-mmu" transform="translate(60,175)"
           onclick="stepChapter(3)" style="cursor:pointer">
          <rect x="0" y="0" width="110" height="60"/>
          <text class="icon" x="55" y="20">🗺</text>
          <text class="label" x="55" y="38">MMU / SV39</text>
          <text class="sub" x="55" y="52">3-level page tables</text>
        </g>
        <!-- PLIC -->
        <g class="node" id="node-plic" transform="translate(365,130)"
           onclick="stepChapter(5)" style="cursor:pointer">
          <rect x="0" y="0" width="110" height="60"/>
          <text class="icon" x="55" y="20">🔀</text>
          <text class="label" x="55" y="38">PLIC</text>
          <text class="sub" x="55" y="52">0x0c000000</text>
        </g>
        <!-- Trap -->
        <g class="node" id="node-trap" transform="translate(365,255)"
           onclick="stepChapter(4)" style="cursor:pointer">
          <rect x="0" y="0" width="110" height="60"/>
          <text class="icon" x="55" y="20">⚡</text>
          <text class="label" x="55" y="38">Trap Handler</text>
          <text class="sub" x="55" y="52">mtvec  m_trap()</text>
        </g>
        <!-- Process -->
        <g class="node" id="node-proc" transform="translate(280,175)"
           onclick="stepChapter(6)" style="cursor:pointer">
          <rect x="0" y="0" width="110" height="60"/>
          <text class="icon" x="55" y="20">🔄</text>
          <text class="label" x="55" y="38">Process</text>
          <text class="sub" x="55" y="52">TrapFrame  pid</text>
        </g>
        <!-- Syscall -->
        <g class="node" id="node-syscall" transform="translate(450,175)"
           onclick="stepChapter(7)" style="cursor:pointer">
          <rect x="0" y="0" width="100" height="60"/>
          <text class="icon" x="50" y="20">📞</text>
          <text class="label" x="50" y="38">Syscall</text>
          <text class="sub" x="50" y="52">ecall  a0=sysno</text>
        </g>
        <!-- Scheduler -->
        <g class="node" id="node-sched" transform="translate(280,295)"
           onclick="stepChapter(6)" style="cursor:pointer">
          <rect x="0" y="0" width="110" height="50"/>
          <text class="icon" x="55" y="18">⏱</text>
          <text class="label" x="55" y="34">Scheduler</text>
          <text class="sub" x="55" y="47">round-robin</text>
        </g>
        <!-- VirtIO -->
        <g class="node" id="node-virtio" transform="translate(60,305)"
           onclick="stepChapter(8)" style="cursor:pointer">
          <rect x="0" y="0" width="110" height="55"/>
          <text class="icon" x="55" y="18">💾</text>
          <text class="label" x="55" y="34">VirtIO Block</text>
          <text class="sub" x="55" y="48">0x10001000</text>
        </g>
        <!-- FS -->
        <g class="node" id="node-fs" transform="translate(280,305)"
           onclick="stepChapter(9)" style="cursor:pointer">
          <rect x="0" y="0" width="110" height="55"/>
          <text class="icon" x="55" y="18">📁</text>
          <text class="label" x="55" y="34">Minix3 FS</text>
          <text class="sub" x="55" y="48">inodes  zones</text>
        </g>
        <!-- ELF -->
        <g class="node" id="node-elf" transform="translate(450,305)"
           onclick="stepChapter(10)" style="cursor:pointer">
          <rect x="0" y="0" width="110" height="55"/>
          <text class="icon" x="55" y="18">🚀</text>
          <text class="label" x="55" y="34">ELF Loader</text>
          <text class="sub" x="55" y="48">segments  entry</text>
        </g>
        <!-- CLINT -->
        <g class="node" id="node-clint" transform="translate(565,175)"
           onclick="stepChapter(4)" style="cursor:pointer">
          <rect x="0" y="0" width="100" height="55"/>
          <text class="icon" x="50" y="18">⏰</text>
          <text class="label" x="50" y="34">CLINT</text>
          <text class="sub" x="50" y="48">0x02000000</text>
        </g>

        <!-- data packet that travels along arrows -->
        <circle class="packet" id="data-packet" cx="0" cy="0" r="5"/>
      </svg>
    </div>

    <!-- ── Controls ── -->
    <div class="controls">
      <div class="ch-grid">
        <button class="ch-btn" id="ch1" onclick="stepChapter(1)">Ch1 Boot</button>
        <button class="ch-btn" id="ch2" onclick="stepChapter(2)">Ch2 UART</button>
        <button class="ch-btn" id="ch3" onclick="stepChapter(3)">Ch3 MMU</button>
        <button class="ch-btn" id="ch4" onclick="stepChapter(4)">Ch4 Trap</button>
        <button class="ch-btn" id="ch5" onclick="stepChapter(5)">Ch5 PLIC</button>
        <button class="ch-btn" id="ch6" onclick="stepChapter(6)">Ch6 Proc</button>
        <button class="ch-btn" id="ch7" onclick="stepChapter(7)">Ch7 Sys</button>
        <button class="ch-btn" id="ch8" onclick="stepChapter(8)">Ch8 VirtIO</button>
        <button class="ch-btn" id="ch9" onclick="stepChapter(9)">Ch9 FS</button>
        <button class="ch-btn" id="ch10" onclick="stepChapter(10)">Ch10 ELF</button>
      </div>
      <div class="run-row">
        <button class="btn primary" onclick="runAll()">▶ Run All Chapters</button>
        <button class="btn danger"  onclick="resetSim()">↺ Reset</button>
        <button class="btn success" id="qemu-btn" onclick="toggleQemu()">
          ▶ Start QEMU
        </button>
      </div>
    </div>
  </div>

  <!-- ── RIGHT: Console / Status / QEMU tabs ── -->
  <div class="right">
    <div class="tabs">
      <div class="tab active" onclick="showTab('console')">Console</div>
      <div class="tab" onclick="showTab('status')">Status</div>
      <div class="tab" onclick="showTab('qemu')">QEMU</div>
    </div>

    <!-- Console tab -->
    <div class="tab-pane active" id="pane-console">
      <div id="console-out"></div>
    </div>

    <!-- Status tab -->
    <div class="tab-pane" id="pane-status">
      <div class="status-grid" id="status-grid">

        <div class="stat-card">
          <div class="stat-title">Current Chapter</div>
          <div class="stat-val" id="s-chapter">—</div>
        </div>

        <div class="stat-card">
          <div class="stat-title">Hart #0</div>
          <div class="stat-val" id="s-hart">
            mhartid=0  satp=0  mstatus=—
          </div>
        </div>

        <div class="stat-card">
          <div class="stat-title">Page Allocator</div>
          <div class="alloc-bar">
            <div class="alloc-used" id="alloc-bar-used" style="width:0%"></div>
          </div>
          <div class="stat-val" id="s-pages">0 / — pages used</div>
        </div>

        <div class="stat-card">
          <div class="stat-title">MMU Tables</div>
          <div class="stat-val" id="s-mmu">0 tables  0 mappings</div>
        </div>

        <div class="stat-card">
          <div class="stat-title">Processes</div>
          <div class="badge-row" id="s-procs"></div>
        </div>

        <div class="stat-card">
          <div class="stat-title">Last Syscall</div>
          <div class="stat-val" id="s-syscall">—</div>
        </div>

        <div class="stat-card">
          <div class="stat-title">Last Trap</div>
          <div class="stat-val" id="s-trap">—</div>
        </div>

        <div class="stat-card">
          <div class="stat-title">VirtIO / FS</div>
          <div class="stat-val" id="s-virtio">—</div>
        </div>

      </div>
    </div>

    <!-- QEMU tab -->
    <div class="tab-pane" id="pane-qemu">
      <div class="qemu-controls">
        <button class="btn success" onclick="toggleQemu()" id="qemu-btn2">▶ Start QEMU</button>
        <button class="btn danger"  onclick="stopQemu()">■ Stop</button>
        <button class="btn" onclick="clearQemuLog()">✕ Clear</button>
      </div>
      <div id="qemu-status-line" style="color:var(--muted);font-size:10px;margin-bottom:4px">
        qemu-system-riscv64 -machine virt -cpu rv64 -m 128M -bios none
      </div>
      <div id="qemu-out">Waiting for QEMU…</div>
    </div>
  </div>
</div>

<div class="statusbar">
  <span><div class="dot" id="dot-sim"></div> Sim: <b id="stat-sim">idle</b></span>
  <span><div class="dot" id="dot-qemu"></div> QEMU: <b id="stat-qemu">off</b></span>
  <span>Pages used: <b id="stat-pages">0</b></span>
  <span>MMU tables: <b id="stat-mmu">0</b></span>
  <span>Processes: <b id="stat-procs">0</b></span>
  <span style="margin-left:auto;color:var(--muted)">
    Click any diagram node to run that chapter ·
    <a href="https://osblog.stephenmarz.com/" target="_blank"
       style="color:var(--accent);text-decoration:none">original blog ↗</a>
  </span>
</div>

<script>
const socket = io();
let qemuRunning = false;
let simRunning  = false;
const doneChs   = new Set();

/* ── Tab switching ── */
function showTab(id) {
  document.querySelectorAll('.tab').forEach((t,i)=>{
    t.classList.toggle('active', ['console','status','qemu'][i]===id);
  });
  document.querySelectorAll('.tab-pane').forEach(p=>{
    p.classList.toggle('active', p.id===`pane-${id}`);
  });
}

/* ── Console logging ── */
const con = document.getElementById('console-out');
function log(txt, cls='log-info') {
  const d = document.createElement('span');
  d.className = cls;
  d.textContent = txt;
  con.appendChild(d);
  con.scrollTop = con.scrollHeight;
}
function logln(txt, cls) { log(txt+'\n', cls); }

/* ── Node highlighting ── */
const nodeMap = {
  hart:'node-hart', uart:'node-uart', mmu:'node-mmu',
  plic:'node-plic', trap:'node-trap', proc:'node-proc',
  syscall:'node-syscall', sched:'node-sched', virtio:'node-virtio',
  fs:'node-fs', elf:'node-elf', clint:'node-clint'
};
let activeNode = null;
function highlightNode(name) {
  if (activeNode) document.getElementById(activeNode)?.classList.remove('active');
  const id = nodeMap[name];
  if (id) { document.getElementById(id)?.classList.add('active'); activeNode=id; }
}

/* ── Arrow activation ── */
const arrowLinks = {
  uart: ['arr-hart-uart'],
  mmu:  ['arr-hart-mmu'],
  plic: ['arr-uart-plic'],
  proc: ['arr-mmu-proc','arr-trap-proc','arr-elf-proc'],
  syscall: ['arr-proc-sys'],
  trap: ['arr-plic-trap','arr-clint-trap'],
  sched: ['arr-proc-sched','arr-sched-hart'],
  virtio: ['arr-virtio-fs'],
  fs: ['arr-virtio-fs'],
  elf: ['arr-fs-elf'],
};
function activateArrows(name) {
  document.querySelectorAll('.arrow').forEach(a=>a.classList.remove('active','pulse'));
  (arrowLinks[name]||[]).forEach(id=>{
    const el=document.getElementById(id);
    if(el){el.classList.add('pulse');setTimeout(()=>el.classList.add('active'),600);}
  });
}

/* ── Chapter button states ── */
function setChBtn(n, state) {
  const b = document.getElementById(`ch${n}`);
  if(!b) return;
  b.classList.remove('done','running');
  if(state==='done'){b.classList.add('done');doneChs.add(n);}
  else if(state==='running') b.classList.add('running');
}

/* ── API calls ── */
function stepChapter(n) { fetch('/api/chapter/'+n, {method:'POST'}); }
function runAll()    { fetch('/api/run_all',  {method:'POST'}); }
function resetSim()  { fetch('/api/reset',    {method:'POST'}).then(()=>{
  con.innerHTML=''; doneChs.clear();
  document.querySelectorAll('.ch-btn').forEach(b=>b.classList.remove('done','running'));
  document.querySelectorAll('.node').forEach(n=>n.classList.remove('active'));
  document.querySelectorAll('.arrow').forEach(a=>a.classList.remove('active','pulse'));
  logln('[reset] Simulation reset.','log-warn');
}); }
function toggleQemu() {
  if(qemuRunning) stopQemu();
  else startQemu();
}
function startQemu() { fetch('/api/qemu/start',{method:'POST'}); showTab('qemu'); }
function stopQemu()  { fetch('/api/qemu/stop', {method:'POST'}); }
function clearQemuLog() { document.getElementById('qemu-out').textContent=''; }

/* ── Socket.IO event handlers ── */
socket.on('console', d=>{
  const cls = d.level==='warn'?'log-warn':d.level==='err'?'log-err':'log-info';
  log(d.text, cls);
});
socket.on('uart_tx', d=>{
  // subtle purple dot in console for each char
});
socket.on('chapter', d=>{
  setChBtn(d.n,'running');
  document.getElementById('s-chapter').innerHTML =
    `<span>Ch${d.n}</span> ${d.title}`;
  document.getElementById('stat-sim').textContent = `ch${d.n}`;
  document.getElementById('dot-sim').className = 'dot yellow';
});
socket.on('chapter_done', d=>{
  setChBtn(d.n,'done');
});
socket.on('highlight_node', d=>{
  highlightNode(d.node);
  activateArrows(d.node);
});
socket.on('boot_done', ()=>{
  document.getElementById('s-hart').innerHTML =
    'mhartid=0  <span>satp=0</span>  mstatus=<span>machine mode</span>';
});
socket.on('page_table', d=>{
  const pct = d.total ? (d.used/d.total*100).toFixed(1) : 0;
  document.getElementById('alloc-bar-used').style.width = pct+'%';
  document.getElementById('s-pages').innerHTML =
    `<span>${d.used}</span> / ${d.total} pages  (${pct}%)`;
  document.getElementById('stat-pages').textContent = d.used;
});
socket.on('mmu_state', d=>{
  document.getElementById('s-mmu').innerHTML =
    `<span>${d.tables}</span> tables  <span>${d.mappings.length}</span> recent mappings`;
  document.getElementById('stat-mmu').textContent = d.tables;
});
socket.on('scheduler', d=>{
  document.getElementById('stat-procs').textContent = d.length;
  const row = document.getElementById('s-procs');
  row.innerHTML = d.map(p=>`<span class="chip ${p.state.toLowerCase()}">${p.kernel?'K':'U'}:${p.pid} ${p.state}</span>`).join('');
});
socket.on('syscall', d=>{
  document.getElementById('s-syscall').innerHTML =
    `<span>sys_${d.name}()</span>  pid=${d.pid}  #${d.number}`;
  logln(`  [SYSCALL] ${d.name}() pid=${d.pid}`, 'log-sys');
  highlightNode('syscall'); activateArrows('syscall');
});
socket.on('trap_event', d=>{
  const t = d.async ? 'ASYNC':'SYNC';
  document.getElementById('s-trap').innerHTML =
    `<span>${d.name}</span>  cause=${d.cause}  [${t}]`;
  logln(`  [TRAP] ${d.name}  cause=0x${d.cause.toString(16)}`, 'log-warn');
  highlightNode('trap'); activateArrows('trap');
});
socket.on('plic_event', d=>{
  logln(`  [PLIC] source=${d.source} (${d.device})`, 'log-ok');
  highlightNode('plic'); activateArrows('plic');
});
socket.on('virtio', d=>{
  document.getElementById('s-virtio').innerHTML =
    `VirtIO <span>${d.status}</span>  sectors=<span>${d.sectors}</span>`;
  logln(`  [VirtIO] ${d.status}  sectors=${d.sectors}`,'log-ok');
  highlightNode('virtio'); activateArrows('virtio');
});
socket.on('fs_dir', d=>{
  logln('  [FS] Directory listing:','log-ok');
  d.entries.forEach(e=>logln(`    [${e.inode}] ${e.perms} ${e.name} ${e.size}B`,'log-ok'));
  highlightNode('fs'); activateArrows('fs');
});
socket.on('elf_loaded', d=>{
  logln(`  [ELF] pid=${d.pid} entry=${d.entry} sp=${d.sp}`,'log-ok');
  highlightNode('elf'); activateArrows('elf');
});
socket.on('sim_done', ()=>{
  document.getElementById('stat-sim').textContent='done';
  document.getElementById('dot-sim').className='dot green';
  document.getElementById('sim-badge').textContent='DONE';
  document.getElementById('sim-badge').classList.add('on');
  logln('\n✓ All chapters complete!','log-ok');
});

/* ── QEMU events ── */
socket.on('qemu', d=>{
  const btn  = document.getElementById('qemu-btn');
  const btn2 = document.getElementById('qemu-btn2');
  const badge= document.getElementById('qemu-badge');
  const dot  = document.getElementById('dot-qemu');
  const stat = document.getElementById('stat-qemu');
  if(d.status==='starting'){
    logln('[QEMU] Starting…','log-warn');
    qemuRunning=true; btn.textContent='■ Stop QEMU'; btn2.textContent='■ Stop';
    badge.textContent='QEMU: STARTING'; badge.className='badge';
    dot.className='dot yellow'; stat.textContent='starting';
    const qo=document.getElementById('qemu-out');
    qo.textContent=''; qo.textContent+=`$ ${d.cmd}\n`;
  } else if(d.status==='running'){
    badge.textContent='QEMU: RUNNING'; badge.className='badge on';
    dot.className='dot green'; stat.textContent='running';
    document.getElementById('qemu-status-line').textContent=`PID: ${d.pid}`;
  } else if(d.status==='stopped' || d.status==='error'){
    qemuRunning=false; btn.textContent='▶ Start QEMU'; btn2.textContent='▶ Start QEMU';
    badge.textContent='QEMU: OFF'; badge.className='badge';
    dot.className='dot red'; stat.textContent='stopped';
    if(d.status==='error') logln(`[QEMU ERROR] ${d.msg}`,'log-err');
  }
});
socket.on('qemu_output', d=>{
  const qo = document.getElementById('qemu-out');
  qo.textContent += d.text;
  qo.scrollTop = qo.scrollHeight;
  // Also mirror to console with green style
  logln('[QEMU] '+d.text.trimEnd(), 'log-ok');
});

/* initial log */
logln('RISC-V OS Simulator ready.','log-ok');
logln('Click a chapter button or a diagram node to begin.','log-info');
logln('Click "▶ Start QEMU" to launch the real emulator.','log-info');
</script>
</body>
</html>
"""

# ── Flask routes ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/api/chapter/<int:n>', methods=['POST'])
def api_chapter(n):
    if 1 <= n <= 10:
        t = threading.Thread(target=run_chapter, args=(n,), daemon=True)
        t.start()
    return jsonify(ok=True)

@app.route('/api/run_all', methods=['POST'])
def api_run_all():
    if not SIM_STATE["running"]:
        SIM_STATE["running"] = True
        t = threading.Thread(target=run_all_chapters, daemon=True)
        t.start()
    return jsonify(ok=True)

@app.route('/api/reset', methods=['POST'])
def api_reset():
    global _next_pid, PAGE_ALLOC, PLIC_CTRL, SCHEDULER, UART0, PHYS_MEM
    SIM_STATE.update(chapter=0, running=False, boot_done=False, hart=None)
    PHYS_MEM = PhysicalMemory()
    PAGE_ALLOC = PageAllocator(HEAP_START, HEAP_SIZE - HEAP_START)
    PLIC_CTRL = PLIC()
    SCHEDULER = Scheduler()
    UART0 = Uart()
    MMU._tables.clear(); MMU._mappings.clear(); MMU._next_addr = 0x9000_0000
    _next_pid = 1
    return jsonify(ok=True)

@app.route('/api/qemu/start', methods=['POST'])
def api_qemu_start():
    t = threading.Thread(target=QEMU_MGR.start, daemon=True)
    t.start()
    return jsonify(ok=True)

@app.route('/api/qemu/stop', methods=['POST'])
def api_qemu_stop():
    QEMU_MGR.stop()
    return jsonify(ok=True)

@app.route('/api/qemu/info')
def api_qemu_info():
    return jsonify(QEMU_MGR.info())

@app.route('/api/status')
def api_status():
    snap = PAGE_ALLOC.snapshot()
    return jsonify({
        'chapter': SIM_STATE['chapter'],
        'running': SIM_STATE['running'],
        'pages': snap,
        'mmu': MMU.snapshot(),
        'procs': SCHEDULER.snapshot(),
        'qemu': QEMU_MGR.info(),
    })

# ── SocketIO events ───────────────────────────────────────────────────────────

@socketio.on('connect')
def on_connect():
    emit('console', {'text': '[browser connected]\n', 'level': 'info'})

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    host = '0.0.0.0'
    port = 5000
    print("=" * 60)
    print("  RISC-V OS Simulator — Web UI")
    print(f"  Open:  http://localhost:{port}")
    print("  Press  Ctrl-C  to quit")
    print("=" * 60)
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)
