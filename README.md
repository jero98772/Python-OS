# Writing a RISC-V Operating System in Python (In process)



### An Educational Translation of [Stephen Marz's "Writing a RISC-V OS in Rust"](https://osblog.stephenmarz.com/index.html)

> **Why Python?**  
> The original blog series targets bare-metal RISC-V hardware using Rust.  
> Python cannot run directly on a CPU without a runtime, but every *concept* —  
> memory maps, UART protocols, page allocators, MMU tables, traps, processes,  
> filesystems — can be faithfully simulated in pure Python.  
> This tutorial keeps the same chapter structure and the same ideas; only the  
> language changes. Run `python3 risc_v_os_python.py` to see every subsystem  
> execute end-to-end.

---

## The Road Ahead

| Chapter | Topic |
|---------|-------|
| 0 | Setup — constants & physical memory |
| 1 | Boot — hart selection, BSS clearing, entering kmain |
| 2 | UART — NS16550a driver and print macros |
| 3.1 | Page-grained memory allocator |
| 3.2 | Memory Management Unit (SV39 paging) |
| 4 | Trap handling |
| 5 | Platform-Level Interrupt Controller (PLIC) |
| 6 | Process structure |
| 7 | System calls |
| 8 | Process scheduler |
| 9 | VirtIO block driver |
| 10 | Minix3 filesystem |
| 11 | ELF loader & userspace processes |

---

## Chapter 0 — Setup: Constants & Physical Memory

### The RISC-V Memory Map

QEMU's `virt` machine wires devices to fixed physical addresses.  
The original C source (`qemu/hw/riscv/virt.c`) looks like:

```c
static const struct MemmapEntry {
    hwaddr base;
    hwaddr size;
} virt_memmap[] = {
    [VIRT_CLINT]  = { 0x2000000,  0x10000  },
    [VIRT_PLIC]   = { 0xc000000,  0x4000000},
    [VIRT_UART0]  = { 0x10000000, 0x100    },
    [VIRT_VIRTIO] = { 0x10001000, 0x1000   },
    [VIRT_DRAM]   = { 0x80000000, 0x0      },
};
```

In Python these become plain integer constants:

```python
VIRT_CLINT  = 0x0200_0000
VIRT_PLIC   = 0x0C00_0000
VIRT_UART0  = 0x1000_0000
VIRT_VIRTIO = 0x1000_1000
VIRT_DRAM   = 0x8000_0000
PAGE_SIZE   = 4096          # 4 KiB
HEAP_SIZE   = 128 * 1024 * 1024  # 128 MiB
```

### Simulated RAM

Real hardware has silicon DRAM. We replace it with a Python `bytearray`:

```python
class PhysicalMemory:
    def __init__(self, size: int = HEAP_SIZE):
        self._mem = bytearray(size)

    def read8(self, addr: int) -> int:
        return self._mem[addr]

    def write8(self, addr: int, val: int):
        self._mem[addr] = val & 0xFF

    def read64(self, addr: int) -> int:
        return struct.unpack_from('<Q', self._mem, addr)[0]

    def write64(self, addr: int, val: int):
        struct.pack_into('<Q', self._mem, addr, val)

    def zero_range(self, start: int, end: int):
        for i in range(start, end):
            self._mem[i] = 0
```

`read_volatile` / `write_volatile` in Rust → ordinary `read` / `write` in Python,  
because Python has no optimizer to outsmart.

---

## Chapter 1 — Boot: Hart Selection, BSS, and kmain

### What boot.S does on real hardware

```asm
_start:
    csrr  t0, mhartid       # read this core's ID
    bnez  t0, 3f            # if not zero, park (wfi loop)
    csrw  satp, zero        # disable MMU
    la    a0, _bss_start
    la    a1, _bss_end
1:  sd    zero, (a0)        # zero BSS 8 bytes at a time
    addi  a0, a0, 8
    bltu  a0, a1, 1b
    la    sp, _stack
    ...
    mret                    # jump to kmain
3:  wfi; j 3b               # non-boot harts spin here
```

### Python translation

```python
class Hart:
    """Simulates a RISC-V Hardware Thread."""
    def __init__(self, hart_id: int):
        self.hart_id = hart_id
        self.registers = [0] * 32
        self.pc = VIRT_DRAM
        self.mhartid = hart_id
        self.satp = 0           # 0 = MMU off
        self.mstatus = 0
        self.mie = 0
        self.parked = False

def boot(num_harts: int = 4) -> Hart:
    harts = [Hart(i) for i in range(num_harts)]

    # All non-zero harts park immediately
    for h in harts[1:]:
        h.parked = True

    boot_hart = harts[0]

    # Disable virtual memory
    boot_hart.satp = 0

    # Zero the BSS section (global uninitialised variables)
    PHYS_MEM.zero_range(BSS_START, BSS_END)

    # Set mstatus: MPP=11 (machine mode), MPIE=1, MIE=1
    MPP  = 0b11 << 11
    MPIE = 1 << 7
    MIE  = 1 << 3
    boot_hart.mstatus = MPP | MPIE | MIE

    return boot_hart
```

**Key ideas:**
- `mhartid` — a read-only CSR that identifies the current core.  
- `satp = 0` — disables the MMU; all addresses are physical.  
- BSS — the segment for uninitialised global variables. The *OS* must zero it because the firmware doesn't.  
- `mret` — "machine return"; sets privilege mode from `mstatus.MPP` and jumps to `mepc`.

---

## Chapter 2 — UART: The NS16550a Driver

### What is UART?

UART (Universal Asynchronous Receiver/Transmitter) is how the kernel talks to  
a serial terminal. QEMU emulates the **NS16550a** chipset at address  
`0x1000_0000`. All communication happens by reading and writing 8-bit registers  
at offsets from that base address.

### NS16550a Register Map

| Offset | Name | Direction | Purpose |
|--------|------|-----------|---------|
| +0 | RBR / THR | read / write | Receive buffer / Transmit holding |
| +1 | IER | write | Interrupt enable |
| +2 | FCR / IIR | write / read | FIFO control / Interrupt identification |
| +3 | LCR | write | Line control (word length, DLAB) |
| +5 | LSR | read | Line status (DR bit = data ready) |

### Initialisation sequence

```python
class Uart:
    def init(self):
        # LCR bits [1:0] = 11  →  8-bit word length
        lcr = (1 << 0) | (1 << 1)
        self._write(LCR, lcr)

        # FCR bit 0 = 1  →  enable FIFOs
        self._write(FCR, 1 << 0)

        # IER bit 0 = 1  →  enable receiver interrupt
        self._write(IER, 1 << 0)

        # Divisor latch: open DLAB, write divisor, close DLAB
        self._write(LCR, lcr | (1 << 7))  # set DLAB
        self._write(0, divisor & 0xFF)
        self._write(1, divisor >> 8)
        self._write(LCR, lcr)             # clear DLAB
```

### Writing to the console

In Rust, `write_str` is the one method required by the `Write` trait —  
everything `write!` / `println!` does flows through it.

```python
def write_str(self, s: str):
    for ch in s:
        self.put(ord(ch))   # put() → write byte to THR at offset 0

def put(self, byte: int):
    self._regs[THR] = byte
    sys.stdout.write(chr(byte))
    sys.stdout.flush()
```

### Reading from the console

```python
def get(self) -> Optional[int]:
    # LSR bit 0 (DR) = 1 means a byte is waiting
    if self._regs[LSR] & 1:
        return self._rx_fifo.popleft()
    return None
```

### The print! macro

In the Rust OS, `print!` is implemented by calling `write!` on a `Uart` struct.  
In Python we expose the same behaviour as a function:

```python
def print_uart(msg: str, end: str = '\n'):
    UART0.write_str(msg + end)
```

---

## Chapter 3.1 — Page-Grained Memory Allocator

### The problem

After the kernel's code, data, and stack, the remaining RAM is the **heap**.  
We hand out this heap in 4 KiB chunks called *pages*. An allocator must track  
which pages are free and which are in use — without wasting much memory doing so.

### Descriptor-based allocation

We keep **one byte per page** as a *descriptor*. Two bits are used:

```python
class PageBits(IntFlag):
    Empty = 0
    Taken = 1 << 0   # this page is allocated
    Last  = 1 << 1   # this is the final page of a contiguous allocation
```

The descriptor array sits at the very start of the heap, and usable pages begin  
immediately after it.

```
HEAP_START  →  [desc 0][desc 1]...[desc N]  [page 0][page 1]...[page N]
               ←──── descriptor array ────→  ←──── usable pages ────→
               ALLOC_START
```

### alloc()

```python
def alloc(self, pages: int) -> Optional[int]:
    for i in range(self.num_pages - pages):
        if self.descriptors[i].is_free():
            # Check entire requested run is free
            if all(self.descriptors[j].is_free()
                   for j in range(i, i + pages)):
                for j in range(i, i + pages):
                    self.descriptors[j].set_taken()
                self.descriptors[i + pages - 1].set_last()
                return self.alloc_start + i * PAGE_SIZE
    return None  # out of memory
```

### dealloc()

```python
def dealloc(self, ptr: int):
    idx = (ptr - self.alloc_start) // PAGE_SIZE
    while self.descriptors[idx].is_taken() and \
          not self.descriptors[idx].is_last():
        self.descriptors[idx].clear()
        idx += 1
    # idx now points at the Last page
    assert self.descriptors[idx].is_last(), "Possible double-free!"
    self.descriptors[idx].clear()
```

### zalloc()

For security, pages handed to user processes must be zeroed first:

```python
def zalloc(self, pages: int) -> Optional[int]:
    addr = self.alloc(pages)
    if addr is not None:
        PHYS_MEM.zero_range(addr, addr + pages * PAGE_SIZE)
    return addr
```

---

## Chapter 3.2 — Memory Management Unit (SV39 Paging)

### What the MMU does

The MMU translates *virtual addresses* (used by programs) to *physical addresses*  
(real RAM locations). This lets every process believe it owns the entire address  
space, while the OS controls exactly which physical pages it actually sees.

### SV39 virtual address layout

```
 38        30 29       21 20       12 11            0
 ┌──────────┬───────────┬───────────┬───────────────┐
 │  VPN[2]  │  VPN[1]   │  VPN[0]   │  page offset  │
 │  9 bits  │  9 bits   │  9 bits   │   12 bits     │
 └──────────┴───────────┴───────────┴───────────────┘
```

Each VPN is a 9-bit index into a 512-entry page table.  
Three levels of tables give us 39-bit virtual → 56-bit physical translation.

### Page table entry bits

| Bit | Name | Meaning |
|-----|------|---------|
| 0 | V (Valid) | Entry is active |
| 1 | R (Read) | Readable |
| 2 | W (Write) | Writeable |
| 3 | X (Execute) | Executable |
| 4 | U (User) | Accessible in user mode |
| 6 | A (Accessed) | Set by hardware on access |
| 7 | D (Dirty) | Set by hardware on write |

If **any** of R/W/X is set → this is a **leaf** (maps actual memory).  
If all R/W/X are 0 → this is a **branch** (points to the next table).

### map() in Python

```python
@classmethod
def map(cls, root_addr, vaddr, paddr, bits, level=0):
    vpn = [(vaddr >> 12) & 0x1FF,   # VPN[0]
           (vaddr >> 21) & 0x1FF,   # VPN[1]
           (vaddr >> 30) & 0x1FF]   # VPN[2]
    ppn = [(paddr >> 12) & 0x1FF,
           (paddr >> 21) & 0x1FF,
           (paddr >> 30) & 0x3FFFFFF]

    root = cls._get_table(root_addr)
    v = root.entries[vpn[2]]

    for i in range(2, level, -1):    # walk down to target level
        if v.is_invalid():
            new_addr = cls._new_table()          # allocate new table
            v.entry = (new_addr >> 2) | Valid
        v = cls._get_table(v.physical_addr()).entries[vpn[i-1]]

    # Install leaf entry
    entry = (ppn[2] << 28) | (ppn[1] << 19) | (ppn[0] << 10) | bits | Valid
    v.entry = entry
```

### virt_to_phys() — software page-table walk

The kernel must walk the page table manually for system-call arguments:

```python
@classmethod
def virt_to_phys(cls, root_addr, vaddr) -> Optional[int]:
    vpn = [(vaddr >> 12) & 0x1FF,
           (vaddr >> 21) & 0x1FF,
           (vaddr >> 30) & 0x1FF]
    v = cls._get_table(root_addr).entries[vpn[2]]
    for i in range(2, -1, -1):
        if v.is_invalid(): return None
        if v.is_leaf():
            off_mask = (1 << (12 + i * 9)) - 1
            return ((v.entry << 2) & ~off_mask) | (vaddr & off_mask)
        v = cls._get_table(v.physical_addr()).entries[vpn[i-1]]
    return None
```

### SATP register

Turning the MMU on means writing the root table address (divided by PAGE_SIZE)  
plus the mode (8 = Sv39) into the `satp` CSR:

```python
root_ppn = root_addr >> 12
satp_val = (8 << 60) | (pid << 44) | root_ppn
hart.satp = satp_val
# In assembly: csrw satp, satp_val
```

---

## Chapter 4 — Trap Handling

### What is a trap?

A *trap* is any event that causes the CPU to stop its current instruction stream  
and jump to the kernel's handler. There are two kinds:

- **Synchronous** — caused by the current instruction (illegal instruction,  
  page fault, `ecall`). The most-significant bit of `mcause` is **0**.
- **Asynchronous** — caused by something external (timer, UART interrupt).  
  The MSB of `mcause` is **1**.

### The mtvec register

`mtvec` holds the address of the trap handler. We set it once at boot:

```asm
la  t2, m_trap_vector
csrw mtvec, t2
```

In Python we simulate this by calling `trap_handler.handle()` directly.

### TrapFrame — freezing a process

When a trap fires, the CPU must save every register so the interrupted process  
can be resumed later:

```python
@dataclass
class TrapFrame:
    regs:       List[int]   # x0..x31 (general purpose)
    fregs:      List[float] # f0..f31 (floating point)
    satp:       int         # MMU root table
    trap_stack: int         # stack for handling this trap
    hartid:     int
    pc:         int
    pid:        int
```

In assembly this is done with `sd x1, 8(t6)` … `sd x31, 248(t6)` in a loop.

### Dispatch logic

```python
def handle(self, hart, cause, epc, tval, frame) -> int:
    is_async  = bool(cause >> 63 & 1)
    cause_num = cause & 0xFFF
    return_pc = epc

    if is_async:
        if cause_num == 7:       # machine timer → context switch
            self._reset_timer()
            return_pc = self._context_switch(frame, return_pc)
        elif cause_num == 11:    # external → PLIC
            return_pc = self._handle_external(frame, return_pc)
    else:
        if cause_num in (8, 9, 11):   # ecall
            return_pc = self.syscall.dispatch(return_pc, frame)
            return_pc += 4            # skip the ecall instruction
        elif cause_num == 13:         # load page fault
            ...
    return return_pc
```

---

## Chapter 5 — Platform-Level Interrupt Controller (PLIC)

### What is the PLIC?

Every external interrupt from every device (UART, disk, network…) arrives at  
the CPU through a **single** wire — the *external interrupt pin*.  
The PLIC sits between the devices and the CPU, routing, prioritising, and  
masking interrupts.

### PLIC registers (MMIO)

| Address | Purpose |
|---------|---------|
| `0x0C00_0000` | Priority per interrupt source (0–7) |
| `0x0C00_1000` | Pending bitmask |
| `0x0C00_2000` | Enable bitmask |
| `0x0C20_0000` | Global threshold |
| `0x0C20_0004` | Claim (read) / Complete (write) |

### Usage

```python
# Configure
PLIC.set_threshold(0)      # allow all priorities through
PLIC.enable(10)            # UART0 is on pin 10
PLIC.set_priority(10, 1)   # give it priority 1

# In the trap handler, when cause == external interrupt:
interrupt_id = PLIC.next()         # claim highest-priority pending
if interrupt_id == 10:
    byte = uart.get()              # read from UART
    echo(byte)
PLIC.complete(interrupt_id)        # tell PLIC we are done
```

The *claim* step tells the PLIC "I'm handling this"; the *complete* step lets  
the device interrupt again in the future.

---

## Chapter 6 — Process Structure

### What a process needs

```python
@dataclass
class Process:
    pid:             int           # unique process ID
    state:           ProcessState  # Running / Sleeping / Waiting / Dead
    frame:           TrapFrame     # full register snapshot
    root_table_addr: int           # page table root (for MMU)
    stack_pages:     int           # physical address of stack allocation
    program_counter: int           # PC to resume at
```

### Process states

```
  ┌─────────┐  schedule()  ┌─────────┐
  │ Waiting │ ──────────→  │ Running │
  └─────────┘              └────┬────┘
       ↑                        │ timer trap
       │                        ↓
  ┌─────────┐  wakeup()   ┌──────────┐
  │ Sleeping│ ←────────── │ Waiting  │
  └─────────┘             └──────────┘
```

### Memory layout for a process

```
  virtual address space (seen by the process)
  ┌──────────────────────────┐
  │  text (code)             │ 0x2000_0000
  │  rodata                  │
  │  data / bss              │
  ├──────────────────────────┤
  │  stack  ↓  (grows down)  │ 0x1000_2000  (top)
  └──────────────────────────┘ 0x1000_0000  (base)
```

The OS maps these virtual addresses to physical RAM pages via the MMU.

---

## Chapter 7 — System Calls

### Why system calls?

User-mode code cannot access privileged resources (hardware registers, other  
processes' memory). It asks the kernel using the `ecall` instruction, which  
causes a synchronous trap with `cause = 8` (user ecall).

### Calling convention

- `a0` (`x10`) — syscall number  
- `a1`–`a7` — up to seven arguments  
- `a0` — return value

```asm
# In user space:
li   a0, 64         # syscall write
li   a1, 1          # fd = stdout
la   a2, message    # buffer
li   a3, 13         # length
ecall
```

### Python dispatcher

```python
def dispatch(self, mepc: int, frame: TrapFrame) -> int:
    sysno = frame.regs[10]          # read a0
    match sysno:
        case 93:  self._exit(frame)
        case 64:  self._write(frame)
        case 63:  self._read(frame)
        case _:   print(f"Unknown syscall {sysno}")
    return mepc                     # caller adds +4 to skip ecall
```

### Filesystem read syscall

```python
def _read(self, frame: TrapFrame):
    inode  = frame.regs[11]   # a1
    size   = frame.regs[13]   # a3
    offset = frame.regs[14]   # a4
    data = self.fs.read_inode(inode, size, offset)
    frame.regs[10] = len(data)   # return value in a0
```

---

## Chapter 8 — Process Scheduler

### The simplest scheduler: round-robin

```python
class Scheduler:
    def __init__(self):
        self._queue: deque = deque()   # deque of Process objects

    def schedule(self, frame, mepc):
        self._queue.rotate(-1)         # move front → back
        proc = self._queue[0]          # new front is next process
        # Perform context switch:
        #   save old frame → load proc.frame → set satp → mret
        proc.step()                    # simulate one time slice
        return proc.program_counter
```

### Context switch in assembly (real hardware)

```asm
switch_to_user:
    csrw  mscratch, a0      # save trap frame pointer
    csrw  mstatus, t0       # MPP=00 → return to User mode
    csrw  mepc, a1          # set PC = process entry point
    csrw  satp,  a2         # switch page table
    sfence.vma              # flush TLB
    # restore all registers from the trap frame
    .set i, 1
    .rept 31
        load_gp %i
        .set i, i+1
    .endr
    mret                    # jump to mepc, apply mstatus
```

The `mret` instruction is the *only* way to switch from machine mode to  
user mode; there is no `jmp`-to-userspace instruction.

### Timer frequency

The CLINT timer fires whenever `mtime >= mtimecmp`. Setting:

```python
mtimecmp = mtime + 10_000_000   # @ 10 MHz clock → 1 second per slice
```

is useful for debugging (you can watch each tick), but real OSes use  
1–10 ms slices (10 000 – 100 000 clock ticks).

---

## Chapter 9 — VirtIO Block Driver

### What is VirtIO?

VirtIO is a standardised I/O protocol for virtual machines. It defines  
a generic queue mechanism and device-specific command formats.  
A *block device* (virtual hard drive) is device type 2.

### Initialisation sequence (nine steps)

```python
def setup(self) -> bool:
    # 1. Reset
    self.mmio_write(Status, 0)
    # 2. Acknowledge
    self.mmio_write(Status, ACKNOWLEDGE)
    # 3. Driver flag
    self.mmio_write(Status, ACKNOWLEDGE | DRIVER)
    # 4. Read features
    features = self.mmio_read(HostFeatures) & ~READ_ONLY_BIT
    # 5. Write accepted features
    self.mmio_write(GuestFeatures, features)
    # 6. Set FEATURES_OK
    self.mmio_write(Status, ACKNOWLEDGE | DRIVER | FEATURES_OK)
    # 7. Verify FEATURES_OK still set
    if not (self.mmio_read(Status) & FEATURES_OK): return False
    # 8. Device-specific queue setup
    self.mmio_write(QueueNum, RING_SIZE)
    # 9. Set DRIVER_OK → device is live
    self.mmio_write(Status, ACKNOWLEDGE | DRIVER | FEATURES_OK | DRIVER_OK)
    return True
```

### Making a read request

```python
def block_op(self, buffer, size, offset, write, watcher_pid=0):
    sector = offset // 512        # disk sectors are 512 bytes
    req = VirtioRequest(
        blktype = VIRTIO_BLK_T_IN,   # read
        sector  = sector,
        data    = bytearray(size),
    )
    # Fill three descriptors: header → buffer → status
    # Write index to avail ring, notify device via QueueNotify
    self.mmio_write(QueueNotify, 0)
```

### Handling the response interrupt

The device signals completion via PLIC interrupt (source 1–8 for VirtIO).  
When we receive it:

```python
if interrupt_id in range(1, 9):          # VirtIO interrupt
    device = VIRTIO_DEVICES[interrupt_id - 1]
    pending(device)                       # drain used ring
    plic.complete(interrupt_id)
```

---

## Chapter 10 — Minix3 Filesystem

### Disk layout

```
Block 0  │ Boot block (reserved)
Block 1  │ Superblock  ← magic, inode count, zone count
Block 2  │ Inode bitmap (1 bit per inode)
Block 3  │ Zone bitmap  (1 bit per zone/block)
Block 4+ │ Inode table  (64 bytes per inode)
Block N  │ Data zones   (1 024 bytes per zone)
```

### The Superblock

```python
@dataclass
class Superblock:
    ninodes:         int   # total number of inodes
    imap_blocks:     int   # inode bitmap size in blocks
    zmap_blocks:     int   # zone bitmap size in blocks
    first_data_zone: int   # first zone holding file data
    block_size:      int   # always 1 024 for Minix3
    magic:           int   # 0x4d5a

    def is_valid(self): return self.magic == 0x4d5a
```

### The Inode

An inode stores metadata for **one** file or directory:

```python
@dataclass
class Inode:
    mode:  int        # permissions + type (dir, regular, …)
    size:  int        # file size in bytes
    zones: List[int]  # 7 direct + 1 indirect + 1 double + 1 triple

    def is_dir(self):     return (self.mode & 0xF000) == 0x4000
    def is_regular(self): return (self.mode & 0xF000) == 0x8000
```

### Zone pointer indirection

| Zone index | Type | Addressable data |
|-----------|------|-----------------|
| 0–6 | Direct | 7 × 1 024 B = 7 KiB |
| 7 | Singly indirect | 256 × 1 024 B = 256 KiB |
| 8 | Doubly indirect | 256² × 1 024 B ≈ 67 MiB |
| 9 | Triply indirect | 256³ × 1 024 B ≈ 17 GiB |

### Reading file data

```python
def read_inode(self, inode_num, size=-1, offset=0):
    inode = self.read_inode_meta(inode_num)
    result = bytearray()

    for zone in inode.zones[:7]:            # direct zones
        if zone == 0: continue
        block = self._read_block(zone)
        result.extend(block)
        if len(result) >= inode.size: break

    # singly-indirect zone (inode.zones[7])
    indirect = self._read_block(inode.zones[7])
    for i in range(256):
        z = struct.unpack_from('<I', indirect, i*4)[0]
        if z: result.extend(self._read_block(z))

    return bytes(result[:size or inode.size])
```

### Directory entries

A directory inode's data is a sequence of 64-byte `DirEntry` records:

```python
@dataclass
class DirEntry:
    inode: int       # 4 bytes — which inode this entry points to
    name:  str       # 60 bytes — file name (null-padded)
```

Inode **1** is always the root directory `/`.

---

## Chapter 11 — ELF Loader & Userspace Processes

### The ELF file format

An ELF (Executable and Linkable Format) file starts with a 64-byte header  
followed by *program headers* that describe memory segments to load.

```
ELF Header (64 bytes)
  magic:    0x7f 'E' 'L' 'F'
  class:    2  (64-bit)
  machine:  0xF3  (RISC-V)
  entry:    virtual address of _start
  phoff:    offset of first program header

Program Header (56 bytes each)
  type:     PT_LOAD (1) = load this segment
  flags:    R=4, W=2, X=1
  offset:   where in the file
  vaddr:    where in virtual memory
  filesz:   bytes in file
  memsz:    bytes in memory (≥ filesz; rest is zeroed)
```

### Loading an ELF

```python
def load(self, elf_bytes: bytes) -> Optional[Process]:
    hdr = ElfHeader.from_bytes(elf_bytes)
    assert hdr.magic   == ELF_MAGIC
    assert hdr.machine == ELF_MACHINE_RISCV
    assert hdr.obj_type == ELF_TYPE_EXEC

    proc = Process()
    program_mem = zalloc(64)     # 256 KiB working area

    for i in range(hdr.phnum):
        ph = ProgramHeader.from_bytes(elf_bytes[hdr.phoff + i*56:])
        if ph.seg_type != PT_LOAD: continue

        # Copy bytes from ELF into simulated RAM
        src = elf_bytes[ph.off : ph.off + ph.filesz]
        PHYS_MEM.write_bytes(program_mem + ph.off, src)

        # Map virtual → physical in the process's page table
        bits = User | (Read if PF_READ) | (Write if PF_WRITE) | (Execute if PF_EXECUTE)
        for j in range(ph.memsz // PAGE_SIZE + 1):
            MMU.map(proc.root_table_addr,
                    ph.vaddr + j*PAGE_SIZE,
                    program_mem + ph.off + j*PAGE_SIZE,
                    bits, level=0)

    proc.frame.pc = hdr.entry_addr        # set program counter
    proc.frame.sp = STACK_ADDR + STACK_PAGES * PAGE_SIZE
    proc.frame.satp = (8 << 60) | (proc.pid << 44) | (proc.root_table_addr >> 12)
    return proc
```

### The _start shim (userspace)

Every ELF needs a tiny assembly stub that calls `main` and then calls `exit`:

```asm
.section .text.init
.global _start
_start:
    call  main          # call C/Rust main()
    li    a0, 93        # syscall number: exit
    ecall               # → kernel trap handler
```

The OS sees `cause == 8` (user ecall), reads `a7 = 93`, and removes the  
process from the scheduler.

---

## Putting It All Together

When you run `python3 risc_v_os_python.py`, the execution flow is:

```
boot()
  → zero BSS
  → set up mstatus (machine mode)
  → return boot hart

kmain()
  ├── Chapter 2:  UART.init()   "Hello from the kernel!"
  ├── Chapter 3.1: alloc / dealloc pages
  ├── Chapter 3.2: MMU.map()  virt_to_phys()
  ├── Chapter 4:  simulate timer trap + page fault
  ├── Chapter 5:  PLIC enable/claim/complete
  ├── Chapter 6+8: create processes, run scheduler
  ├── Chapter 7:  dispatch system calls
  ├── Chapter 9:  VirtIO block device setup + read
  ├── Chapter 10: Minix3 FS → list root dir → read file
  └── Chapter 11: parse ELF → load into process → schedule
```

---

## Key Differences: Python vs Rust/Assembly

| Concept | Rust / Assembly | Python |
|---------|----------------|--------|
| MMIO access | `ptr.write_volatile(val)` | `dict` or `bytearray` |
| Interrupt entry | `csrw mtvec, fn_addr` | `trap_handler.handle()` call |
| Context save | `sd x1, 8(t6)` … | `TrapFrame` dataclass |
| Privilege mode | `mstatus.MPP` bits | simulated flag |
| MMU enable | `csrw satp, val` | `hart.satp = val` |
| Inline assembly | `asm!("wfi")` | `pass` (no-op) |
| `unsafe` blocks | required for raw pointers | all Python is "unsafe" by default |
| `no_std` | required for bare-metal | N/A |

---

## Further Reading

- [RISC-V ISA Specification](https://github.com/riscv/riscv-isa-manual)
- [RISC-V Privileged Specification](https://github.com/riscv/riscv-isa-manual)
- [Original Rust blog series](https://osblog.stephenmarz.com/)
- [Source code (GitHub)](https://github.com/sgmarz/osblog)
- [VirtIO Specification](https://docs.oasis-open.org/virtio/virtio/v1.2/virtio-v1.2.html)
- [Minix3 Filesystem](https://wiki.minix3.org/doku.php?id=developersguide:fileformats)
- [NS16550A UART Datasheet](http://www.ti.com/lit/ds/symlink/pc16550d.pdf)