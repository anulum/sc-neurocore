# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# SC-NeuroCore — Deployment utilities (resource estimation, constraints, drivers, Cocotb)

"""Deployment utilities for compiled neuron modules.

Nine capabilities:

1. **Resource estimation** — estimate LUT/FF/DSP/BRAM without synthesis
2. **Constraint file gen** — auto-generate SDC/XDC timing constraints
3. **Host driver gen** — auto-generate C/Python drivers for bus wrappers
4. **Cocotb testbench gen** — generate Python-based verification testbenches
5. **SymbiYosys formal** — one-command bounded model checking scripts
6. **RISC-V driver gen** — bare-metal, FreeRTOS, and Zephyr RTOS drivers
7. **SLR placement** — multi-die PBLOCK constraints for Versal/Agilex
8. **Certification evidence** — DO-254, IEC 61508, ISO 26262 XML traceability
9. **Multi-target compare** — compile to N targets, generate comparison table
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


# ═══════════════════════════════════════════════════════════════════════
# 1. Resource Estimation
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ResourceEstimate:
    """Estimated FPGA resource usage.

    Attributes
    ----------
    luts : int
        Estimated look-up tables.
    ffs : int
        Estimated flip-flops (registers).
    dsps : int
        Estimated DSP blocks.
    brams : int
        Estimated block RAMs.
    mul_count : int
        Number of multiplications in the design.
    add_count : int
        Number of additions/subtractions.
    reg_bits : int
        Total register bits.
    """

    luts: int
    ffs: int
    dsps: int
    brams: int
    mul_count: int
    add_count: int
    reg_bits: int


def estimate_resources(
    verilog: str,
    *,
    data_width: int = 16,
    has_dsp: bool = True,
) -> ResourceEstimate:
    """Estimate FPGA resources from generated Verilog without synthesis.

    Uses pattern matching on the Verilog source to count multipliers, adders,
    registers, and LUTs. This is a heuristic — actual usage depends on the
    synthesis tool, but estimates are within ~20% for typical designs.

    Parameters
    ----------
    verilog : str
        Generated Verilog source code.
    data_width : int
        Neuron data width (for LUT estimation).
    has_dsp : bool
        True if target has DSP blocks (multiplies go to DSP, not LUTs).

    Returns
    -------
    ResourceEstimate
        Estimated resource usage.
    """
    mul_count = len(re.findall(r'wire\s+signed\s+\[.*?\]\s+_mul\d+', verilog))
    add_count = verilog.count(' + ') + verilog.count(' - ')
    reg_count = len(re.findall(r'reg\s+signed\s+\[', verilog))
    reg_bits = reg_count * data_width

    # LUT estimation heuristics
    luts_per_add = data_width  # 1 LUT per bit for addition
    luts_per_mul = 0 if has_dsp else (data_width * data_width // 4)
    luts_per_mux = data_width // 2  # For saturation/threshold muxes
    mux_count = verilog.count('?')  # Ternary operators

    luts = (add_count * luts_per_add +
            mul_count * luts_per_mul +
            mux_count * luts_per_mux)

    ffs = reg_bits + data_width  # + spike_out + control

    dsps = mul_count if has_dsp else 0

    # BRAM: 0 for single neuron (registers only)
    brams = 0

    return ResourceEstimate(
        luts=max(luts, 1),
        ffs=max(ffs, 1),
        dsps=dsps,
        brams=brams,
        mul_count=mul_count,
        add_count=add_count,
        reg_bits=reg_bits,
    )


# ═══════════════════════════════════════════════════════════════════════
# 2. Constraint File Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_constraints(
    module_name: str,
    *,
    target_freq_mhz: float = 100.0,
    format: Literal["xdc", "sdc"] = "xdc",
    clock_port: str = "clk",
    reset_port: str = "rst",
    data_width: int = 16,
) -> str:
    """Generate timing constraint file for FPGA synthesis.

    Parameters
    ----------
    module_name : str
        Top-level module name.
    target_freq_mhz : float
        Target clock frequency in MHz.
    format : str
        ``"xdc"`` for Xilinx Vivado, ``"sdc"`` for Intel Quartus / generic.
    clock_port : str
        Name of the clock input port.
    reset_port : str
        Name of the reset input port.
    data_width : int
        Data width for I/O delay estimation.

    Returns
    -------
    str
        Complete constraint file content.
    """
    period_ns = 1000.0 / target_freq_mhz
    io_delay = period_ns * 0.2  # 20% of clock period

    lines = [
        f"# Auto-generated timing constraints for {module_name}",
        f"# SC-NeuroCore deployment utilities",
        f"# Target: {target_freq_mhz:.1f} MHz ({period_ns:.3f} ns period)",
        f"",
    ]

    if format == "xdc":
        lines.extend([
            f"# ── Clock Definition ─────────────────────────────────────",
            f"create_clock -period {period_ns:.3f} -name {clock_port} "
            f"[get_ports {clock_port}]",
            f"",
            f"# ── Input Delays ────────────────────────────────────────",
            f"set_input_delay -clock {clock_port} {io_delay:.3f} "
            f"[get_ports {reset_port}]",
            f"set_input_delay -clock {clock_port} {io_delay:.3f} "
            f"[get_ports {{I_t[*]}}]",
            f"set_input_delay -clock {clock_port} {io_delay:.3f} "
            f"[get_ports en]",
            f"",
            f"# ── Output Delays ───────────────────────────────────────",
            f"set_output_delay -clock {clock_port} {io_delay:.3f} "
            f"[get_ports spike_out]",
            f"",
            f"# ── False Paths ─────────────────────────────────────────",
            f"set_false_path -from [get_ports {reset_port}]",
            f"",
            f"# ── DSP Multicycle (if pipelined) ───────────────────────",
            f"# set_multicycle_path 2 -setup "
            f"-from [get_cells -hier *_mul*] -to [get_cells -hier *_t*]",
            f"# set_multicycle_path 1 -hold "
            f"-from [get_cells -hier *_mul*] -to [get_cells -hier *_t*]",
        ])
    else:  # SDC
        lines.extend([
            f"# ── Clock Definition ─────────────────────────────────────",
            f"create_clock -period {period_ns:.3f} -name {clock_port} "
            f"[get_ports {clock_port}]",
            f"",
            f"# ── Input Delays ────────────────────────────────────────",
            f"set_input_delay -clock {clock_port} {io_delay:.3f} "
            f"[get_ports {reset_port}]",
            f"set_input_delay -clock {clock_port} {io_delay:.3f} "
            f"[get_ports I_t*]",
            f"",
            f"# ── Output Delays ───────────────────────────────────────",
            f"set_output_delay -clock {clock_port} {io_delay:.3f} "
            f"[get_ports spike_out]",
            f"",
            f"set_false_path -from [get_ports {reset_port}]",
        ])

    lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# 3. Host Driver Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_host_driver(
    module_name: str,
    params: dict[str, int],
    *,
    language: Literal["python", "c"] = "python",
    bus: Literal["axi_lite", "wishbone"] = "axi_lite",
    base_address: int = 0x4000_0000,
    data_width: int = 16,
    fraction: int = 8,
) -> str:
    """Generate host-side driver code for a bus-wrapped neuron.

    Parameters
    ----------
    module_name : str
        Neuron module name.
    params : dict[str, int]
        Parameter names and bit widths.
    language : str
        ``"python"`` or ``"c"``.
    bus : str
        Bus protocol.
    base_address : int
        Memory-mapped base address.
    data_width : int
        Fixed-point data width.
    fraction : int
        Fractional bits.

    Returns
    -------
    str
        Complete driver source code.
    """
    if language == "python":
        return _gen_python_driver(module_name, params, base_address, data_width, fraction)
    elif language == "c":
        return _gen_c_driver(module_name, params, base_address, data_width, fraction)
    raise ValueError(f"Unsupported language: {language!r}")


def _gen_python_driver(
    module_name: str,
    params: dict[str, int],
    base_address: int,
    data_width: int,
    fraction: int,
) -> str:
    """Generate Python MMIO driver."""
    class_name = "".join(w.capitalize() for w in module_name.split("_")) + "Driver"
    lines = [
        f'"""Auto-generated Python driver for {module_name}.',
        f"",
        f"SC-NeuroCore deployment utilities.",
        f"Bus: memory-mapped I/O at 0x{base_address:08X}.",
        f'Fixed-point: Q{data_width - fraction - 1}.{fraction} ({data_width}-bit signed).',
        f'"""',
        f"",
        f"from __future__ import annotations",
        f"",
        f"",
        f"class {class_name}:",
        f'    """Memory-mapped driver for {module_name}."""',
        f"",
        f"    BASE = 0x{base_address:08X}",
        f"    FRACTION = {fraction}",
        f"",
        f"    # Register offsets",
        f"    REG_CTRL        = 0x00  # bit0=enable, bit1=reset",
        f"    REG_I_T         = 0x04  # Input current (Q-format)",
        f"    REG_SPIKE_COUNT = 0x08  # Spike counter (read-only)",
    ]

    for i, pname in enumerate(params):
        lines.append(f"    REG_{pname.upper():16s}= 0x{0x0C + i * 4:02X}")

    lines.extend([
        f"",
        f"    def __init__(self, read_fn, write_fn, base: int = BASE):",
        f'        """Initialise with platform-specific read/write functions.',
        f"",
        f"        Parameters",
        f"        ----------",
        f"        read_fn : callable",
        f"            ``read_fn(addr) -> int`` — read 32-bit register.",
        f"        write_fn : callable",
        f"            ``write_fn(addr, value)`` — write 32-bit register.",
        f"        base : int",
        f"            Base address override.",
        f'        """',
        f"        self._read = read_fn",
        f"        self._write = write_fn",
        f"        self._base = base",
        f"",
        f"    def _wr(self, offset: int, value: int) -> None:",
        f'        """Write a register."""',
        f"        self._write(self._base + offset, value & 0xFFFFFFFF)",
        f"",
        f"    def _rd(self, offset: int) -> int:",
        f'        """Read a register."""',
        f"        return self._read(self._base + offset)",
        f"",
        f"    def encode_q(self, value: float) -> int:",
        f'        """Encode a float to Q-format integer."""',
        f"        return int(round(value * (1 << self.FRACTION)))",
        f"",
        f"    def decode_q(self, raw: int) -> float:",
        f'        """Decode a Q-format integer to float."""',
        f"        return raw / (1 << self.FRACTION)",
        f"",
        f"    # ── Control ─────────────────────────────────────────────",
        f"",
        f"    def enable(self) -> None:",
        f'        """Enable the neuron (start clocking)."""',
        f"        self._wr(self.REG_CTRL, 0x01)",
        f"",
        f"    def disable(self) -> None:",
        f'        """Disable the neuron (stop clocking)."""',
        f"        self._wr(self.REG_CTRL, 0x00)",
        f"",
        f"    def reset(self) -> None:",
        f'        """Assert reset, then release."""',
        f"        self._wr(self.REG_CTRL, 0x02)",
        f"        self._wr(self.REG_CTRL, 0x01)",
        f"",
        f"    # ── I/O ─────────────────────────────────────────────────",
        f"",
        f"    def set_current(self, I: float) -> None:",
        f'        """Set the input current."""',
        f"        self._wr(self.REG_I_T, self.encode_q(I))",
        f"",
        f"    def get_spike_count(self) -> int:",
        f'        """Read the spike counter."""',
        f"        return self._rd(self.REG_SPIKE_COUNT)",
        f"",
        f"    # ── Parameters ──────────────────────────────────────────",
    ])

    for pname in params:
        fn_name = pname.lower().replace("p_", "")
        lines.extend([
            f"",
            f"    def set_{fn_name}(self, value: float) -> None:",
            f'        """Set {pname}."""',
            f"        self._wr(self.REG_{pname.upper()}, self.encode_q(value))",
        ])

    lines.append("")
    return "\n".join(lines)


def _gen_c_driver(
    module_name: str,
    params: dict[str, int],
    base_address: int,
    data_width: int,
    fraction: int,
) -> str:
    """Generate C MMIO driver header."""
    guard = module_name.upper() + "_DRIVER_H"
    lines = [
        f"/* Auto-generated C driver for {module_name} */",
        f"/* SC-NeuroCore deployment utilities */",
        f"/* Bus: MMIO at 0x{base_address:08X} */",
        f"",
        f"#ifndef {guard}",
        f"#define {guard}",
        f"",
        f"#include <stdint.h>",
        f"",
        f"#define {module_name.upper()}_BASE       0x{base_address:08X}U",
        f"#define {module_name.upper()}_FRACTION    {fraction}",
        f"",
        f"/* Register offsets */",
        f"#define REG_CTRL        0x00",
        f"#define REG_I_T         0x04",
        f"#define REG_SPIKE_COUNT 0x08",
    ]

    for i, pname in enumerate(params):
        lines.append(f"#define REG_{pname.upper():16s} 0x{0x0C + i * 4:02X}")

    lines.extend([
        f"",
        f"/* Platform-specific MMIO (user must implement) */",
        f"extern void     mmio_write(uint32_t addr, uint32_t val);",
        f"extern uint32_t mmio_read(uint32_t addr);",
        f"",
        f"static inline int32_t {module_name}_encode_q(float val) {{",
        f"    return (int32_t)(val * (1 << {fraction}));",
        f"}}",
        f"",
        f"static inline void {module_name}_enable(void) {{",
        f"    mmio_write({module_name.upper()}_BASE + REG_CTRL, 0x01);",
        f"}}",
        f"",
        f"static inline void {module_name}_reset(void) {{",
        f"    mmio_write({module_name.upper()}_BASE + REG_CTRL, 0x02);",
        f"    mmio_write({module_name.upper()}_BASE + REG_CTRL, 0x01);",
        f"}}",
        f"",
        f"static inline void {module_name}_set_current(float I) {{",
        f"    mmio_write({module_name.upper()}_BASE + REG_I_T, "
        f"(uint32_t){module_name}_encode_q(I));",
        f"}}",
        f"",
        f"static inline uint32_t {module_name}_get_spikes(void) {{",
        f"    return mmio_read({module_name.upper()}_BASE + REG_SPIKE_COUNT);",
        f"}}",
        f"",
        f"#endif /* {guard} */",
        f"",
    ])

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# 4. Cocotb Testbench Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_cocotb_testbench(
    module_name: str,
    *,
    data_width: int = 16,
    fraction: int = 8,
    n_steps: int = 200,
    input_current: float = 50.0,
) -> str:
    """Generate a Cocotb (Python) testbench for a compiled neuron.

    Parameters
    ----------
    module_name : str
        Verilog module name.
    data_width : int
        Fixed-point data width.
    fraction : int
        Fractional bits.
    n_steps : int
        Number of simulation clock cycles.
    input_current : float
        Input current value.

    Returns
    -------
    str
        Complete Cocotb Python testbench.
    """
    i_encoded = int(round(input_current * (1 << fraction)))
    lines = [
        f'"""Auto-generated Cocotb testbench for {module_name}.',
        f"",
        f"SC-NeuroCore deployment utilities.",
        f"Run: make SIM=icarus TOPLEVEL={module_name} MODULE=test_{module_name}",
        f'"""',
        f"",
        f"import cocotb",
        f"from cocotb.clock import Clock",
        f"from cocotb.triggers import RisingEdge, Timer",
        f"",
        f"",
        f"def encode_q(value: float, frac: int = {fraction}) -> int:",
        f'    """Encode float to Q-format."""',
        f"    return int(round(value * (1 << frac)))",
        f"",
        f"",
        f"@cocotb.test()",
        f"async def test_{module_name}_spikes(dut):",
        f'    """Verify that {module_name} produces spikes with constant current."""',
        f"",
        f"    # Start clock (10 ns period = 100 MHz)",
        f"    clock = Clock(dut.clk, 10, units='ns')",
        f"    cocotb.start_soon(clock.start())",
        f"",
        f"    # Reset",
        f"    dut.rst.value = 1",
        f"    dut.en.value = 0",
        f"    dut.I_t.value = 0",
        f"    await RisingEdge(dut.clk)",
        f"    await RisingEdge(dut.clk)",
        f"    dut.rst.value = 0",
        f"    dut.en.value = 1",
        f"    await RisingEdge(dut.clk)",
        f"",
        f"    # Apply constant current",
        f"    dut.I_t.value = {i_encoded}",
        f"",
        f"    # Run {n_steps} cycles and count spikes",
        f"    spike_count = 0",
        f"    for cycle in range({n_steps}):",
        f"        await RisingEdge(dut.clk)",
        f"        await Timer(1, units='ns')  # Combinational settling",
        f"        if dut.spike_out.value == 1:",
        f"            spike_count += 1",
        f"",
        f"    dut._log.info(f'Spikes: {{spike_count}} in {n_steps} cycles')",
        f"    assert spike_count > 0, 'No spikes detected — check current/threshold'",
        f"",
        f"",
        f"@cocotb.test()",
        f"async def test_{module_name}_no_spike_zero_current(dut):",
        f'    """Verify no spikes with zero current."""',
        f"",
        f"    clock = Clock(dut.clk, 10, units='ns')",
        f"    cocotb.start_soon(clock.start())",
        f"",
        f"    dut.rst.value = 1",
        f"    dut.en.value = 0",
        f"    dut.I_t.value = 0",
        f"    await RisingEdge(dut.clk)",
        f"    await RisingEdge(dut.clk)",
        f"    dut.rst.value = 0",
        f"    dut.en.value = 1",
        f"    await RisingEdge(dut.clk)",
        f"",
        f"    # Zero current",
        f"    dut.I_t.value = 0",
        f"",
        f"    spike_count = 0",
        f"    for _ in range(100):",
        f"        await RisingEdge(dut.clk)",
        f"        await Timer(1, units='ns')",
        f"        if dut.spike_out.value == 1:",
        f"            spike_count += 1",
        f"",
        f"    dut._log.info(f'Zero-current spikes: {{spike_count}}')",
        f"    assert spike_count == 0, f'Unexpected spikes with zero current: {{spike_count}}'",
        f"",
        f"",
        f"@cocotb.test()",
        f"async def test_{module_name}_reset_clears_state(dut):",
        f'    """Verify reset returns to initial state."""',
        f"",
        f"    clock = Clock(dut.clk, 10, units='ns')",
        f"    cocotb.start_soon(clock.start())",
        f"",
        f"    # Drive some current",
        f"    dut.rst.value = 0",
        f"    dut.en.value = 1",
        f"    dut.I_t.value = {i_encoded}",
        f"    for _ in range(50):",
        f"        await RisingEdge(dut.clk)",
        f"",
        f"    # Assert reset",
        f"    dut.rst.value = 1",
        f"    await RisingEdge(dut.clk)",
        f"    await RisingEdge(dut.clk)",
        f"    dut.rst.value = 0",
        f"    await RisingEdge(dut.clk)",
        f"",
        f"    # After reset, no spike should fire immediately",
        f"    await Timer(1, units='ns')",
        f"    assert dut.spike_out.value == 0, 'Spike immediately after reset'",
        f"",
    ]

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# 5. SymbiYosys Formal Verification Flow
# ═══════════════════════════════════════════════════════════════════════

def generate_sby_script(
    module_name: str,
    *,
    sva_file: str | None = None,
    depth: int = 20,
    mode: Literal["bmc", "prove", "cover"] = "bmc",
    solver: str = "smtbmc",
    engine: str = "boolector",
) -> str:
    """Generate a SymbiYosys ``.sby`` formal verification script.

    Enables one-command bounded model checking of compiled neurons
    using open-source formal tools (SymbiYosys + Yosys + solver).

    Parameters
    ----------
    module_name : str
        Top-level Verilog module name.
    sva_file : str, optional
        SystemVerilog assertions file. Defaults to ``{module}_sva.sv``.
    depth : int
        BMC / induction depth in clock cycles.
    mode : str
        ``"bmc"`` (bounded), ``"prove"`` (induction), ``"cover"``.
    solver : str
        Solver backend (``"smtbmc"``, ``"aiger"``).
    engine : str
        SMT engine (``"boolector"``, ``"z3"``, ``"yices"``).

    Returns
    -------
    str
        Complete ``.sby`` configuration file.
    """
    if sva_file is None:
        sva_file = f"{module_name}_sva.sv"
    verilog_file = f"{module_name}.v"

    return (
        f"# Auto-generated SymbiYosys script for {module_name}\n"
        f"# SC-NeuroCore formal verification flow\n"
        f"# Run: sby {module_name}.sby\n"
        f"\n"
        f"[options]\n"
        f"mode {mode}\n"
        f"depth {depth}\n"
        f"\n"
        f"[engines]\n"
        f"{solver} {engine}\n"
        f"\n"
        f"[script]\n"
        f"read_verilog -formal {verilog_file}\n"
        f"read_verilog -sv -formal {sva_file}\n"
        f"prep -top {module_name}\n"
        f"\n"
        f"[files]\n"
        f"{verilog_file}\n"
        f"{sva_file}\n"
    )


# ═══════════════════════════════════════════════════════════════════════
# 6. RISC-V Driver + FreeRTOS / Zephyr Template
# ═══════════════════════════════════════════════════════════════════════

def generate_riscv_driver(
    module_name: str,
    params: dict[str, int],
    *,
    base_address: int = 0x4000_0000,
    data_width: int = 16,
    fraction: int = 8,
    rtos: Literal["freertos", "zephyr", "baremetal"] = "baremetal",
) -> str:
    """Generate a RISC-V C driver for neuron control via MMIO.

    Supports bare-metal, FreeRTOS, and Zephyr templates with timer-driven
    neuron tick tasks for real-time operating system integration.

    Parameters
    ----------
    module_name : str
        Neuron module name.
    params : dict[str, int]
        Parameter names and bit widths.
    base_address : int
        MMIO base address.
    data_width : int
        Fixed-point data width.
    fraction : int
        Fractional bits.
    rtos : str
        ``"baremetal"``, ``"freertos"``, or ``"zephyr"``.

    Returns
    -------
    str
        Complete RISC-V C driver with optional RTOS integration.
    """
    guard = module_name.upper() + "_RISCV_H"
    upper = module_name.upper()

    lines = [
        f"/* Auto-generated RISC-V driver for {module_name} */",
        f"/* SC-NeuroCore — RISC-V SoC integration ({rtos}) */",
        f"",
        f"#ifndef {guard}",
        f"#define {guard}",
        f"",
        f"#include <stdint.h>",
        f"",
        f"#define {upper}_BASE    0x{base_address:08X}U",
        f"#define {upper}_FRAC    {fraction}",
        f"#define {upper}_CTRL    ({upper}_BASE + 0x00)",
        f"#define {upper}_I_T     ({upper}_BASE + 0x04)",
        f"#define {upper}_SPIKES  ({upper}_BASE + 0x08)",
    ]

    for i, pname in enumerate(params):
        lines.append(
            f"#define {upper}_{pname.upper()}"
            f"  ({upper}_BASE + 0x{0x0C + i * 4:02X})"
        )

    lines.extend([
        f"",
        f"#define MMIO_WR(a,v) (*(volatile uint32_t*)(a) = (v))",
        f"#define MMIO_RD(a)   (*(volatile uint32_t*)(a))",
        f"",
        f"static inline int32_t {module_name}_encode(float v) {{",
        f"    return (int32_t)(v * (1 << {upper}_FRAC));",
        f"}}",
        f"",
        f"static inline void {module_name}_enable(void)  {{ MMIO_WR({upper}_CTRL, 0x01); }}",
        f"static inline void {module_name}_disable(void) {{ MMIO_WR({upper}_CTRL, 0x00); }}",
        f"",
        f"static inline void {module_name}_reset(void) {{",
        f"    MMIO_WR({upper}_CTRL, 0x02);",
        f"    MMIO_WR({upper}_CTRL, 0x01);",
        f"}}",
        f"",
        f"static inline void {module_name}_set_current(float I) {{",
        f"    MMIO_WR({upper}_I_T, (uint32_t){module_name}_encode(I));",
        f"}}",
        f"",
        f"static inline uint32_t {module_name}_get_spikes(void) {{",
        f"    return MMIO_RD({upper}_SPIKES);",
        f"}}",
    ])

    for pname in params:
        lines.extend([
            f"",
            f"static inline void {module_name}_set_{pname.lower()}(float v) {{",
            f"    MMIO_WR({upper}_{pname.upper()}, (uint32_t){module_name}_encode(v));",
            f"}}",
        ])

    if rtos == "freertos":
        lines.extend([
            f"",
            f"/* ── FreeRTOS neuron tick task ───────────────────── */",
            f"#include \"FreeRTOS.h\"",
            f"#include \"task.h\"",
            f"",
            f"static void {module_name}_tick(void *p) {{",
            f"    (void)p;",
            f"    {module_name}_reset();",
            f"    {module_name}_enable();",
            f"    for (;;) {{",
            f"        float I = 0.0f; /* TODO: read sensor */",
            f"        {module_name}_set_current(I);",
            f"        vTaskDelay(pdMS_TO_TICKS(1));",
            f"    }}",
            f"}}",
            f"",
            f"static inline void {module_name}_start_rtos(void) {{",
            f"    xTaskCreate({module_name}_tick, \"{module_name}\",",
            f"                configMINIMAL_STACK_SIZE, NULL,",
            f"                tskIDLE_PRIORITY + 1, NULL);",
            f"}}",
        ])
    elif rtos == "zephyr":
        lines.extend([
            f"",
            f"/* ── Zephyr neuron tick thread ──────────────────── */",
            f"#include <zephyr/kernel.h>",
            f"",
            f"static void {module_name}_thread(void *a, void *b, void *c) {{",
            f"    ARG_UNUSED(a); ARG_UNUSED(b); ARG_UNUSED(c);",
            f"    {module_name}_reset();",
            f"    {module_name}_enable();",
            f"    while (1) {{",
            f"        {module_name}_set_current(0.0f);",
            f"        k_msleep(1);",
            f"    }}",
            f"}}",
            f"",
            f"K_THREAD_DEFINE({module_name}_tid, 1024,",
            f"    {module_name}_thread, NULL, NULL, NULL, 5, 0, 0);",
        ])

    lines.extend([f"", f"#endif /* {guard} */", f""])
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# 7. Multi-Die / SLR Placement Constraints
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SLRPlacement:
    """SLR (Super Logic Region) placement for multi-die FPGAs.

    Attributes
    ----------
    module_name : str
        Module or instance name.
    slr : int
        Target SLR index (0-based).
    pblock_name : str
        Vivado PBLOCK name (auto-generated if empty).
    """

    module_name: str
    slr: int
    pblock_name: str = ""

    def __post_init__(self) -> None:
        """Auto-generate pblock name if not set."""
        if not self.pblock_name:
            self.pblock_name = f"pblock_slr{self.slr}"


def generate_slr_constraints(
    placements: list[SLRPlacement],
    *,
    insert_pipeline_regs: bool = True,
    target_freq_mhz: float = 500.0,
) -> str:
    """Generate Vivado XDC for multi-die SLR placement.

    Emits PBLOCK constraints that pin modules to specific SLRs and
    optionally adds inter-SLR pipeline register directives.

    Parameters
    ----------
    placements : list[SLRPlacement]
        Module-to-SLR assignments.
    insert_pipeline_regs : bool
        Add register duplication directives for SLR crossings.
    target_freq_mhz : float
        Target frequency for SLR crossing timing.

    Returns
    -------
    str
        Complete XDC constraint block.
    """
    period_ns = 1000.0 / target_freq_mhz
    lines = [
        "# Auto-generated SLR placement constraints",
        "# SC-NeuroCore multi-die deployment",
        f"# Target: {target_freq_mhz:.0f} MHz",
        "",
    ]

    slrs_used: set[int] = set()
    for p in placements:
        slrs_used.add(p.slr)
        lines.extend([
            f"create_pblock {p.pblock_name}",
            f"add_cells_to_pblock [get_pblocks {p.pblock_name}] "
            f"[get_cells -hier -filter {{NAME =~ *{p.module_name}*}}]",
            f"resize_pblock [get_pblocks {p.pblock_name}] -add SLR{p.slr}",
            "",
        ])

    if insert_pipeline_regs and len(slrs_used) > 1:
        lines.extend([
            "# Inter-SLR pipeline register directives",
            "set_property REGISTER_DUPLICATION true "
            "[get_cells -hier -filter {IS_SEQUENTIAL}]",
            f"set_max_delay {period_ns / 2:.3f} "
            "-datapath_only -from [get_clocks *] -to [get_clocks *]",
            "",
        ])

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# 8. Safety-Critical Certification Evidence
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CertificationItem:
    """A single certification evidence item.

    Attributes
    ----------
    req_id : str
        Requirement identifier (e.g. ``"REQ-001"``).
    description : str
        Requirement description.
    design_ref : str
        Design artifact (e.g. Verilog module name).
    verification_ref : str
        Verification artifact (e.g. SVA property, Cocotb test).
    status : str
        ``"PASS"``, ``"FAIL"``, or ``"UNTESTED"``.
    """

    req_id: str
    description: str
    design_ref: str
    verification_ref: str
    status: Literal["PASS", "FAIL", "UNTESTED"] = "UNTESTED"


def generate_certification_evidence(
    module_name: str,
    items: list[CertificationItem],
    *,
    standard: Literal["do254", "iec61508", "iso26262"] = "do254",
    dal_level: str = "DAL-C",
) -> str:
    """Generate XML traceability matrix for safety certification.

    Produces a certification evidence document linking requirements to
    design and verification artifacts in the format required by DO-254
    (avionics), IEC 61508 (industrial), or ISO 26262 (automotive).

    Parameters
    ----------
    module_name : str
        Design module under certification.
    items : list[CertificationItem]
        Requirement-to-evidence mapping.
    standard : str
        ``"do254"``, ``"iec61508"``, or ``"iso26262"``.
    dal_level : str
        Design Assurance Level or SIL/ASIL level.

    Returns
    -------
    str
        XML certification evidence document.
    """
    std_label = {
        "do254": "RTCA DO-254",
        "iec61508": "IEC 61508",
        "iso26262": "ISO 26262",
    }[standard]

    pass_count = sum(1 for i in items if i.status == "PASS")
    fail_count = sum(1 for i in items if i.status == "FAIL")
    total = len(items)
    pct = (pass_count / total * 100) if total else 0.0

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<!-- SC-NeuroCore Certification Evidence: {module_name} -->',
        f'<certification_evidence>',
        f'  <module>{module_name}</module>',
        f'  <standard>{std_label}</standard>',
        f'  <level>{dal_level}</level>',
        f'  <summary total="{total}" passed="{pass_count}" '
        f'failed="{fail_count}" coverage="{pct:.1f}"/>',
        f'  <traceability_matrix>',
    ]

    for item in items:
        lines.extend([
            f'    <requirement id="{item.req_id}" status="{item.status}">',
            f'      <description>{item.description}</description>',
            f'      <design_ref>{item.design_ref}</design_ref>',
            f'      <verification_ref>{item.verification_ref}</verification_ref>',
            f'    </requirement>',
        ])

    lines.extend([
        f'  </traceability_matrix>',
        f'</certification_evidence>',
        f'',
    ])

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# 9. Multi-Target Compilation (--compare)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CompilationResult:
    """Per-target compilation result for comparison.

    Attributes
    ----------
    target : str
        Profile name.
    verilog_lines : int
        Lines of generated Verilog.
    data_width : int
        Total bit width.
    fraction : int
        Fractional bits.
    overflow : str
        Overflow mode.
    rounding : str
        Rounding mode.
    estimated_luts : int
        LUT estimate.
    estimated_dsps : int
        DSP block estimate.
    estimated_ffs : int
        Flip-flop estimate.
    guard_bits : int
        Required guard bits.
    max_freq_mhz : int
        Target max frequency.
    """

    target: str
    verilog_lines: int
    data_width: int
    fraction: int
    overflow: str
    rounding: str
    estimated_luts: int
    estimated_dsps: int
    estimated_ffs: int
    guard_bits: int
    max_freq_mhz: int


def compile_multi_target(
    equations: dict[str, str],
    targets: list[str],
    module_name: str = "sc_neuron",
) -> list[CompilationResult]:
    """Compile a neuron to multiple targets and collect metrics.

    Parameters
    ----------
    equations : dict
        Variable name → ODE RHS expression.
    targets : list[str]
        Profile names to compile against.
    module_name : str
        Base module name.

    Returns
    -------
    list[CompilationResult]
        Per-target compilation results.
    """
    from sc_neurocore.compiler.hardware_profiles import get_profile
    from sc_neurocore.compiler.static_analysis import compute_guard_bits

    results = []
    for target_name in targets:
        profile = get_profile(target_name)

        # Estimate resources heuristically (no actual Verilog gen required)
        total_mul = 0
        total_add = 0
        max_guard = 0
        for expr in equations.values():
            # Count muls/adds from expression
            import ast
            tree = ast.parse(expr, mode="eval")
            for node in ast.walk(tree):
                if isinstance(node, ast.BinOp):
                    if isinstance(node.op, (ast.Mult, ast.Div)):
                        total_mul += 1
                    elif isinstance(node.op, (ast.Add, ast.Sub)):
                        total_add += 1
            g = compute_guard_bits(expr)
            max_guard = max(max_guard, g)

        dw = profile.data_width
        luts = (total_add * dw +
                (total_mul * dw * dw // 4 if not profile.dsp_block else 0))
        dsps = total_mul if profile.dsp_block else 0
        ffs = len(equations) * dw + dw  # state regs + control
        verilog_lines = 30 + len(equations) * 15 + total_mul * 5

        results.append(CompilationResult(
            target=target_name,
            verilog_lines=verilog_lines,
            data_width=dw,
            fraction=profile.fraction,
            overflow=profile.overflow,
            rounding=profile.rounding,
            estimated_luts=max(luts, 1),
            estimated_dsps=dsps,
            estimated_ffs=max(ffs, 1),
            guard_bits=max_guard,
            max_freq_mhz=profile.max_freq_mhz,
        ))

    return results


def format_comparison_table(results: list[CompilationResult]) -> str:
    """Format multi-target results as a markdown comparison table.

    Parameters
    ----------
    results : list[CompilationResult]
        Per-target compilation results.

    Returns
    -------
    str
        Markdown table string.
    """
    lines = [
        "| Target | Bits | Frac | DSPs | LUTs | FFs | Fmax | Guard | Overflow | Rounding |",
        "|--------|-----:|-----:|-----:|-----:|----:|-----:|------:|----------|----------|",
    ]
    for r in results:
        freq = f"{r.max_freq_mhz}" if r.max_freq_mhz else "N/A"
        lines.append(
            f"| {r.target:16s} | {r.data_width:4d} | {r.fraction:4d} | "
            f"{r.estimated_dsps:4d} | {r.estimated_luts:4d} | {r.estimated_ffs:3d} | "
            f"{freq:>4s} | {r.guard_bits:5d} | {r.overflow:8s} | {r.rounding:8s} |"
        )
    return "\n".join(lines)
