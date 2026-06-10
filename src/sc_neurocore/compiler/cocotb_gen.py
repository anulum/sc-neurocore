# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cocotb testbench generation

"""Cocotb testbench generation utilities for compiled neuron modules.

Generates Python-based verification testbenches for use with Cocotb and
open-source simulators like Icarus Verilog.
"""

from __future__ import annotations


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
        "",
        "SC-NeuroCore deployment utilities.",
        f"Run: make SIM=icarus TOPLEVEL={module_name} MODULE=test_{module_name}",
        '"""',
        "",
        "import cocotb",
        "from cocotb.clock import Clock",
        "from cocotb.triggers import RisingEdge, Timer",
        "",
        "",
        f"def encode_q(value: float, frac: int = {fraction}) -> int:",
        '    """Encode float to Q-format."""',
        "    return int(round(value * (1 << frac)))",
        "",
        "",
        "@cocotb.test()",
        f"async def test_{module_name}_spikes(dut):",
        f'    """Verify that {module_name} produces spikes with constant current."""',
        "",
        "    # Start clock (10 ns period = 100 MHz)",
        "    clock = Clock(dut.clk, 10, units='ns')",
        "    cocotb.start_soon(clock.start())",
        "",
        "    # Reset",
        "    dut.rst.value = 1",
        "    dut.en.value = 0",
        "    dut.I_t.value = 0",
        "    await RisingEdge(dut.clk)",
        "    await RisingEdge(dut.clk)",
        "    dut.rst.value = 0",
        "    dut.en.value = 1",
        "    await RisingEdge(dut.clk)",
        "",
        "    # Apply constant current",
        f"    dut.I_t.value = {i_encoded}",
        "",
        f"    # Run {n_steps} cycles and count spikes",
        "    spike_count = 0",
        f"    for cycle in range({n_steps}):",
        "        await RisingEdge(dut.clk)",
        "        await Timer(1, units='ns')  # Combinational settling",
        "        if dut.spike_out.value == 1:",
        "            spike_count += 1",
        "",
        f"    dut._log.info(f'Spikes: {{spike_count}} in {n_steps} cycles')",
        "    assert spike_count > 0, 'No spikes detected — check current/threshold'",
        "",
        "",
        "@cocotb.test()",
        f"async def test_{module_name}_no_spike_zero_current(dut):",
        '    """Verify no spikes with zero current."""',
        "",
        "    clock = Clock(dut.clk, 10, units='ns')",
        "    cocotb.start_soon(clock.start())",
        "",
        "    dut.rst.value = 1",
        "    dut.en.value = 0",
        "    dut.I_t.value = 0",
        "    await RisingEdge(dut.clk)",
        "    await RisingEdge(dut.clk)",
        "    dut.rst.value = 0",
        "    dut.en.value = 1",
        "    await RisingEdge(dut.clk)",
        "",
        "    # Zero current",
        "    dut.I_t.value = 0",
        "",
        "    spike_count = 0",
        "    for _ in range(100):",
        "        await RisingEdge(dut.clk)",
        "        await Timer(1, units='ns')",
        "        if dut.spike_out.value == 1:",
        "            spike_count += 1",
        "",
        "    dut._log.info(f'Zero-current spikes: {spike_count}')",
        "    assert spike_count == 0, f'Unexpected spikes with zero current: {spike_count}'",
        "",
        "",
        "@cocotb.test()",
        f"async def test_{module_name}_reset_clears_state(dut):",
        '    """Verify reset returns to initial state."""',
        "",
        "    clock = Clock(dut.clk, 10, units='ns')",
        "    cocotb.start_soon(clock.start())",
        "",
        "    # Drive some current",
        "    dut.rst.value = 0",
        "    dut.en.value = 1",
        f"    dut.I_t.value = {i_encoded}",
        "    for _ in range(50):",
        "        await RisingEdge(dut.clk)",
        "",
        "    # Assert reset",
        "    dut.rst.value = 1",
        "    await RisingEdge(dut.clk)",
        "    await RisingEdge(dut.clk)",
        "    dut.rst.value = 0",
        "    await RisingEdge(dut.clk)",
        "",
        "    # After reset, no spike should fire immediately",
        "    await Timer(1, units='ns')",
        "    assert dut.spike_out.value == 0, 'Spike immediately after reset'",
        "",
    ]

    return "\n".join(lines)
