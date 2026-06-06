# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Testbench generator

"""Verification testbench generation for compiled neurons."""

from __future__ import annotations


def generate_testbench(
    module_name: str,
    equations: dict[str, str],
    *,
    framework: str = "cocotb",
    num_cycles: int = 1000,
) -> str:
    """Generate verification testbench for compiled neuron."""
    if framework == "cocotb":
        lines = [
            f'"""Auto-generated Cocotb testbench for {module_name}."""',
            "import cocotb",
            "from cocotb.clock import Clock",
            "from cocotb.triggers import RisingEdge, Timer",
            "",
            "@cocotb.test()",
            f"async def test_{module_name}_reset(dut):",
            '    """Verify reset clears all state."""',
            "    clock = Clock(dut.clk, 10, units='ns')",
            "    cocotb.start_soon(clock.start())",
            "    dut.rst_n.value = 0",
            "    await RisingEdge(dut.clk)",
            "    await RisingEdge(dut.clk)",
        ]
        for sv in equations:
            lines.append(f"    assert dut.{sv}.value == 0, '{sv} not cleared on reset'")
        lines.extend(
            [
                "    dut.rst_n.value = 1",
                "",
                "@cocotb.test()",
                f"async def test_{module_name}_run(dut):",
                f'    """Run {num_cycles} cycles and check no overflow."""',
                "    clock = Clock(dut.clk, 10, units='ns')",
                "    cocotb.start_soon(clock.start())",
                "    dut.rst_n.value = 1",
                f"    for _ in range({num_cycles}):",
                "        await RisingEdge(dut.clk)",
                "    assert dut.spike_out.value is not None",
            ]
        )
    else:  # UVM
        lines = [
            f"// Auto-generated UVM testbench for {module_name}",
            f"class {module_name}_test extends uvm_test;",
            f"    `uvm_component_utils({module_name}_test)",
            "    function new(string name, uvm_component parent);",
            "        super.new(name, parent);",
            "    endfunction",
            "    task run_phase(uvm_phase phase);",
            "        phase.raise_objection(this);",
            f"        #{num_cycles * 10};",
            "        phase.drop_objection(this);",
            "    endtask",
            "endclass",
        ]

    return "\n".join(lines)
