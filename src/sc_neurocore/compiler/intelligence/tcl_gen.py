# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — TCL project generation

"""FPGA project TCL generation utilities.

Generates project creation and build scripts for Xilinx Vivado and
Intel Quartus.
"""

from __future__ import annotations

from typing import Literal


def generate_tcl_project(
    module_name: str,
    *,
    tool: Literal["vivado", "quartus"] = "vivado",
    part: str = "xc7a35tcpg236-1",
    verilog_files: list[str] | None = None,
    constraint_file: str | None = None,
) -> str:
    """Generate FPGA project TCL script.

    Parameters
    ----------
    module_name : str
        Top-level module name.
    tool : str
        ``"vivado"`` or ``"quartus"``.
    part : str
        Target FPGA part number.
    verilog_files : list, optional
        Verilog source files.
    constraint_file : str, optional
        Constraint file (XDC/SDC).

    Returns
    -------
    str
        Complete TCL script.
    """
    if verilog_files is None:
        verilog_files = [f"{module_name}.v"]

    if tool == "vivado":
        return _gen_vivado_tcl(module_name, part, verilog_files, constraint_file)
    elif tool == "quartus":
        return _gen_quartus_tcl(module_name, part, verilog_files, constraint_file)
    raise ValueError(f"Unsupported tool: {tool!r}")


def _gen_vivado_tcl(
    module_name: str,
    part: str,
    verilog_files: list[str],
    constraint_file: str | None,
) -> str:
    """Generate Xilinx Vivado project TCL."""
    lines = [
        f"# Auto-generated Vivado project TCL for {module_name}",
        "# SC-NeuroCore deployment utilities",
        "",
        f"create_project {module_name} ./{module_name}_project -part {part} -force",
        "set_property target_language Verilog [current_project]",
        "",
        "# Add source files",
    ]

    for vf in verilog_files:
        lines.append(f"add_files {vf}")

    if constraint_file:
        lines.extend(
            [
                "",
                "# Add constraints",
                f"add_files -fileset constrs_1 {constraint_file}",
            ]
        )

    lines.extend(
        [
            "",
            "# Set top module",
            f"set_property top {module_name} [current_fileset]",
            "",
            "# Run synthesis",
            f"synth_design -top {module_name} -part {part}",
            "",
            "# Run implementation",
            "opt_design",
            "place_design",
            "route_design",
            "",
            "# Reports",
            f"report_utilization -file {module_name}_util.rpt",
            f"report_timing_summary -file {module_name}_timing.rpt",
            f"report_power -file {module_name}_power.rpt",
            "",
            "# Generate bitstream",
            f"write_bitstream -force {module_name}.bit",
            "",
            f'puts "Build complete: {module_name}.bit"',
            "",
        ]
    )

    return "\n".join(lines)


def _gen_quartus_tcl(
    module_name: str,
    part: str,
    verilog_files: list[str],
    constraint_file: str | None,
) -> str:
    """Generate Intel Quartus project TCL."""
    lines = [
        f"# Auto-generated Quartus project TCL for {module_name}",
        "# SC-NeuroCore deployment utilities",
        "",
        "package require ::quartus::project",
        "",
        f"project_new {module_name} -overwrite",
        'set_global_assignment -name FAMILY "Cyclone V"',
        f"set_global_assignment -name DEVICE {part}",
        f"set_global_assignment -name TOP_LEVEL_ENTITY {module_name}",
        "",
    ]

    for vf in verilog_files:
        lines.append(f"set_global_assignment -name VERILOG_FILE {vf}")

    if constraint_file:
        lines.append(f"set_global_assignment -name SDC_FILE {constraint_file}")

    lines.extend(
        [
            "",
            "# Compile",
            "execute_flow -compile",
            "",
            "project_close",
            "",
        ]
    )

    return "\n".join(lines)
