# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Constraint file generation

"""Constraint file generation utilities for FPGA synthesis.

Auto-generates SDC/XDC timing constraints.
"""

from __future__ import annotations

from typing import Literal


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
        "# SC-NeuroCore deployment utilities",
        f"# Target: {target_freq_mhz:.1f} MHz ({period_ns:.3f} ns period)",
        "",
    ]

    if format == "xdc":
        lines.extend(
            [
                "# ── Clock Definition ─────────────────────────────────────",
                f"create_clock -period {period_ns:.3f} -name {clock_port} [get_ports {clock_port}]",
                "",
                "# ── Input Delays ────────────────────────────────────────",
                f"set_input_delay -clock {clock_port} {io_delay:.3f} [get_ports {reset_port}]",
                f"set_input_delay -clock {clock_port} {io_delay:.3f} [get_ports {{I_t[*]}}]",
                f"set_input_delay -clock {clock_port} {io_delay:.3f} [get_ports en]",
                "",
                "# ── Output Delays ───────────────────────────────────────",
                f"set_output_delay -clock {clock_port} {io_delay:.3f} [get_ports spike_out]",
                "",
                "# ── False Paths ─────────────────────────────────────────",
                f"set_false_path -from [get_ports {reset_port}]",
                "",
                "# ── DSP Multicycle (if pipelined) ───────────────────────",
                "# set_multicycle_path 2 -setup "
                "-from [get_cells -hier *_mul*] -to [get_cells -hier *_t*]",
                "# set_multicycle_path 1 -hold "
                "-from [get_cells -hier *_mul*] -to [get_cells -hier *_t*]",
            ]
        )
    else:  # SDC
        lines.extend(
            [
                "# ── Clock Definition ─────────────────────────────────────",
                f"create_clock -period {period_ns:.3f} -name {clock_port} [get_ports {clock_port}]",
                "",
                "# ── Input Delays ────────────────────────────────────────",
                f"set_input_delay -clock {clock_port} {io_delay:.3f} [get_ports {reset_port}]",
                f"set_input_delay -clock {clock_port} {io_delay:.3f} [get_ports I_t*]",
                "",
                "# ── Output Delays ───────────────────────────────────────",
                f"set_output_delay -clock {clock_port} {io_delay:.3f} [get_ports spike_out]",
                "",
                f"set_false_path -from [get_ports {reset_port}]",
            ]
        )

    lines.append("")
    return "\n".join(lines)
