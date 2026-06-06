# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Debug probe insertion

"""Debug probe insertion utilities for FPGA deployment.

Auto-inserts Xilinx ILA or Intel SignalTap debug cores into generated
neuron modules.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DebugProbeSpec:
    """Auto-generated debug probe specification.

    Attributes
    ----------
    probe_type : str
        ``"ila"`` (Xilinx) or ``"signaltap"`` (Intel).
    signals : list[str]
        Probed signal names.
    depth : int
        Capture depth.
    tcl_commands : str
        Vendor-specific TCL to insert probes.
    """

    probe_type: str
    signals: list[str]
    depth: int
    tcl_commands: str


def insert_debug_probes(
    module_name: str,
    equations: dict[str, str],
    *,
    vendor: str = "xilinx",
    depth: int = 1024,
) -> DebugProbeSpec:
    """Auto-insert ILA/SignalTap debug probes.

    Parameters
    ----------
    module_name : str
        Module name.
    equations : dict[str, str]
        ODE equations (state variables become probed signals).
    vendor : str
        ``"xilinx"`` or ``"intel"``.
    depth : int
        Capture depth in samples.

    Returns
    -------
    DebugProbeSpec
        Probe specification with TCL commands.
    """
    signals = list(equations.keys()) + ["spike_out", "clk", "rst_n"]
    probe_type = "ila" if vendor == "xilinx" else "signaltap"

    if vendor == "xilinx":
        tcl = [
            f"# ILA probe insertion for {module_name}",
            "create_debug_core u_ila_0 ila",
            f"set_property C_DATA_DEPTH {depth} [get_debug_cores u_ila_0]",
        ]
        for sig in signals:
            tcl.append(f"connect_debug_port u_ila_0/probe0 [get_nets {module_name}/{sig}]")
    else:
        tcl = [
            f"# SignalTap probe insertion for {module_name}",
            "set_global_assignment -name ENABLE_SIGNALTAP ON",
        ]
        for sig in signals:
            tcl.append(f"set_instance_assignment -name CONNECT_TO_SLD_NODE {module_name}|{sig}")

    return DebugProbeSpec(
        probe_type=probe_type,
        signals=signals,
        depth=depth,
        tcl_commands="\n".join(tcl),
    )
