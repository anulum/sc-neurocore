# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Chiplet package top and its per-die clock constraints

"""Emit the package top with one clock port per die, and the XDC that times those clocks.

Each die runs on its own clock, ``clk_die_<d>``, at its ``clock_mhz``. A bridge writes on
its source die's clock and reads on its destination die's clock, so the clocks the XDC
creates are exactly the clocks the RTL has, and every link constraint joins two of them.
The one ``rst_n`` input reaches each clock domain through that die's own two-stage
synchroniser (``sc_sync_2ff``): asserted at once, released on two edges of the die's clock.
"""

from __future__ import annotations

import textwrap

from sc_neurocore.chiplet._sv import SPDX_HEADER
from sc_neurocore.chiplet.topology import ChipletTopology


def die_reset(die_id: int) -> str:
    """Return the active-low reset of one die's clock domain, released in that domain.

    Parameters
    ----------
    die_id
        Die identifier.

    Returns
    -------
    str
        ``rst_n_die_<die_id>``, the output of the die's reset synchroniser.
    """
    return f"rst_n_die_{die_id}"


def die_clock(die_id: int) -> str:
    """Return the top-level clock port, and XDC clock name, of one die.

    Parameters
    ----------
    die_id
        Die identifier.

    Returns
    -------
    str
        ``clk_die_<die_id>``.
    """
    return f"clk_die_{die_id}"


def emit_top(topology: ChipletTopology) -> str:
    """Return ``sc_chiplet_top``: dies on their own clocks, joined by clock-crossing bridges.

    Parameters
    ----------
    topology
        Validated die and link graph.

    Returns
    -------
    str
        SystemVerilog source of the package top.
    """
    die_ports = [
        f"    input wire die_{d.die_id}_spike_valid, output wire die_{d.die_id}_spike_ready,\n"
        f"    input wire [{d.aer_id_width - 1}:0] die_{d.die_id}_spike_id,\n"
        f"    output wire die_{d.die_id}_out_valid, input wire die_{d.die_id}_out_ready,\n"
        f"    output wire [{d.aer_id_width - 1}:0] die_{d.die_id}_out_id,\n"
        f"    output wire signed [{d.data_width - 1}:0] die_{d.die_id}_out_weight"
        for d in topology.dies
    ]
    clock_ports = [f"    input wire {die_clock(d.die_id)}" for d in topology.dies]
    top_port_block = ",\n".join([*clock_ports, "    input wire rst_n", *die_ports])
    wire_lines: list[str] = []
    for die in topology.dies:
        d = die.die_id
        wire_lines.append(
            f"    wire {die_reset(d)};\n"
            f"    sc_sync_2ff rst_sync_die_{d} (\n"
            f"        .clk_dst({die_clock(d)}), .rst_dst(!rst_n), .data_in(1'b1),\n"
            f"        .data_out({die_reset(d)})\n    );"
        )
    for lk in topology.links:
        name = f"link_{lk.src_die}_{lk.dst_die}"
        wire_lines.append(
            f"    wire [{lk.data_width - 1}:0] {name}_tdata, {name}_rx_tdata;\n"
            f"    wire {name}_tvalid, {name}_tready, {name}_rx_tvalid, {name}_rx_tready;"
        )
    die_instances: list[str] = []
    for die in topology.dies:
        d = die.die_id
        conns = [
            f"        .link_out_{lk.dst_die}_tdata(link_{lk.src_die}_{lk.dst_die}_tdata),\n"
            f"        .link_out_{lk.dst_die}_tvalid(link_{lk.src_die}_{lk.dst_die}_tvalid),\n"
            f"        .link_out_{lk.dst_die}_tready(link_{lk.src_die}_{lk.dst_die}_tready)"
            for lk in topology.get_links_from(d)
        ]
        conns += [
            f"        .link_in_{lk.src_die}_tdata(link_{lk.src_die}_{lk.dst_die}_rx_tdata),\n"
            f"        .link_in_{lk.src_die}_tvalid(link_{lk.src_die}_{lk.dst_die}_rx_tvalid),\n"
            f"        .link_in_{lk.src_die}_tready(link_{lk.src_die}_{lk.dst_die}_rx_tready)"
            for lk in topology.get_links_to(d)
        ]
        link_block = ",\n" + ",\n".join(conns) if conns else ""
        die_instances.append(
            f"    sc_chiplet_die_{d} die_{d}_inst (\n"
            f"        .clk({die_clock(d)}), .rst_n({die_reset(d)}){link_block},\n"
            f"        .local_spike_valid(die_{d}_spike_valid),\n"
            f"        .local_spike_ready(die_{d}_spike_ready),\n"
            f"        .local_spike_id(die_{d}_spike_id), .local_out_valid(die_{d}_out_valid),\n"
            f"        .local_out_ready(die_{d}_out_ready), .local_out_id(die_{d}_out_id),\n"
            f"        .local_out_weight(die_{d}_out_weight)\n    );"
        )
    bridges: list[str] = []
    for lk in topology.links:
        name = f"link_{lk.src_die}_{lk.dst_die}"
        bridges.append(
            f"    sc_chiplet_bridge_{lk.src_die}_to_{lk.dst_die} "
            f"bridge_{lk.src_die}_{lk.dst_die}_inst (\n"
            f"        .src_clk({die_clock(lk.src_die)}), .src_rst(!{die_reset(lk.src_die)}),\n"
            f"        .dst_clk({die_clock(lk.dst_die)}), .dst_rst(!{die_reset(lk.dst_die)}),\n"
            f"        .s_tdata({name}_tdata), .s_tvalid({name}_tvalid), .s_tready({name}_tready),\n"
            f"        .m_tdata({name}_rx_tdata), .m_tvalid({name}_rx_tvalid),\n"
            f"        .m_tready({name}_rx_tready)\n    );"
        )
    wire_block = "\n".join(wire_lines)
    die_block = "\n\n".join(die_instances)
    bridge_block = "\n\n".join(bridges)
    return textwrap.dedent(f"""\
{SPDX_HEADER}
// SC-NeuroCore Chiplet — Top-level multi-die package
// Dies: {topology.num_dies}
// Links: {len(topology.links)}
// Clocks: one per die (clk_die_<d>); each bridge crosses from its source die's clock
// to its destination die's clock. Reset: rst_n is released in each die's clock domain
// by that die's two-stage synchronizer (rst_n_die_<d>).
module sc_chiplet_top (
{top_port_block}
);
{wire_block}
{die_block}
{bridge_block}
endmodule
""")


def emit_constraints(topology: ChipletTopology) -> str:
    """Return the XDC: one clock per die, the reset's false path, and one bound per link.

    A link's crossing (the FIFO's Gray pointers and data) is bounded by the shorter of its two
    clock periods; the interposer latency itself is modelled by the bridge's ``LATENCY_CYC``.

    Parameters
    ----------
    topology
        Validated die and link graph.

    Returns
    -------
    str
        Xilinx design constraints text.
    """
    hdr = (
        "# SC-NeuroCore Chiplet — Timing constraints\n"
        "# Auto-generated for multi-die package: one clock per die, created on the\n"
        "# sc_chiplet_top port that drives that die and its bridges' sides.\n"
    )
    dies = "\n".join(
        f"# Die {d.die_id}: {d.clock_mhz} MHz\n"
        f"create_clock -name {die_clock(d.die_id)} -period {d.clock_period_ns:.3f} "
        f"[get_ports {die_clock(d.die_id)}]"
        for d in topology.dies
    )
    period = {d.die_id: d.clock_period_ns for d in topology.dies}
    links = "\n".join(
        f"# Link {lk.src_die} → {lk.dst_die}: {lk.latency_ns} ns in the bridge's pipeline; "
        f"crossing bounded by the faster clock\n"
        f"set_max_delay -datapath_only -from [get_clocks {die_clock(lk.src_die)}] "
        f"-to [get_clocks {die_clock(lk.dst_die)}] "
        f"{min(period[lk.src_die], period[lk.dst_die]):.3f}"
        for lk in topology.links
    )
    reset = (
        "# Reset: rst_n is released in each die's domain by its own synchroniser\n"
        "set_false_path -from [get_ports rst_n]"
    )
    return f"{hdr}\n{dies}\n\n{reset}\n\n{links}\n"


__all__ = ["die_clock", "die_reset", "emit_constraints", "emit_top"]
