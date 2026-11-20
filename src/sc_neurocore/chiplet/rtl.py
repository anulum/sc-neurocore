# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Multi-die AER and AXI-Stream RTL generator

"""Emit die wrappers, CDC bridges, routing tables, package RTL, and XDC."""

from __future__ import annotations

from dataclasses import dataclass

from sc_neurocore.chiplet._rtl_die import emit_die_wrapper
from sc_neurocore.chiplet._rtl_links import emit_bridge, emit_routing_table
from sc_neurocore.chiplet._rtl_top import emit_constraints, emit_top
from sc_neurocore.chiplet.routing import (
    RoutingTable,
    compute_decorrelation_seeds,
    simulate_timing,
)
from sc_neurocore.chiplet.topology import ChipletDie, ChipletTopology


def _packet_widths(topology: ChipletTopology) -> tuple[int, int, int, int]:
    max_die_id = max((die.die_id for die in topology.dies), default=0)
    die_id_w = max(4, max_die_id.bit_length() + 1)
    pkg_aer_w = max((die.aer_id_width for die in topology.dies), default=10)
    pkg_data_w = max((die.data_width for die in topology.dies), default=16)
    return die_id_w, pkg_aer_w, pkg_data_w, die_id_w + pkg_data_w + pkg_aer_w


def _validate_inputs(
    topology: ChipletTopology,
    routing: dict[int, RoutingTable] | None,
    pkt_w: int,
) -> dict[int, dict[int, int]]:
    die_by_id: dict[int, ChipletDie] = {}
    for die in topology.dies:
        if die.die_id in die_by_id:
            raise ValueError(f"Duplicate die_id {die.die_id} in ChipletTopology")
        die_by_id[die.die_id] = die
    seen_links: set[tuple[int, int]] = set()
    for link in topology.links:
        if link.src_die not in die_by_id or link.dst_die not in die_by_id:
            raise ValueError(f"Link ({link.src_die} -> {link.dst_die}) references absent die_id")
        edge = (link.src_die, link.dst_die)
        if edge in seen_links:
            raise ValueError(f"Duplicate directed link ({link.src_die} -> {link.dst_die})")
        seen_links.add(edge)
        if link.data_width < pkt_w:
            raise ValueError(
                f"Link ({link.src_die} -> {link.dst_die}) data_width={link.data_width} < {pkt_w}"
            )
    next_hops: dict[int, dict[int, int]] = {die_id: {} for die_id in die_by_id}
    for src_id in die_by_id:
        for dst_id in die_by_id:
            if src_id != dst_id and (timed := simulate_timing(topology, src_id, dst_id)):
                next_hops[src_id][dst_id] = timed.path[1]
    if routing is not None:
        for die_id, table in routing.items():
            if die_id not in die_by_id or table.die_id != die_id:
                raise ValueError(f"Invalid routing table key or die_id for die {die_id}")
            src_die = die_by_id[die_id]
            src_lim = min(src_die.n_neurons, 1 << src_die.aer_id_width)
            for entry in table.entries:
                if not 0 <= entry.src_neuron < src_lim:
                    raise ValueError(f"src_neuron {entry.src_neuron} out of range on die {die_id}")
                if entry.dst_die not in die_by_id:
                    raise ValueError(f"dst_die {entry.dst_die} is absent from topology")
                dst_die = die_by_id[entry.dst_die]
                dst_lim = min(dst_die.n_neurons, 1 << dst_die.aer_id_width)
                eff_w = min(src_die.data_width, dst_die.data_width)
                if not 0 <= entry.dst_neuron < dst_lim:
                    raise ValueError(f"dst_neuron {entry.dst_neuron} out of range")
                if not -(1 << (eff_w - 1)) <= entry.weight_q88 <= (1 << (eff_w - 1)) - 1:
                    raise ValueError(f"weight_q88 {entry.weight_q88} out of signed range")
                if entry.dst_die != die_id and entry.dst_die not in next_hops[die_id]:
                    raise ValueError(f"No directed link path from die {die_id} to {entry.dst_die}")
    return next_hops


@dataclass
class ChipletOutput:
    """Contain generated source and constraint artefacts for one topology."""

    top_sv: str
    die_modules: dict[int, str]
    link_bridges: dict[tuple[int, int], str]
    routing_tables: dict[int, str]
    constraints_xdc: str
    filelist: list[str]

    def to_dict(self) -> dict[str, str]:
        """Return generated artefacts keyed by their output filename."""
        return {
            "sc_chiplet_top.sv": self.top_sv,
            **{f"sc_chiplet_die_{d}.sv": s for d, s in self.die_modules.items()},
            **{f"sc_chiplet_bridge_{u}_to_{v}.sv": s for (u, v), s in self.link_bridges.items()},
            **{f"sc_chiplet_rtable_{d}.sv": s for d, s in self.routing_tables.items()},
            "chiplet_constraints.xdc": self.constraints_xdc,
        }


class ChipletGenerator:
    """Generate connected multi-die routing RTL from a chiplet topology."""

    def generate(
        self,
        topology: ChipletTopology,
        routing: dict[int, RoutingTable] | None = None,
        *,
        decorrelate_links: bool = False,
    ) -> ChipletOutput:
        """Generate package RTL and timing constraints.

        Parameters
        ----------
        topology : ChipletTopology
            Directed die and interposer link graph. Per-die ``local_spike_id``
            inputs must stay strictly below ``die.n_neurons``.
        routing : dict[int, RoutingTable] | None, optional
            Optional per-die AER routing tables.
        decorrelate_links : bool, optional
            Scramble bridge ``s_tdata`` before the CDC FIFO and descramble after
            the latency pipeline from a shared per-link LFSR seed when ``True``
            (default ``False``), decorrelating in-flight interposer lanes while
            delivering bit-exact AER packets end to end.

        Returns
        -------
        ChipletOutput
            Generated top, die, bridge, route-table, and XDC artefacts.
        """
        die_id_w, pkg_aer_w, pkg_data_w, pkt_w = _packet_widths(topology)
        next_hops = _validate_inputs(topology, routing, pkt_w)
        seeds = compute_decorrelation_seeds(topology)
        die_modules: dict[int, str] = {}
        link_bridges: dict[tuple[int, int], str] = {}
        routing_tables: dict[int, str] = {}
        die_by_id = {die.die_id: die for die in topology.dies}
        for die in topology.dies:
            table = routing.get(die.die_id) if routing else None
            die_modules[die.die_id] = emit_die_wrapper(
                die, topology, table, next_hops[die.die_id], die_id_w, pkg_aer_w, pkg_data_w
            )
            if table is not None:
                routing_tables[die.die_id] = emit_routing_table(table, die, die_id_w, pkg_aer_w)
        for link in topology.links:
            seed = seeds.get((link.src_die, link.dst_die), 0xACE1)
            link_bridges[(link.src_die, link.dst_die)] = emit_bridge(
                link, die_by_id[link.src_die], die_by_id[link.dst_die], seed, decorrelate_links
            )
        top, xdc = emit_top(topology), emit_constraints(topology)
        output = ChipletOutput(top, die_modules, link_bridges, routing_tables, xdc, [])
        output.filelist = list(output.to_dict())
        return output


__all__ = ["ChipletGenerator", "ChipletOutput"]
