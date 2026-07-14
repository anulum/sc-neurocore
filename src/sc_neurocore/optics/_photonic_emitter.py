# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Photonic netlist emission

"""Emit deterministic Lumerical-compatible netlists from SC IR graphs."""

from __future__ import annotations

from collections import deque
from typing import Any, List


class PhotonicEmitter:
    """Emit photonic netlists in dependency order for a selected PDK."""

    def __init__(self, target_pdk: str = "generic_si_photonics"):
        if not isinstance(target_pdk, str) or not target_pdk.strip():
            raise ValueError("target_pdk must be a non-empty string")
        self.target_pdk = target_pdk

    def _topological_sort(self, nodes: List[Any]) -> List[Any]:
        """Return nodes in stable dependency order and reject malformed graphs."""
        node_ids = [node.id for node in nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("photonic IR node identifiers must be unique")
        outputs = [node.output for node in nodes]
        if len(set(outputs)) != len(outputs):
            raise ValueError("photonic IR node outputs must be unique")

        in_degree = {node.id: 0 for node in nodes}
        node_map = {node.id: node for node in nodes}
        adjacency: dict[str, list[str]] = {node.id: [] for node in nodes}
        output_to_id = {node.output: node.id for node in nodes}

        for node in nodes:
            for input_name in node.inputs:
                if input_name in output_to_id:
                    adjacency[output_to_id[input_name]].append(node.id)
                    in_degree[node.id] += 1

        queue = deque(node_id for node_id, degree in in_degree.items() if degree == 0)
        sorted_nodes: List[Any] = []
        while queue:
            current = queue.popleft()
            sorted_nodes.append(node_map[current])
            for neighbour in adjacency[current]:
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)
        if len(sorted_nodes) != len(nodes):
            raise ValueError("photonic IR graph contains a dependency cycle")
        return sorted_nodes

    def emit_lumerical_netlist(self, ir_graph: Any) -> str:
        """Emit a Lumerical-compatible photonic netlist from an IR graph."""
        if not hasattr(ir_graph, "nodes"):
            raise TypeError("ir_graph must expose a nodes collection")
        sorted_nodes = self._topological_sort(ir_graph.nodes)
        netlist = ["# SC-NeuroCore Photonic Design", f"# PDK: {self.target_pdk}", ""]

        for node in sorted_nodes:
            if node.type == "SC_AND":
                if len(node.inputs) < 2:
                    raise ValueError(f"SC_AND node {node.id!r} requires two inputs")
                netlist.append(f"ADD MZI_MODULATOR {node.id}")
                netlist.append(f"CONNECT {node.id}:in1 {node.inputs[0]}")
                netlist.append(f"CONNECT {node.id}:in2 {node.inputs[1]}")
                netlist.append(f"SET {node.id}:phase_pi 3.14159")
            elif node.type == "LIF_MEMBRANE":
                if not node.inputs:
                    raise ValueError(f"LIF_MEMBRANE node {node.id!r} requires an input")
                netlist.append(f"ADD RESONANT_CAVITY {node.id}")
                netlist.append(f"CONNECT {node.id}:input {node.inputs[0]}")
                netlist.append(f"SET {node.id}:Q_factor 15000")

        return "\n".join(netlist)


__all__ = ["PhotonicEmitter"]
