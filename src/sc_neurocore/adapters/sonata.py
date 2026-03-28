# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SONATA network format importer

"""Import SONATA network files (nodes.h5 + edges.h5) into SC-NeuroCore.

SONATA is the standard network interchange format used by Allen Institute,
Blue Brain Project, and the BRAIN Initiative. It stores network topology
in HDF5 with separate files for nodes (neurons) and edges (synapses).

Dai et al. (2020). The SONATA data format for efficient description of
large-scale network models. PLoS Comput Biol 16(2):e1007696.

Requires: h5py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class SONATANode:
    """A single node (neuron) from a SONATA population."""

    node_id: int
    node_type_id: int
    model_type: str = "point_neuron"
    model_template: str = ""
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class SONATAEdge:
    """A single edge (synapse) from a SONATA population."""

    source_id: int
    target_id: int
    edge_type_id: int
    weight: float = 1.0
    delay: float = 0.0
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class SONATANetwork:
    """Parsed SONATA network with nodes and edges."""

    nodes: list[SONATANode]
    edges: list[SONATAEdge]
    node_populations: dict[str, list[int]]
    edge_populations: dict[str, list[int]]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def connectivity_matrix(self) -> np.ndarray:
        """Build dense connectivity matrix (n_nodes x n_nodes)."""
        N = self.n_nodes
        W = np.zeros((N, N))
        id_map = {n.node_id: i for i, n in enumerate(self.nodes)}
        for e in self.edges:
            src = id_map.get(e.source_id)
            tgt = id_map.get(e.target_id)
            if src is not None and tgt is not None:
                W[tgt, src] = e.weight
        return W


def import_sonata_nodes(path: str | Path) -> list[SONATANode]:
    """Parse a SONATA nodes HDF5 file.

    Expected structure:
      /nodes/<population_name>/node_id
      /nodes/<population_name>/node_type_id
      /nodes/<population_name>/0/model_type  (optional)
    """
    import h5py

    nodes = []
    with h5py.File(path, "r") as f:
        if "nodes" not in f:
            return nodes

        for pop_name in f["nodes"]:
            pop = f["nodes"][pop_name]
            n = len(pop["node_id"]) if "node_id" in pop else 0

            node_ids = pop["node_id"][:] if "node_id" in pop else np.arange(n)
            type_ids = pop["node_type_id"][:] if "node_type_id" in pop else np.zeros(n, dtype=int)

            # Optional per-node properties from group "0"
            props_group = pop.get("0")

            for i in range(n):
                props = {}
                if props_group:
                    for key in props_group:
                        ds = props_group[key]
                        if hasattr(ds, "shape") and len(ds.shape) > 0:
                            props[key] = ds[i] if i < len(ds) else None

                model_type = "point_neuron"
                if "model_type" in props:
                    model_type = str(props.pop("model_type"))

                nodes.append(
                    SONATANode(
                        node_id=int(node_ids[i]),
                        node_type_id=int(type_ids[i]),
                        model_type=model_type,
                        properties=props,
                    )
                )

    return nodes


def import_sonata_edges(path: str | Path) -> list[SONATAEdge]:
    """Parse a SONATA edges HDF5 file.

    Expected structure:
      /edges/<population_name>/source_node_id
      /edges/<population_name>/target_node_id
      /edges/<population_name>/edge_type_id
      /edges/<population_name>/0/syn_weight  (optional)
      /edges/<population_name>/0/delay       (optional)
    """
    import h5py

    edges = []
    with h5py.File(path, "r") as f:
        if "edges" not in f:
            return edges

        for pop_name in f["edges"]:
            pop = f["edges"][pop_name]
            src_ids = pop["source_node_id"][:] if "source_node_id" in pop else np.array([])
            tgt_ids = pop["target_node_id"][:] if "target_node_id" in pop else np.array([])
            type_ids = (
                pop["edge_type_id"][:]
                if "edge_type_id" in pop
                else np.zeros(len(src_ids), dtype=int)
            )

            props_group = pop.get("0")
            weights = None
            delays = None
            if props_group:
                if "syn_weight" in props_group:
                    weights = props_group["syn_weight"][:]
                if "delay" in props_group:
                    delays = props_group["delay"][:]

            for i in range(len(src_ids)):
                edges.append(
                    SONATAEdge(
                        source_id=int(src_ids[i]),
                        target_id=int(tgt_ids[i]),
                        edge_type_id=int(type_ids[i]),
                        weight=float(weights[i]) if weights is not None else 1.0,
                        delay=float(delays[i]) if delays is not None else 0.0,
                    )
                )

    return edges


def import_sonata(
    nodes_path: str | Path,
    edges_path: str | Path | None = None,
) -> SONATANetwork:
    """Import a complete SONATA network from nodes + edges files.

    Parameters
    ----------
    nodes_path : path to nodes.h5
    edges_path : path to edges.h5 (optional)

    Returns
    -------
    SONATANetwork with parsed nodes, edges, and connectivity.
    """
    nodes = import_sonata_nodes(nodes_path)

    edges = []
    if edges_path is not None:
        edges = import_sonata_edges(edges_path)

    # Group by population
    node_pops: dict[str, list[int]] = {}
    for n in nodes:
        pop = str(n.node_type_id)
        node_pops.setdefault(pop, []).append(n.node_id)

    edge_pops: dict[str, list[int]] = {}
    for i, e in enumerate(edges):
        pop = str(e.edge_type_id)
        edge_pops.setdefault(pop, []).append(i)

    return SONATANetwork(
        nodes=nodes,
        edges=edges,
        node_populations=node_pops,
        edge_populations=edge_pops,
    )
