# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
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

Files are read through libsonata, the reference reader (the ``sonata`` extra),
so population identity and property groups follow the specification: a node is
identified by its population and its id within that population, a node's
properties come from the group its ``node_group_id``/``node_group_index`` name,
and an edge names the populations of its source and target nodes. Only the
per-population ``node_type_id``/``edge_type_id`` datasets, which libsonata does
not expose, are read directly.

Nothing is invented. A file without the SONATA ``magic`` attribute, without
node populations, or without a population's type ids is refused, and so is a
population spread over several property groups, which libsonata does not read;
an attribute libsonata cannot read is an error, not a missing value. A node whose
file states no ``model_type`` keeps ``None`` rather than a guessed type, and an
edge without ``syn_weight`` or ``delay`` keeps ``None`` for it; building a
connectivity matrix refuses edges without a weight. Node-type and edge-type CSV
files are not read. Edges that name a node the network does not contain are
refused. SONATA is import-only: SC-NeuroCore has no SONATA exporter.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SONATA_MAGIC = 0x0A7A
"""The ``magic`` root attribute every SONATA HDF5 file carries."""

_NODE_RESERVED = frozenset({"model_type", "model_template"})
_EDGE_RESERVED = frozenset({"syn_weight", "delay"})


@dataclass
class SONATANode:
    """A single node (neuron) from a SONATA population.

    ``node_id`` is the node's id within ``population``; ``model_type`` and
    ``model_template`` are ``None`` when the file does not state them.
    """

    node_id: int
    node_type_id: int
    model_type: str | None = None
    model_template: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    population: str = ""


@dataclass
class SONATAEdge:
    """A single edge (synapse) from a SONATA population.

    ``source_id`` and ``target_id`` are ids within ``source_population`` and
    ``target_population``; ``weight`` and ``delay`` are ``None`` when the file
    does not state them.
    """

    source_id: int
    target_id: int
    edge_type_id: int
    weight: float | None = None
    delay: float | None = None
    properties: dict[str, Any] = field(default_factory=dict)
    source_population: str = ""
    target_population: str = ""
    population: str = ""


@dataclass
class SONATANetwork:
    """Parsed SONATA network with nodes and edges.

    ``node_populations`` maps each node population name to its node ids, and
    ``edge_populations`` each edge population name to indices into ``edges``.
    ``metadata`` records the file's SONATA version and how many edges state no
    weight or no delay.
    """

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

    def connectivity_matrix(self) -> np.ndarray[Any, Any]:
        """Build the dense connectivity matrix (n_nodes x n_nodes), rows = targets.

        Nodes are ordered as in ``nodes``; parallel edges between the same pair
        add their weights.

        Raises
        ------
        ValueError
            An edge states no weight, or names a node not in ``nodes``.
        """
        index = {(node.population, node.node_id): i for i, node in enumerate(self.nodes)}
        unweighted = sum(1 for edge in self.edges if edge.weight is None)
        if unweighted:
            raise ValueError(
                f"{unweighted} SONATA edge(s) state no syn_weight; a connectivity "
                "matrix needs every weight"
            )
        matrix = np.zeros((self.n_nodes, self.n_nodes))
        for edge in self.edges:
            source = index.get((edge.source_population, edge.source_id))
            target = index.get((edge.target_population, edge.target_id))
            if source is None or target is None:
                raise ValueError(f"SONATA edge {edge!r} names a node outside the network")
            matrix[target, source] += float(edge.weight or 0.0)
        return matrix


def _require_sonata_file(path: str | Path, group: str) -> Any:
    """Check the SONATA magic and return the file's version.

    Raises
    ------
    ValueError
        The file lacks the SONATA magic or the ``group`` root group.
    """
    import h5py

    with h5py.File(path, "r") as handle:
        magic = handle.attrs.get("magic")
        if magic is None or int(np.asarray(magic).ravel()[0]) != SONATA_MAGIC:
            raise ValueError(
                f"{path} is not a SONATA file: its magic attribute is missing or wrong"
            )
        if group not in handle:
            raise ValueError(f"SONATA file {path} has no /{group} group")
        version = handle.attrs.get("version")
    return None if version is None else [int(part) for part in np.asarray(version).ravel()]


def _population_type_ids(path: str | Path, group: str, population: str) -> list[int]:
    """Check what libsonata needs of a population and return its type ids.

    Raises
    ------
    ValueError
        The population lacks its type-id dataset, spans several property
        groups, or its type ids do not cover its members.
    """
    import h5py

    kind = group[:-1]  # "node" or "edge"
    dataset = f"{kind}_type_id"
    with h5py.File(path, "r") as handle:
        members = handle[group][population]
        if dataset not in members:
            raise ValueError(f"SONATA {group} population {population!r} has no {dataset} dataset")
        values = members[dataset][:]
        group_ids = members[f"{kind}_group_id"][:] if f"{kind}_group_id" in members else []
        size = len(members["source_node_id"]) if kind == "edge" else len(values)
    if len(set(int(value) for value in group_ids)) > 1:
        raise ValueError(
            f"SONATA {group} population {population!r} spans several property groups; "
            "libsonata reads single-group populations only"
        )
    if len(values) != size:
        raise ValueError(
            f"SONATA {group} population {population!r} {dataset} has {len(values)} "
            f"entries for {size} members"
        )
    return [int(value) for value in values]


def _attribute_columns(population: Any, names: Iterable[str]) -> dict[str, list[Any]]:
    """Read every attribute for every member of a single-group population.

    Raises
    ------
    ValueError
        libsonata cannot read an attribute, for example one stored as
        fixed-length bytes instead of a string.
    """
    import libsonata

    selection = population.select_all()
    columns: dict[str, list[Any]] = {}
    for name in sorted(names):
        try:
            columns[name] = list(population.get_attribute(name, selection))
        except libsonata.SonataError as exc:
            raise ValueError(
                f"SONATA attribute {name!r} of population {population.name!r} cannot be read: {exc}"
            ) from exc
    return columns


def _plain(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def import_sonata_nodes(path: str | Path) -> list[SONATANode]:
    """Parse a SONATA nodes HDF5 file through libsonata.

    Returns
    -------
    list of SONATANode
        Every node of every population, populations in name order.

    Raises
    ------
    ValueError
        The file is not a SONATA file, has no node population, or a population
        lacks ``node_type_id``.
    """
    import libsonata

    _require_sonata_file(path, "nodes")
    storage = libsonata.NodeStorage(str(path))
    names = sorted(storage.population_names)
    if not names:
        raise ValueError(f"SONATA nodes file {path} declares no node populations")
    nodes: list[SONATANode] = []
    for name in names:
        type_ids = _population_type_ids(path, "nodes", name)
        population = storage.open_population(name)
        size = int(population.size)
        columns = _attribute_columns(population, population.attribute_names)
        for member in range(size):
            values = {key: _plain(column[member]) for key, column in columns.items()}
            properties = {
                key: value
                for key, value in values.items()
                if key not in _NODE_RESERVED and value is not None
            }
            model_type = values.get("model_type")
            model_template = values.get("model_template")
            nodes.append(
                SONATANode(
                    node_id=member,
                    node_type_id=type_ids[member],
                    model_type=None if model_type is None else str(model_type),
                    model_template=None if model_template is None else str(model_template),
                    properties=properties,
                    population=name,
                )
            )
    return nodes


def import_sonata_edges(path: str | Path) -> list[SONATAEdge]:
    """Parse a SONATA edges HDF5 file through libsonata.

    Returns
    -------
    list of SONATAEdge
        Every edge of every population, populations in name order.

    Raises
    ------
    ValueError
        The file is not a SONATA file or a population lacks ``edge_type_id``.
    """
    import libsonata

    _require_sonata_file(path, "edges")
    storage = libsonata.EdgeStorage(str(path))
    edges: list[SONATAEdge] = []
    for name in sorted(storage.population_names):
        type_ids = _population_type_ids(path, "edges", name)
        population = storage.open_population(name)
        size = int(population.size)
        selection = population.select_all()
        sources = population.source_nodes(selection)
        targets = population.target_nodes(selection)
        columns = _attribute_columns(population, population.attribute_names)
        for member in range(size):
            values = {key: _plain(column[member]) for key, column in columns.items()}
            weight = values.get("syn_weight")
            delay = values.get("delay")
            edges.append(
                SONATAEdge(
                    source_id=int(sources[member]),
                    target_id=int(targets[member]),
                    edge_type_id=type_ids[member],
                    weight=None if weight is None else float(weight),
                    delay=None if delay is None else float(delay),
                    properties={
                        key: value
                        for key, value in values.items()
                        if key not in _EDGE_RESERVED and value is not None
                    },
                    source_population=str(population.source),
                    target_population=str(population.target),
                    population=name,
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
    SONATANetwork
        Parsed nodes, edges, populations and version metadata.

    Raises
    ------
    ValueError
        A file is not SONATA or is incomplete, or an edge names a node the
        nodes file does not contain.
    """
    version = _require_sonata_file(nodes_path, "nodes")
    nodes = import_sonata_nodes(nodes_path)
    edges = [] if edges_path is None else import_sonata_edges(edges_path)

    node_populations: dict[str, list[int]] = {}
    for node in nodes:
        node_populations.setdefault(node.population, []).append(node.node_id)
    known = {(node.population, node.node_id) for node in nodes}
    edge_populations: dict[str, list[int]] = {}
    for index, edge in enumerate(edges):
        for population, member in (
            (edge.source_population, edge.source_id),
            (edge.target_population, edge.target_id),
        ):
            if (population, member) not in known:
                raise ValueError(
                    f"SONATA edge population {edge.population!r} names node {member} of "
                    f"population {population!r}, which the nodes file does not contain"
                )
        edge_populations.setdefault(edge.population, []).append(index)

    return SONATANetwork(
        nodes=nodes,
        edges=edges,
        node_populations=node_populations,
        edge_populations=edge_populations,
        metadata={
            "sonata_version": version,
            "edges_without_weight": sum(1 for edge in edges if edge.weight is None),
            "edges_without_delay": sum(1 for edge in edges if edge.delay is None),
        },
    )
