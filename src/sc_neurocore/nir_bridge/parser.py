# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NIR graph parser

"""Execute NIR graphs with explicit clocks, reset conventions and metadata custody."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

try:
    import nir
except ImportError as e:
    raise ImportError("pip install nir") from e

from .interchange_notes import InterchangeNote, import_notes, read_nir_file_version
from .execution_metadata import read_execution_settings
from .graph_identity import graph_order
from .node_map import map_node


@dataclass
class _UnitDelayNode:
    """Implicit unit-delay inserted on recurrent (back) edges.

    Acts as a DAG source: outputs the previous timestep's buffered value.
    Buffer is updated externally by SCNetwork.step() after execution.
    """

    name: str
    _buffer: np.ndarray[Any, Any] | None = None

    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if self._buffer is None:
            x = np.atleast_1d(np.asarray(x, dtype=np.float64))
            self._buffer = np.zeros_like(x)
        return self._buffer.copy()

    def update_buffer(self, value: np.ndarray[Any, Any]) -> None:
        self._buffer = np.atleast_1d(np.asarray(value, dtype=np.float64)).copy()

    def reset(self) -> None:
        self._buffer = None


@dataclass
class SCSubgraphNode:
    """Executable wrapper for a nested NIR subgraph (single I/O port)."""

    name: str
    network: SCNetwork

    def __post_init__(self) -> None:
        """Require one named input and output at a single-port nested boundary."""
        if len(self.network.input_nodes) != 1 or len(self.network.output_nodes) != 1:
            raise ValueError("Nested NIRGraph nodes must expose exactly one input and one output")

    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Advance the nested graph once using its sole input and output ports."""
        outputs = self.network.step({self.network.input_nodes[0]: np.atleast_1d(np.asarray(x))})
        return outputs[self.network.output_nodes[0]]

    def reset(self) -> None:
        """Restore every stateful node inside the nested graph."""
        self.network.reset()


@dataclass
class SCMultiPortSubgraphNode:
    """Executable wrapper for a nested NIR subgraph with multiple I/O ports.

    Supports modular architectures where subgraphs expose multiple named
    inputs and outputs (e.g., encoder-decoder, skip connections).
    """

    name: str
    network: SCNetwork

    def __post_init__(self) -> None:
        """Require a nested graph with at least one input and output boundary."""
        if not self.network.input_nodes or not self.network.output_nodes:
            raise ValueError("Multi-port subgraph must have at least one input and one output")

    @property
    def input_ports(self) -> list[str]:
        """Return the nested graph's input names in their declared order."""
        return self.network.input_nodes

    @property
    def output_ports(self) -> list[str]:
        """Return the nested graph's output names in their declared order."""
        return self.network.output_nodes

    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Single-input convenience: feeds x to first input, returns first output."""
        inputs = {self.network.input_nodes[0]: np.atleast_1d(np.asarray(x))}
        outputs = self.network.step(inputs)
        return outputs[self.network.output_nodes[0]]

    def forward_multi(
        self, inputs: dict[str, np.ndarray[Any, Any]]
    ) -> dict[str, np.ndarray[Any, Any]]:
        """Multi-port forward: provide named inputs, get named outputs."""
        return self.network.step(inputs)

    def reset(self) -> None:
        """Restore every stateful node across all nested ports."""
        self.network.reset()


@dataclass
class SCNetwork:
    """Executable network parsed from a NIR graph.

    Nodes are stored by name. Edges define the forward pass order.
    Calling ``run()`` feeds input through the graph for the given
    number of timesteps and returns the output node's accumulated result.

    Recurrent edges (cycles) are automatically handled by inserting
    unit-delay nodes that feed from the previous timestep.

    Original graph and primitive metadata are kept independently of the input
    object. NIR export records the timestep and reset convention in the
    versioned execution extension; it does not export an in-progress state
    checkpoint.
    """

    nodes: dict[str, Any] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)
    input_nodes: list[str] = field(default_factory=list)
    output_nodes: list[str] = field(default_factory=list)
    _topo_order: list[str] | None = None
    # Maps delay_node_name → source_node_name for recurrent connections
    _recurrent_map: dict[str, str] = field(default_factory=dict)
    # How the graph was imported, and what that import assumed or approximated.
    dt: float = 1.0
    reset_mode: str = "reset"
    nir_version: str | None = None
    interchange_notes: list[InterchangeNote] = field(default_factory=list)
    nir_metadata: dict[str, Any] = field(default_factory=dict)
    nir_node_metadata: dict[str, dict[str, Any]] = field(default_factory=dict)
    nir_node_definitions: dict[str, Any] = field(default_factory=dict)
    _nir_source_edges: list[tuple[str, str]] = field(default_factory=list)

    @classmethod
    def from_nir(
        cls, source: Any, dt: float | None = None, reset_mode: str | None = None
    ) -> SCNetwork:
        """Build a network using the same metadata and admission as the importer.

        Parameters
        ----------
        source : object
            Reference NIR graph or path to its HDF5 file.
        dt : float or None
            Explicit timestep, or the recorded clock and foreign-graph default.
        reset_mode : str or None
            Explicit reset convention, or the recorded convention and default.

        Returns
        -------
        SCNetwork
            Executable graph with independent metadata custody.

        Raises
        ------
        ValueError
            The graph, execution metadata or explicit import options are invalid.
        """
        network = from_nir(source, dt=dt, reset_mode=reset_mode)
        if not isinstance(network, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(network).__name__}")
        return network

    def to_hardware(
        self,
        *,
        module_name: str = "sc_nir_network",
        data_width: int = 16,
        fraction: int = 8,
        bitstream_length: int = 256,
        source_kind: Literal["lfsr", "sobol"] = "lfsr",
        base_seed: int = 1,
        target: str = "artix7",
        dt: float | None = None,
        online_learning: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> Any:
        """Compile this parsed network to the existing FPGA artefact bundle.

        Returns
        -------
        NetworkCompilationResult
            The normal hardware compiler result for the declared target and
            stream, precision, seed and learning configuration.

        Raises
        ------
        ValueError
            The graph or requested hardware configuration cannot be lowered.
        """
        from .fpga_compiler import compile_network_to_fpga
        from .neuron_graph import from_scnetwork

        neuron_graph = from_scnetwork(self, dt=dt)
        return compile_network_to_fpga(
            neuron_graph,
            module_name=module_name,
            data_width=data_width,
            fraction=fraction,
            bitstream_length=bitstream_length,
            source_kind=source_kind,
            base_seed=base_seed,
            target=target,
            online_learning=online_learning,
        )

    def _find_back_edges(self) -> list[tuple[str, str]]:
        """DFS-based back-edge detection."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {n: WHITE for n in self.nodes}
        adj: dict[str, list[str]] = {n: [] for n in self.nodes}
        for src, dst in self.edges:
            adj[src].append(dst)

        back_edges: list[tuple[str, str]] = []

        def dfs(u: str) -> None:
            color[u] = GRAY
            for v in adj[u]:
                if v not in color:
                    continue
                if color[v] == GRAY:
                    back_edges.append((u, v))
                elif color[v] == WHITE:
                    dfs(v)
            color[u] = BLACK

        for n in self.nodes:
            if color[n] == WHITE:
                dfs(n)
        return back_edges

    def _break_cycles(self) -> None:
        """Replace back edges with unit-delay source nodes."""
        back_edges = self._find_back_edges()
        if not back_edges:
            return

        for src, dst in back_edges:
            delay_name = f"_delay_{src}_to_{dst}"
            self.edges.remove((src, dst))
            # Delay node is a DAG source (no incoming edges) — feeds dst
            self.nodes[delay_name] = _UnitDelayNode(name=delay_name)
            self.edges.append((delay_name, dst))
            self._recurrent_map[delay_name] = src

    def _topological_sort(self) -> list[str]:
        """Kahn's algorithm with automatic cycle breaking via delay nodes."""
        self._break_cycles()

        adj: dict[str, list[str]] = {n: [] for n in self.nodes}
        in_deg: dict[str, int] = {n: 0 for n in self.nodes}
        for src, dst in self.edges:
            adj[src].append(dst)
            in_deg[dst] = in_deg.get(dst, 0) + 1

        queue = [n for n, d in in_deg.items() if d == 0]
        order = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for nxt in adj[node]:
                in_deg[nxt] -= 1
                if in_deg[nxt] == 0:
                    queue.append(nxt)

        if len(order) != len(self.nodes):
            raise ValueError("NIR graph contains a cycle that cannot be broken by delay insertion")
        return order

    @property
    def topo_order(self) -> list[str]:
        """Return the execution order, inserting declared recurrent delays once."""
        if self._topo_order is None:
            self._topo_order = self._topological_sort()
        return self._topo_order

    def step(self, inputs: dict[str, np.ndarray[Any, Any]]) -> dict[str, np.ndarray[Any, Any]]:
        """Execute one timestep through the graph.

        Parameters
        ----------
        inputs : dict mapping input node name → input array

        Returns
        -------
        dict mapping output node name → output array
        """
        values: dict[str, np.ndarray[Any, Any]] = {}

        for name in self.topo_order:
            node = self.nodes[name]

            if name in self.input_nodes:
                x = inputs.get(name, np.array([0.0]))
                values[name] = node.forward(x)
            elif isinstance(node, _UnitDelayNode):
                # Delay nodes are sources — forward() returns buffered value
                values[name] = node.forward(np.array([0.0]))
            else:
                predecessors = [src for src, dst in self.edges if dst == name]
                if len(predecessors) == 1:
                    x = values[predecessors[0]]
                elif len(predecessors) > 1:
                    x = sum(values[p] for p in predecessors)  # type: ignore[assignment]
                else:
                    x = np.array([0.0])
                values[name] = node.forward(x)

            # Primitive executors use flat state vectors or squeeze convolution
            # results internally. Restore the NIR edge's declared tensor rank
            # before the next graph node consumes it. A scalar broadcast import
            # without a matching declared width retains its inferred width.
            definition = self.nir_node_definitions.get(name)
            declared = getattr(definition, "output_type", None)
            if declared and len(declared) == 1:
                shape = next(iter(declared.values()))
                if shape is not None:
                    dimensions = tuple(int(value) for value in np.asarray(shape).reshape(-1))
                    if np.prod(dimensions, dtype=np.int64) == np.size(values[name]):
                        values[name] = np.asarray(values[name]).reshape(dimensions)

        # Update delay buffers with this timestep's source values
        for delay_name, src_name in self._recurrent_map.items():
            if src_name in values:
                self.nodes[delay_name].update_buffer(values[src_name])

        return {name: values[name] for name in self.output_nodes if name in values}

    def run(
        self, inputs: dict[str, np.ndarray[Any, Any]], steps: int = 100
    ) -> dict[str, list[np.ndarray[Any, Any]]]:
        """Run the network for multiple timesteps.

        Parameters
        ----------
        inputs : dict mapping input node name → input array (constant across steps)
        steps : number of timesteps

        Returns
        -------
        dict mapping output node name → list of output arrays per timestep
        """
        results: dict[str, list[np.ndarray[Any, Any]]] = {n: [] for n in self.output_nodes}
        for _ in range(steps):
            out = self.step(inputs)
            for name, val in out.items():
                results[name].append(val.copy())
        return results

    def reset(self) -> None:
        """Reset all stateful nodes."""
        for node in self.nodes.values():
            if hasattr(node, "reset"):
                node.reset()

    def summary(self) -> str:
        """Human-readable network summary."""
        lines = [f"SCNetwork: {len(self.nodes)} nodes, {len(self.edges)} edges"]
        for name in self.topo_order:
            node = self.nodes[name]
            lines.append(f"  {name}: {type(node).__name__}")
        if self._recurrent_map:
            lines.append(f"  recurrent: {list(self._recurrent_map.values())}")
        lines.append(f"  inputs: {self.input_nodes}")
        lines.append(f"  outputs: {self.output_nodes}")
        return "\n".join(lines)


def from_nir(source: object, dt: float | None = None, reset_mode: str | None = None) -> SCNetwork:
    """Convert a NIR graph to an executable SC-NeuroCore network.

    Parameters
    ----------
    source : nir.NIRGraph or str or Path
        NIR graph object, or path to a .nir file.
    dt : float or None
        Timestep for leaky integrator dynamics. Recorded execution metadata is
        used when omitted; foreign graphs without it default to one time unit.
    reset_mode : str or None
        Spike reset mechanism: "reset" (v = v_reset, bridge default)
        or "subtract" (ordinary SC subtraction). "snntorch-subtract" also accounts
        for the previous membrane reset signal before detection. When omitted,
        recorded settings apply, or foreign graphs default to "reset".

    Returns
    -------
    SCNetwork
        Executable network with topologically sorted forward pass. Its
        ``interchange_notes`` list what the import assumed or approximated,
        and ``nir_version`` is the version a ``.nir`` file records.

    Raises
    ------
    ValueError
        ``dt`` is not a positive finite number, ``reset_mode`` is unsupported,
        or the graph is malformed.
    """
    version: str | None = None
    if isinstance(source, (str, Path)):
        graph = _read_nir_file(source)
        version = read_nir_file_version(source)
    elif isinstance(source, nir.NIRGraph):
        graph = source
    else:
        raise TypeError(f"Expected NIRGraph or path, got {type(source)}")

    _validate_nir_graph_boundary(graph)
    settings = read_execution_settings(graph.metadata, dt=dt, reset_mode=reset_mode)
    network = _parse_graph(graph, dt=settings.dt, reset_mode=settings.reset_mode)
    network.nir_version = version
    network.interchange_notes = import_notes(
        graph, network, dt=settings.dt, reset_mode=settings.reset_mode
    )
    return network


def _read_nir_file(source: str | Path) -> Any:
    try:
        return nir.read(str(source))
    except Exception as exc:
        raise ValueError(f"Failed to read NIR file {source!s}") from exc


def _validate_nir_graph_boundary(graph: Any, context: str = "NIR graph") -> None:
    if not isinstance(graph, nir.NIRGraph):
        raise TypeError(f"Expected NIRGraph, got {type(graph)}")

    if not isinstance(graph.nodes, Mapping):
        raise ValueError(f"{context} nodes must be a mapping")
    node_names = set()
    for name, node in graph.nodes.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"{context} node names must be non-empty strings")
        node_names.add(name)
        if isinstance(node, nir.NIRGraph):
            _validate_nir_graph_boundary(node, context=f"{context}.{name}")

    if not isinstance(graph.edges, Sequence) or isinstance(graph.edges, (str, bytes)):
        raise ValueError(f"{context} edges must be a sequence")
    for index, edge in enumerate(graph.edges):
        if not isinstance(edge, Sequence) or isinstance(edge, (str, bytes)) or len(edge) != 2:
            raise ValueError(f"{context} edge {index} must contain source and destination")
        src, dst = edge
        if not isinstance(src, str) or not src:
            raise ValueError(f"{context} edge {index} source must be a non-empty string")
        if not isinstance(dst, str) or not dst:
            raise ValueError(f"{context} edge {index} destination must be a non-empty string")
        if src not in node_names:
            raise ValueError(f"{context} edge {index} source {src!r} not found")
        if dst not in node_names:
            raise ValueError(f"{context} edge {index} destination {dst!r} not found")


def _parse_graph(
    graph: nir.NIRGraph,
    dt: float = 1.0,
    reset_mode: str = "reset",
) -> SCNetwork:
    """Recursively parse a NIR graph into an SCNetwork."""
    read_execution_settings(graph.metadata, dt=dt, reset_mode=reset_mode)
    ordered_names, edges = graph_order(graph)
    nodes = {}
    input_nodes = []
    output_nodes = []

    for name in ordered_names:
        node = graph.nodes[name]
        if isinstance(node, nir.NIRGraph):
            sub_net = _parse_graph(node, dt=dt, reset_mode=reset_mode)
            if len(sub_net.input_nodes) == 1 and len(sub_net.output_nodes) == 1:
                nodes[name] = SCSubgraphNode(name=name, network=sub_net)
            else:
                nodes[name] = SCMultiPortSubgraphNode(name=name, network=sub_net)  # type: ignore[assignment]
        else:
            # Factories own their numerical arrays and mutable geometry. Retain
            # one independent original definition below, without copying twice.
            sc_node = map_node(name, node, dt=dt, reset_mode=reset_mode)
            nodes[name] = sc_node
            if isinstance(node, nir.Input):
                input_nodes.append(name)
            elif isinstance(node, nir.Output):
                output_nodes.append(name)

    return SCNetwork(
        nodes=nodes,
        edges=edges,
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        dt=float(dt),
        reset_mode=reset_mode,
        _nir_source_edges=edges.copy(),
        nir_node_definitions={
            name: deepcopy(node)
            for name, node in graph.nodes.items()
            if not isinstance(node, nir.NIRGraph)
        },
        nir_metadata=deepcopy(graph.metadata),
        nir_node_metadata={
            name: deepcopy(getattr(node, "metadata", {}))
            for name, node in graph.nodes.items()
            if not isinstance(node, nir.NIRGraph)
        },
    )
