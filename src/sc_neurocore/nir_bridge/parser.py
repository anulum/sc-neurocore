# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NIR graph parser

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

try:
    import nir
except ImportError as e:
    raise ImportError("pip install nir") from e

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
        if len(self.network.input_nodes) != 1 or len(self.network.output_nodes) != 1:
            raise ValueError("Nested NIRGraph nodes must expose exactly one input and one output")

    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        outputs = self.network.step({self.network.input_nodes[0]: np.atleast_1d(np.asarray(x))})
        return outputs[self.network.output_nodes[0]]

    def reset(self) -> None:
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
        if not self.network.input_nodes or not self.network.output_nodes:
            raise ValueError("Multi-port subgraph must have at least one input and one output")

    @property
    def input_ports(self) -> list[str]:
        return self.network.input_nodes

    @property
    def output_ports(self) -> list[str]:
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
        self.network.reset()


@dataclass
class SCNetwork:
    """Executable network parsed from a NIR graph.

    Nodes are stored by name. Edges define the forward pass order.
    Calling ``run()`` feeds input through the graph for the given
    number of timesteps and returns the output node's accumulated result.

    Recurrent edges (cycles) are automatically handled by inserting
    unit-delay nodes that feed from the previous timestep.
    """

    nodes: dict[str, Any] = field(default_factory=dict)
    edges: list[tuple[str, str]] = field(default_factory=list)
    input_nodes: list[str] = field(default_factory=list)
    output_nodes: list[str] = field(default_factory=list)
    _topo_order: list[str] | None = None
    # Maps delay_node_name → source_node_name for recurrent connections
    _recurrent_map: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_nir(cls, source: Any, dt: float = 1.0, reset_mode: str = "reset") -> SCNetwork:
        """Build an ``SCNetwork`` directly from a NIR graph or file path."""

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
        """Compile this parsed network to the existing FPGA artefact bundle."""

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


def from_nir(source, dt: float = 1.0, reset_mode: str = "reset") -> SCNetwork:  # type: ignore[no-untyped-def]
    """Convert a NIR graph to an executable SC-NeuroCore network.

    Parameters
    ----------
    source : nir.NIRGraph or str or Path
        NIR graph object, or path to a .nir file.
    dt : float
        Timestep for leaky integrator dynamics.
    reset_mode : str
        Spike reset mechanism: "reset" (v = v_reset, NIR spec default)
        or "subtract" (v = v - v_threshold, used by snnTorch).

    Returns
    -------
    SCNetwork
        Executable network with topologically sorted forward pass.
    """
    _validate_import_options(dt, reset_mode)

    if isinstance(source, (str, Path)):
        graph = _read_nir_file(source)
    elif isinstance(source, nir.NIRGraph):
        graph = source
    else:
        raise TypeError(f"Expected NIRGraph or path, got {type(source)}")

    _validate_nir_graph_boundary(graph)
    return _parse_graph(graph, dt=dt, reset_mode=reset_mode)


def _validate_import_options(dt: float, reset_mode: str) -> None:
    if not isinstance(dt, (int, float)) or isinstance(dt, bool) or not math.isfinite(float(dt)):
        raise ValueError("NIR import dt must be a finite number")
    if float(dt) <= 0:
        raise ValueError("NIR import dt must be positive")


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
    nodes = {}
    input_nodes = []
    output_nodes = []

    for name, node in graph.nodes.items():
        if isinstance(node, nir.NIRGraph):
            sub_net = _parse_graph(node, dt=dt, reset_mode=reset_mode)
            if len(sub_net.input_nodes) == 1 and len(sub_net.output_nodes) == 1:
                nodes[name] = SCSubgraphNode(name=name, network=sub_net)
            else:
                nodes[name] = SCMultiPortSubgraphNode(name=name, network=sub_net)  # type: ignore[assignment]
        else:
            sc_node = map_node(name, node, dt=dt, reset_mode=reset_mode)
            nodes[name] = sc_node
            if isinstance(node, nir.Input):
                input_nodes.append(name)
            elif isinstance(node, nir.Output):
                output_nodes.append(name)

    edges = [(src, dst) for src, dst in graph.edges]

    return SCNetwork(
        nodes=nodes,
        edges=edges,
        input_nodes=input_nodes,
        output_nodes=output_nodes,
    )
