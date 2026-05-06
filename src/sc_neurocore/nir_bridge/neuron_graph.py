# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NeuronGraph intermediate representation for FPGA compilation

"""NeuronGraph: hardware-targeted IR for FPGA synthesis.

The NeuronGraph sits between the NIR/ONNX import layer and the Verilog
emitter.  It describes a spiking neural network as an ordered sequence of
*neuron populations* connected by *weighted edges*, ready for direct
translation into synthesisable RTL.

Architecture
~~~~~~~~~~~~

The graph is constructed by iterating through the topologically-sorted
nodes of a parsed ``SCNetwork`` (from ``nir_bridge/parser.py``).  Neuron
nodes (LIF, IF, CubaLIF, LI, CubaLI) become ``NeuronSpec`` entries.
Weight-carrying nodes (Affine, Linear) become ``ConnectionSpec`` entries
that bind the source population to the destination population.  Non-compute
nodes (Input, Output, Scale, Flatten, Threshold, Delay) are folded into
the graph metadata or the adjacent connection.

Canonical ODE Templates
~~~~~~~~~~~~~~~~~~~~~~~~

Each ``NeuronSpec.neuron_type`` maps to a canonical ODE string understood
by ``equation_compiler.compile_to_verilog()``:

- ``"lif"``      → ``dv/dt = -(v - v_leak) / tau + I * r / tau``
- ``"if"``       → ``dv/dt = I * r``
- ``"li"``       → ``dv/dt = -(v - v_leak) / tau + I * r / tau``
- ``"cuba_lif"`` → ``di/dt = -i / tau_syn + I * w_in; dv/dt = -(v - v_leak) / tau_mem + i * r / tau_mem``
- ``"cuba_li"``  → ``di/dt = -i / tau_syn + I * w_in; dv/dt = -(v - v_leak) / tau_mem + i * r / tau_mem``

Usage
~~~~~

::

    import nir
    from sc_neurocore.nir_bridge import from_nir
    from sc_neurocore.nir_bridge.neuron_graph import from_scnetwork

    graph = nir.read("model.nir")
    network = from_nir(graph, dt=1e-3)
    neuron_graph = from_scnetwork(network)
    # → NeuronGraph with populations, connections, ready for compile
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class NeuronSpec:
    """One neuron population (layer) in the compiled graph.

    Attributes
    ----------
    name : str
        Unique population identifier (matches the NIR node name).
    neuron_type : str
        Canonical neuron type: ``"lif"``, ``"if"``, ``"li"``,
        ``"cuba_lif"``, ``"cuba_li"``.
    n_neurons : int
        Number of neurons in this population.
    params : dict[str, np.ndarray]
        Neuron parameters keyed by canonical names:
        ``tau``, ``r``, ``v_leak``, ``v_threshold``, ``v_reset``,
        ``tau_syn``, ``tau_mem``, ``w_in`` (type-dependent).
    dt : float
        Simulation timestep used during NIR import.
    """

    name: str
    neuron_type: str
    n_neurons: int
    params: dict[str, np.ndarray] = field(default_factory=dict)
    dt: float = 1.0


@dataclass
class ConnectionSpec:
    """Weighted edge between two neuron populations.

    Attributes
    ----------
    src : str
        Source population name.
    dst : str
        Destination population name.
    weights : np.ndarray
        Weight matrix of shape ``(n_dst, n_src)`` in float32.
        Row *i* contains the weights from all source neurons to
        destination neuron *i*.
    bias : np.ndarray | None
        Optional bias vector of shape ``(n_dst,)``.
    """

    src: str
    dst: str
    weights: np.ndarray
    bias: np.ndarray | None = None


@dataclass
class NeuronGraph:
    """Complete network description ready for FPGA compilation.

    Attributes
    ----------
    populations : list[NeuronSpec]
        Ordered list of neuron populations (topological order).
    connections : list[ConnectionSpec]
        Weighted connections between populations.
    input_pop : str
        Name of the input population.
    output_pop : str
        Name of the output population.
    dt : float
        Global simulation timestep.
    """

    populations: list[NeuronSpec]
    connections: list[ConnectionSpec]
    input_pop: str
    output_pop: str
    dt: float = 1.0

    @property
    def total_neurons(self) -> int:
        """Total neuron count across all populations."""
        return sum(pop.n_neurons for pop in self.populations)

    @property
    def total_synapses(self) -> int:
        """Total synapse count across all connections."""
        return sum(conn.weights.size for conn in self.connections)

    @property
    def neuron_types(self) -> set[str]:
        """Set of unique neuron types in the graph."""
        return {pop.neuron_type for pop in self.populations}

    def summary(self) -> str:
        """Human-readable summary of the network graph."""
        lines = [
            f"NeuronGraph: {len(self.populations)} populations, "
            f"{len(self.connections)} connections",
            f"  Total neurons:  {self.total_neurons}",
            f"  Total synapses: {self.total_synapses}",
            f"  Neuron types:   {', '.join(sorted(self.neuron_types))}",
            f"  Input:  {self.input_pop}",
            f"  Output: {self.output_pop}",
            f"  dt: {self.dt}",
            "",
            "  Populations:",
        ]
        for pop in self.populations:
            lines.append(
                f"    {pop.name}: {pop.neuron_type} × {pop.n_neurons}"
            )
        lines.append("")
        lines.append("  Connections:")
        for conn in self.connections:
            shape = f"{conn.weights.shape[1]}→{conn.weights.shape[0]}"
            bias_str = " +bias" if conn.bias is not None else ""
            lines.append(f"    {conn.src} → {conn.dst}: {shape}{bias_str}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Neuron Type Classification
# ═══════════════════════════════════════════════════════════════════════

# Maps SC node class names to canonical neuron types
_SC_NODE_TO_TYPE: dict[str, str] = {
    "SCLIFNode": "lif",
    "SCIFNode": "if",
    "SCLINode": "li",
    "SCCubaLIFNode": "cuba_lif",
    "SCCubaLINode": "cuba_li",
    "SCIntegratorNode": "integrator",
}

# Maps SC node class names to weight-carrying connection types
_SC_WEIGHT_NODES: set[str] = {
    "SCAffineNode",
    "SCLinearNode",
}

# Maps SC node class names to pass-through nodes (folded into graph)
_SC_PASSTHROUGH_NODES: set[str] = {
    "SCInputNode",
    "SCOutputNode",
    "SCScaleNode",
    "SCFlattenNode",
    "SCThresholdNode",
    "SCDelayNode",
    "_UnitDelayNode",
}


def _extract_neuron_params(node: Any, neuron_type: str) -> dict[str, np.ndarray]:
    """Extract canonical parameters from an SC neuron node.

    Parameters
    ----------
    node : Any
        SC node instance (e.g. ``SCLIFNode``, ``SCCubaLIFNode``).
    neuron_type : str
        Canonical neuron type string.

    Returns
    -------
    dict[str, np.ndarray]
        Parameter dictionary with type-appropriate keys.
    """
    params: dict[str, np.ndarray] = {}

    # Common parameters
    for attr in ("tau", "r", "v_leak", "v_threshold", "v_reset"):
        val = getattr(node, attr, None)
        if val is not None:
            params[attr] = np.atleast_1d(np.asarray(val, dtype=np.float64))

    # CubaLIF/CubaLI-specific
    if neuron_type in ("cuba_lif", "cuba_li"):
        for attr in ("tau_syn", "tau_mem", "w_in"):
            val = getattr(node, attr, None)
            if val is not None:
                params[attr] = np.atleast_1d(np.asarray(val, dtype=np.float64))

    # IF neuron: no tau
    if neuron_type == "if":
        params.pop("tau", None)
        params.pop("v_leak", None)

    # Integrator: just r
    if neuron_type == "integrator":
        for attr in ("tau", "v_leak", "v_threshold", "v_reset"):
            params.pop(attr, None)

    return params


# ═══════════════════════════════════════════════════════════════════════
# SCNetwork → NeuronGraph Conversion
# ═══════════════════════════════════════════════════════════════════════


def from_scnetwork(network: Any, dt: float | None = None) -> NeuronGraph:
    """Convert a parsed SCNetwork to a NeuronGraph for FPGA compilation.

    Walks the topologically-sorted node list and partitions nodes into
    neuron populations and weighted connections.  Pass-through nodes
    (Input, Output, Scale, Flatten, Threshold) are folded into the
    adjacent edges.

    Parameters
    ----------
    network : SCNetwork
        A parsed SC-NeuroCore network (from ``from_nir()``).
    dt : float, optional
        Override the simulation timestep.  If ``None``, uses the
        timestep stored in the network's neuron nodes.

    Returns
    -------
    NeuronGraph
        Network description ready for FPGA compilation.

    Raises
    ------
    ValueError
        If the network contains no neuron populations or no connections.
    """
    topo_order = network.topo_order
    nodes = network.nodes
    edges = list(network.edges)

    # Build adjacency: node_name → list of successor node names
    successors: dict[str, list[str]] = {}
    predecessors: dict[str, list[str]] = {}
    for src, dst in edges:
        successors.setdefault(src, []).append(dst)
        predecessors.setdefault(dst, []).append(src)

    populations: list[NeuronSpec] = []
    connections: list[ConnectionSpec] = []
    input_pop = ""
    output_pop = ""

    # Track which weight node feeds which neuron node
    # Pattern: Input → [Affine/Linear] → [Neuron] → [Affine/Linear] → [Neuron] → Output
    pending_weights: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
    # Maps a neuron node name → the weight node that feeds it
    weight_source_for: dict[str, str] = {}

    # First pass: classify nodes
    for name in topo_order:
        node = nodes[name]
        class_name = type(node).__name__

        if class_name == "SCInputNode":
            # Input node: find the first neuron population downstream
            if not input_pop:
                input_pop = name
            continue

        if class_name == "SCOutputNode":
            if not output_pop:
                output_pop = name
            continue

        if class_name in _SC_WEIGHT_NODES:
            # Weight-carrying node: store weights for the downstream neuron
            weight = getattr(node, "weight", None)
            bias = getattr(node, "bias", None)
            if weight is None:
                w = getattr(node, "weights", None)
                if w is not None:
                    weight = w
            if weight is not None:
                weight = np.asarray(weight, dtype=np.float32)
                if bias is not None:
                    bias = np.asarray(bias, dtype=np.float32)
                pending_weights[name] = (weight, bias)

                # Find the neuron this feeds into
                for succ in successors.get(name, []):
                    succ_class = type(nodes[succ]).__name__
                    if succ_class in _SC_NODE_TO_TYPE:
                        weight_source_for[succ] = name
            continue

        if class_name in _SC_PASSTHROUGH_NODES:
            continue

        # Neuron node
        neuron_type = _SC_NODE_TO_TYPE.get(class_name)
        if neuron_type is None:
            logger.warning(
                "Skipping unsupported node type %s (%s) in FPGA compilation",
                class_name, name,
            )
            continue

        n_neurons = getattr(node, "n_neurons", 1)
        node_dt = dt if dt is not None else getattr(node, "dt", 1.0)
        params = _extract_neuron_params(node, neuron_type)

        populations.append(
            NeuronSpec(
                name=name,
                neuron_type=neuron_type,
                n_neurons=max(1, n_neurons),
                params=params,
                dt=node_dt,
            )
        )

    # Second pass: build connections from weight nodes
    for i, pop in enumerate(populations):
        weight_node_name = weight_source_for.get(pop.name)
        if weight_node_name is None:
            continue

        weight_data = pending_weights.get(weight_node_name)
        if weight_data is None:
            continue

        weights, bias = weight_data

        # Find the source population: the neuron or input node that feeds
        # the weight node
        src_name = ""
        for pred in predecessors.get(weight_node_name, []):
            pred_class = type(nodes[pred]).__name__
            if pred_class in _SC_NODE_TO_TYPE:
                src_name = pred
                break
            if pred_class == "SCInputNode":
                src_name = pred
                break

        if not src_name:
            # Use first predecessor
            preds = predecessors.get(weight_node_name, [])
            if preds:
                src_name = preds[0]
            else:
                src_name = input_pop or "input"

        connections.append(
            ConnectionSpec(
                src=src_name,
                dst=pop.name,
                weights=weights,
                bias=bias,
            )
        )

    # Determine effective input/output
    if not input_pop and populations:
        input_pop = populations[0].name
    if not output_pop and populations:
        output_pop = populations[-1].name

    if not populations:
        raise ValueError(
            "NeuronGraph requires at least one neuron population. "
            "The NIR graph may contain only pass-through nodes."
        )

    global_dt = dt if dt is not None else (populations[0].dt if populations else 1.0)

    graph = NeuronGraph(
        populations=populations,
        connections=connections,
        input_pop=input_pop,
        output_pop=output_pop,
        dt=global_dt,
    )

    logger.info(
        "Built NeuronGraph: %d populations, %d connections, %d neurons, %d synapses",
        len(populations), len(connections), graph.total_neurons, graph.total_synapses,
    )

    return graph
