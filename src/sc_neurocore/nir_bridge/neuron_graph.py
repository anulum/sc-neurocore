# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
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
nodes (Input, Output, Flatten, Threshold) are folded into the graph metadata
or the adjacent connection.  Scale nodes adjacent to a weight-carrying node are
folded into that connection's columns or rows.  Explicit NIR Delay nodes on the
source side of a weight-carrying node are preserved as
``ConnectionSpec.delay_steps`` when their per-channel delay is homogeneous.

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
    delay_steps : int
        Number of explicit unit-delay timesteps on this connection.  Recurrent
        NIR edges broken by the parser currently use ``1``; feed-forward
        connections use ``0``.
    source_threshold : np.ndarray | None
        Optional threshold vector applied to source signals before the weight
        matrix.  Represents NIR ``Threshold`` on the source side.
    destination_threshold : np.ndarray | None
        Optional threshold vector applied after this connection's affine
        accumulation and before the destination population input.
    """

    src: str
    dst: str
    weights: np.ndarray
    bias: np.ndarray | None = None
    delay_steps: int = 0
    source_threshold: np.ndarray | None = None
    destination_threshold: np.ndarray | None = None


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
            lines.append(f"    {pop.name}: {pop.neuron_type} × {pop.n_neurons}")
        lines.append("")
        lines.append("  Connections:")
        for conn in self.connections:
            shape = f"{conn.weights.shape[1]}→{conn.weights.shape[0]}"
            bias_str = " +bias" if conn.bias is not None else ""
            delay_str = f" delay={conn.delay_steps}" if conn.delay_steps else ""
            lines.append(f"    {conn.src} → {conn.dst}: {shape}{bias_str}{delay_str}")
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
    "SCConv1dNode",
    "SCSumPool2dNode",
    "SCAvgPool2dNode",
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

_DELAY_NODE_NAME = "SCDelayNode"
_SCALE_NODE_NAME = "SCScaleNode"
_FLATTEN_NODE_NAME = "SCFlattenNode"
_THRESHOLD_NODE_NAME = "SCThresholdNode"


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


def _homogeneous_delay_steps(node: Any, node_name: str) -> int:
    """Return a scalar delay for an explicit NIR Delay node or fail closed."""

    raw_steps = getattr(node, "delay_steps", None)
    if raw_steps is None:
        raise ValueError(f"Delay node {node_name!r} does not expose delay_steps")
    steps = np.atleast_1d(np.asarray(raw_steps, dtype=np.int64)).reshape(-1)
    if steps.size == 0:
        raise ValueError(f"Delay node {node_name!r} must contain at least one delay value")
    if np.any(steps < 0):
        raise ValueError(f"Delay node {node_name!r} contains negative delay_steps")
    if not np.all(steps == steps[0]):
        raise ValueError(
            f"Delay node {node_name!r} has heterogeneous delay_steps; "
            "split the delayed channels before FPGA lowering"
        )
    return int(steps[0])


def _scale_vector(node: Any, node_name: str) -> np.ndarray:
    """Return a finite one-dimensional scale vector from an SCScaleNode."""

    raw_scale = getattr(node, "scale", None)
    if raw_scale is None:
        raise ValueError(f"Scale node {node_name!r} does not expose scale")
    scale = np.atleast_1d(np.asarray(raw_scale, dtype=np.float32)).reshape(-1)
    if scale.size == 0:
        raise ValueError(f"Scale node {node_name!r} must contain at least one scale value")
    if not np.all(np.isfinite(scale)):
        raise ValueError(f"Scale node {node_name!r} contains non-finite scale values")
    return scale


def _threshold_vector(node: Any, node_name: str) -> np.ndarray:
    """Return a finite one-dimensional threshold vector from an SCThresholdNode."""

    raw_threshold = getattr(node, "threshold", None)
    if raw_threshold is None:
        raise ValueError(f"Threshold node {node_name!r} does not expose threshold")
    threshold = np.atleast_1d(np.asarray(raw_threshold, dtype=np.float32)).reshape(-1)
    if threshold.size == 0:
        raise ValueError(f"Threshold node {node_name!r} must contain at least one threshold")
    if not np.all(np.isfinite(threshold)):
        raise ValueError(f"Threshold node {node_name!r} contains non-finite threshold values")
    return threshold


def _compose_scale(left: np.ndarray | None, right: np.ndarray) -> np.ndarray:
    """Compose adjacent scale vectors under NumPy broadcasting rules."""

    if left is None:
        return right
    try:
        return np.multiply(left, right, dtype=np.float32)
    except ValueError as exc:
        raise ValueError(
            "Adjacent Scale nodes have incompatible shapes for FPGA lowering"
        ) from exc


def _broadcast_scale(scale: np.ndarray | None, size: int, label: str) -> np.ndarray | None:
    """Broadcast a scalar/vector scale to ``size`` or fail closed."""

    if scale is None:
        return None
    if scale.size == 1:
        return np.full(size, float(scale[0]), dtype=np.float32)
    if scale.size == size:
        return scale.astype(np.float32, copy=False)
    raise ValueError(f"{label} scale length {scale.size} does not match required width {size}")


def _broadcast_threshold(
    threshold: np.ndarray | None,
    size: int,
    label: str,
) -> np.ndarray | None:
    """Broadcast a scalar/vector threshold to ``size`` or fail closed."""

    if threshold is None:
        return None
    if threshold.size == 1:
        return np.full(size, float(threshold[0]), dtype=np.float32)
    if threshold.size == size:
        return threshold.astype(np.float32, copy=False)
    raise ValueError(f"{label} threshold length {threshold.size} does not match required width {size}")


def _shape_width(shape: tuple[int, ...] | None, *, node_name: str, label: str) -> int:
    """Return the element count for a finite NIR shape or fail closed."""

    if shape is None:
        raise ValueError(f"Flatten node {node_name!r} lacks {label} shape metadata")
    if not shape:
        return 1
    width = int(np.prod(np.asarray(shape, dtype=np.int64)))
    if width <= 0:
        raise ValueError(f"Flatten node {node_name!r} has invalid {label} shape {shape}")
    return width


def _flatten_widths(node: Any, node_name: str) -> tuple[int, int]:
    """Return input/output element counts for a shape-typed SCFlattenNode."""

    input_width = _shape_width(
        getattr(node, "input_shape", None),
        node_name=node_name,
        label="input",
    )
    output_width = _shape_width(
        getattr(node, "output_shape", None),
        node_name=node_name,
        label="output",
    )
    if input_width != output_width:
        raise ValueError(
            f"Flatten node {node_name!r} changes element count "
            f"from {input_width} to {output_width}"
        )
    return input_width, output_width


def _conv1d_to_dense_matrix(node: Any, node_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Lower a shape-known NIR Conv1d node to an exact dense matrix."""

    weight = np.asarray(getattr(node, "weight", None), dtype=np.float32)
    if weight.ndim != 3:
        raise ValueError(f"Conv1d node {node_name!r} weight must have shape (C_out, C_in/group, K)")

    input_shape = getattr(node, "input_shape", None)
    if input_shape is None:
        raise ValueError(
            f"Conv1d node {node_name!r} requires input_shape for FPGA lowering"
        )
    input_length = int(np.asarray(input_shape).reshape(-1)[0])
    if input_length <= 0:
        raise ValueError(f"Conv1d node {node_name!r} input_shape must be positive")

    stride = int(getattr(node, "stride", 1))
    padding = getattr(node, "padding", 0)
    if isinstance(padding, str):
        raise ValueError(
            f"Conv1d node {node_name!r} string padding requires explicit pre-lowering"
        )
    padding = int(padding)
    dilation = int(getattr(node, "dilation", 1))
    groups = int(getattr(node, "groups", 1))
    if stride <= 0 or padding < 0 or dilation <= 0 or groups <= 0:
        raise ValueError(f"Conv1d node {node_name!r} has invalid stride/padding/dilation/groups")

    out_channels, in_channels_per_group, kernel_size = weight.shape
    if out_channels % groups != 0:
        raise ValueError(f"Conv1d node {node_name!r} output channels must be divisible by groups")
    in_channels = in_channels_per_group * groups
    output_length = (input_length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    if output_length <= 0:
        raise ValueError(f"Conv1d node {node_name!r} output length is not positive")

    dense = np.zeros((out_channels * output_length, in_channels * input_length), dtype=np.float32)
    out_channels_per_group = out_channels // groups
    for out_channel in range(out_channels):
        group = out_channel // out_channels_per_group
        in_channel_offset = group * in_channels_per_group
        for out_pos in range(output_length):
            row = out_channel * output_length + out_pos
            for local_channel in range(in_channels_per_group):
                in_channel = in_channel_offset + local_channel
                for kernel_pos in range(kernel_size):
                    in_pos = out_pos * stride + kernel_pos * dilation - padding
                    if 0 <= in_pos < input_length:
                        col = in_channel * input_length + in_pos
                        dense[row, col] = weight[out_channel, local_channel, kernel_pos]

    raw_bias = getattr(node, "bias", None)
    if raw_bias is None:
        bias = np.zeros(out_channels, dtype=np.float32)
    else:
        bias = np.asarray(raw_bias, dtype=np.float32).reshape(-1)
    if bias.size != out_channels:
        raise ValueError(f"Conv1d node {node_name!r} bias length must equal output channels")

    return dense, np.repeat(bias, output_length).astype(np.float32, copy=False)


def _pool2d_to_dense_matrix(node: Any, node_name: str) -> tuple[np.ndarray, None]:
    """Lower a shape-known NIR Pool2d node to an exact dense matrix."""

    class_name = type(node).__name__
    input_shape = getattr(node, "input_shape", None)
    output_shape = getattr(node, "output_shape", None)
    if input_shape is None or output_shape is None:
        primitive = class_name.removeprefix("SC").removesuffix("Node")
        raise ValueError(
            f"{primitive} node {node_name!r} requires input/output shape metadata for FPGA lowering"
        )
    if len(input_shape) != 3 or len(output_shape) != 3:
        raise ValueError(f"{class_name} {node_name!r} requires CHW input/output shape metadata")

    channels, input_height, input_width = (int(value) for value in input_shape)
    out_channels, output_height, output_width = (int(value) for value in output_shape)
    if channels <= 0 or input_height <= 0 or input_width <= 0:
        raise ValueError(f"{class_name} {node_name!r} has invalid input shape {input_shape}")
    if out_channels != channels or output_height <= 0 or output_width <= 0:
        raise ValueError(f"{class_name} {node_name!r} has invalid output shape {output_shape}")

    kernel_height, kernel_width = (int(value) for value in node.kernel_size)
    stride_height, stride_width = (int(value) for value in node.stride)
    pad_height, pad_width = (int(value) for value in node.padding)
    if (
        kernel_height <= 0
        or kernel_width <= 0
        or stride_height <= 0
        or stride_width <= 0
        or pad_height < 0
        or pad_width < 0
    ):
        raise ValueError(f"{class_name} {node_name!r} has invalid kernel/stride/padding")

    dense = np.zeros(
        (channels * output_height * output_width, channels * input_height * input_width),
        dtype=np.float32,
    )
    coefficient = 1.0
    if class_name == "SCAvgPool2dNode":
        coefficient = 1.0 / float(kernel_height * kernel_width)

    for channel in range(channels):
        for out_y in range(output_height):
            for out_x in range(output_width):
                row = (channel * output_height + out_y) * output_width + out_x
                for kernel_y in range(kernel_height):
                    in_y = out_y * stride_height + kernel_y - pad_height
                    if not 0 <= in_y < input_height:
                        continue
                    for kernel_x in range(kernel_width):
                        in_x = out_x * stride_width + kernel_x - pad_width
                        if not 0 <= in_x < input_width:
                            continue
                        col = (channel * input_height + in_y) * input_width + in_x
                        dense[row, col] += coefficient

    return dense, None


def _weight_matrix_and_bias(node: Any, node_name: str) -> tuple[np.ndarray, np.ndarray | None]:
    """Return dense weight and bias arrays for a weight-carrying NIR node."""

    class_name = type(node).__name__
    if class_name == "SCConv1dNode":
        return _conv1d_to_dense_matrix(node, node_name)
    if class_name in {"SCSumPool2dNode", "SCAvgPool2dNode"}:
        return _pool2d_to_dense_matrix(node, node_name)

    weight = getattr(node, "weight", None)
    bias = getattr(node, "bias", None)
    if weight is None:
        weight = getattr(node, "weights", None)
    if weight is None:
        raise ValueError(f"Weight node {node_name!r} does not expose weights")
    dense_weight = np.asarray(weight, dtype=np.float32)
    dense_bias = None if bias is None else np.asarray(bias, dtype=np.float32)
    return dense_weight, dense_bias


def _node_logical_width(node: Any) -> int | None:
    """Return the flattened channel width for a source/destination node."""

    class_name = type(node).__name__
    if class_name == "SCInputNode":
        shape = getattr(node, "shape", None)
        if shape is None:
            return None
        if not shape:
            return 1
        return int(np.prod(np.asarray(shape, dtype=np.int64)))
    if class_name in _SC_NODE_TO_TYPE:
        return int(getattr(node, "n_neurons", 1))
    return None


def _resolve_weight_source(
    node_name: str,
    *,
    nodes: dict[str, Any],
    predecessors: dict[str, list[str]],
    accumulated_delay_steps: int = 0,
    accumulated_scale: np.ndarray | None = None,
    accumulated_threshold: np.ndarray | None = None,
    required_source_width: int | None = None,
    flatten_output_width: int | None = None,
    seen: frozenset[str] = frozenset(),
) -> tuple[str, int, np.ndarray | None, int | None, np.ndarray | None] | None:
    """Resolve the population/input source feeding a weight node.

    Traverses pass-through nodes immediately upstream of ``Affine``/``Linear``
    nodes and accumulates explicit NIR Delay metadata.  Ambiguous fan-in fails
    closed so the compiler does not invent a source for hardware handoff.
    """

    if node_name in seen:
        raise ValueError(f"Cycle while resolving weighted source through {node_name!r}")
    node = nodes[node_name]
    class_name = type(node).__name__
    if class_name in _SC_NODE_TO_TYPE or class_name == "SCInputNode":
        node_width = _node_logical_width(node)
        if (
            required_source_width is not None
            and node_width is not None
            and node_width != required_source_width
        ):
            raise ValueError(
                f"Flatten input width {required_source_width} does not match "
                f"source {node_name!r} width {node_width}"
            )
        return (
            node_name,
            accumulated_delay_steps,
            accumulated_scale,
            flatten_output_width,
            accumulated_threshold,
        )

    if class_name not in _SC_PASSTHROUGH_NODES:
        return None

    delay_steps = accumulated_delay_steps
    if class_name == _DELAY_NODE_NAME:
        delay_steps += _homogeneous_delay_steps(node, node_name)
    scale = accumulated_scale
    if class_name == _SCALE_NODE_NAME:
        scale = _compose_scale(scale, _scale_vector(node, node_name))
    threshold = accumulated_threshold
    if class_name == _THRESHOLD_NODE_NAME:
        if threshold is not None:
            raise ValueError(
                "Multiple source-side Threshold nodes on one connection require explicit "
                "pre-lowering before FPGA compilation"
            )
        threshold = _threshold_vector(node, node_name)
    next_required_source_width = required_source_width
    next_flatten_output_width = flatten_output_width
    if class_name == _FLATTEN_NODE_NAME:
        input_width, output_width = _flatten_widths(node, node_name)
        if required_source_width is not None and output_width != required_source_width:
            raise ValueError(
                f"Flatten output width {output_width} does not match downstream "
                f"source width {required_source_width}"
            )
        next_required_source_width = input_width
        next_flatten_output_width = output_width if flatten_output_width is None else flatten_output_width

    upstream = predecessors.get(node_name, [])
    if len(upstream) != 1:
        raise ValueError(
            f"Pass-through node {node_name!r} has {len(upstream)} upstream sources; "
            "explicit pre-lowering is required before FPGA compilation"
        )
    return _resolve_weight_source(
        upstream[0],
        nodes=nodes,
        predecessors=predecessors,
        accumulated_delay_steps=delay_steps,
        accumulated_scale=scale,
        accumulated_threshold=threshold,
        required_source_width=next_required_source_width,
        flatten_output_width=next_flatten_output_width,
        seen=seen | {node_name},
    )


def _resolve_weight_destination(
    node_name: str,
    *,
    nodes: dict[str, Any],
    successors: dict[str, list[str]],
    accumulated_scale: np.ndarray | None = None,
    accumulated_threshold: np.ndarray | None = None,
    required_destination_width: int | None = None,
    flatten_input_width: int | None = None,
    seen: frozenset[str] = frozenset(),
) -> tuple[str, np.ndarray | None, int | None, np.ndarray | None] | None:
    """Resolve the neuron destination fed by a weight node.

    Traverses pass-through nodes immediately downstream of ``Affine``/``Linear``
    and accumulates post-weight Scale metadata.  The scale is later folded into
    connection rows and bias terms.
    """

    if node_name in seen:
        raise ValueError(f"Cycle while resolving weighted destination through {node_name!r}")
    node = nodes[node_name]
    class_name = type(node).__name__
    if class_name in _SC_NODE_TO_TYPE:
        node_width = _node_logical_width(node)
        if (
            required_destination_width is not None
            and node_width is not None
            and node_width != required_destination_width
        ):
            raise ValueError(
                f"Flatten output width {required_destination_width} does not match "
                f"destination {node_name!r} width {node_width}"
            )
        return node_name, accumulated_scale, flatten_input_width, accumulated_threshold

    if class_name not in _SC_PASSTHROUGH_NODES or class_name in {"SCInputNode", "SCOutputNode"}:
        return None

    scale = accumulated_scale
    if class_name == _SCALE_NODE_NAME:
        scale = _compose_scale(scale, _scale_vector(node, node_name))
    threshold = accumulated_threshold
    if class_name == _THRESHOLD_NODE_NAME:
        if threshold is not None:
            raise ValueError(
                "Multiple post-weight Threshold nodes on one connection require explicit "
                "pre-lowering before FPGA compilation"
            )
        threshold = _threshold_vector(node, node_name)
    next_required_destination_width = required_destination_width
    next_flatten_input_width = flatten_input_width
    if class_name == _FLATTEN_NODE_NAME:
        input_width, output_width = _flatten_widths(node, node_name)
        if required_destination_width is not None and input_width != required_destination_width:
            raise ValueError(
                f"Flatten input width {input_width} does not match upstream "
                f"destination width {required_destination_width}"
            )
        next_required_destination_width = output_width
        next_flatten_input_width = input_width if flatten_input_width is None else flatten_input_width

    downstream = successors.get(node_name, [])
    if len(downstream) != 1:
        raise ValueError(
            f"Pass-through node {node_name!r} has {len(downstream)} downstream targets; "
            "explicit pre-lowering is required before FPGA compilation"
        )
    return _resolve_weight_destination(
        downstream[0],
        nodes=nodes,
        successors=successors,
        accumulated_scale=scale,
        accumulated_threshold=threshold,
        required_destination_width=next_required_destination_width,
        flatten_input_width=next_flatten_input_width,
        seen=seen | {node_name},
    )


def _fold_connection_scales(
    weights: np.ndarray,
    bias: np.ndarray | None,
    *,
    source_scale: np.ndarray | None,
    destination_scale: np.ndarray | None,
    src: str,
    dst: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fold adjacent Scale nodes into a connection's weights and bias."""

    folded_weights = np.asarray(weights, dtype=np.float32).copy()
    folded_bias = None if bias is None else np.asarray(bias, dtype=np.float32).copy()

    src_scale = _broadcast_scale(
        source_scale,
        folded_weights.shape[1],
        f"source-side Scale for connection {src}->{dst}",
    )
    if src_scale is not None:
        folded_weights *= src_scale[np.newaxis, :]

    dst_scale = _broadcast_scale(
        destination_scale,
        folded_weights.shape[0],
        f"post-weight Scale for connection {src}->{dst}",
    )
    if dst_scale is not None:
        folded_weights *= dst_scale[:, np.newaxis]
        if folded_bias is not None:
            folded_bias *= dst_scale

    return folded_weights, folded_bias


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
    # Maps a neuron node name → (weight node, post-weight scale, flatten width, threshold)
    weight_source_for: dict[
        str,
        tuple[str, np.ndarray | None, int | None, np.ndarray | None],
    ] = {}

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
            weight, bias = _weight_matrix_and_bias(node, name)
            pending_weights[name] = (weight, bias)

            # Find the neuron this feeds into, preserving post-weight
            # Scale nodes as row/bias multipliers.
            for succ in successors.get(name, []):
                resolved_dst = _resolve_weight_destination(
                    succ,
                    nodes=nodes,
                    successors=successors,
                )
                if resolved_dst is not None:
                    (
                        dst_name,
                        destination_scale,
                        destination_flatten_width,
                        destination_threshold,
                    ) = resolved_dst
                    weight_source_for[dst_name] = (
                        name,
                        destination_scale,
                        destination_flatten_width,
                        destination_threshold,
                    )
            continue

        if class_name in _SC_PASSTHROUGH_NODES:
            continue

        # Neuron node
        neuron_type = _SC_NODE_TO_TYPE.get(class_name)
        if neuron_type is None:
            logger.warning(
                "Skipping unsupported node type %s (%s) in FPGA compilation",
                class_name,
                name,
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
    for pop in populations:
        weight_source = weight_source_for.get(pop.name)
        if weight_source is None:
            continue
        (
            weight_node_name,
            destination_scale,
            destination_flatten_width,
            destination_threshold,
        ) = weight_source

        weight_data = pending_weights.get(weight_node_name)
        if weight_data is None:
            continue

        weights, bias = weight_data

        # Find the source population: the neuron or input node that feeds the
        # weight node, preserving homogeneous explicit NIR Delay nodes along
        # the immediate source path as scalar connection delay metadata.
        src_name = ""
        delay_steps = 0
        source_scale: np.ndarray | None = None
        source_flatten_width: int | None = None
        source_threshold: np.ndarray | None = None
        for pred in predecessors.get(weight_node_name, []):
            resolved = _resolve_weight_source(pred, nodes=nodes, predecessors=predecessors)
            if resolved is not None:
                (
                    src_name,
                    delay_steps,
                    source_scale,
                    source_flatten_width,
                    source_threshold,
                ) = resolved
                break

        if not src_name:
            # Use first predecessor
            preds = predecessors.get(weight_node_name, [])
            if preds:
                src_name = preds[0]
            else:
                src_name = input_pop or "input"

        if source_flatten_width is not None and source_flatten_width != int(weights.shape[1]):
            raise ValueError(
                f"Flatten output width {source_flatten_width} does not match "
                f"weight input width {int(weights.shape[1])} for connection {src_name}->{pop.name}"
            )
        if destination_flatten_width is not None and destination_flatten_width != int(
            weights.shape[0]
        ):
            raise ValueError(
                f"Flatten input width {destination_flatten_width} does not match "
                f"weight output width {int(weights.shape[0])} for connection {src_name}->{pop.name}"
            )
        source_threshold = _broadcast_threshold(
            source_threshold,
            int(weights.shape[1]),
            f"source-side Threshold for connection {src_name}->{pop.name}",
        )
        destination_threshold = _broadcast_threshold(
            destination_threshold,
            int(weights.shape[0]),
            f"post-weight Threshold for connection {src_name}->{pop.name}",
        )

        folded_weights, folded_bias = _fold_connection_scales(
            weights,
            bias,
            source_scale=source_scale,
            destination_scale=destination_scale,
            src=src_name,
            dst=pop.name,
        )

        connections.append(
            ConnectionSpec(
                src=src_name,
                dst=pop.name,
                weights=folded_weights,
                bias=folded_bias,
                delay_steps=delay_steps,
                source_threshold=source_threshold,
                destination_threshold=destination_threshold,
            )
        )

    # Recurrent edges are broken into _UnitDelayNode sources by SCNetwork.
    # If the delayed source is a weight-carrying node, rebuild the original
    # weighted recurrent connection with an explicit one-step delay marker so
    # SC-NIR and downstream HDL do not silently lose the feedback stream.
    recurrent_map = getattr(network, "_recurrent_map", {})
    for delay_name, recurrent_src in recurrent_map.items():
        weight_data = pending_weights.get(recurrent_src)
        if weight_data is None:
            continue

        src_name = ""
        for pred in predecessors.get(recurrent_src, []):
            if type(nodes[pred]).__name__ in _SC_NODE_TO_TYPE:
                src_name = pred
                break
        if not src_name:
            continue

        dst_names = [
            dst
            for dst in successors.get(delay_name, [])
            if type(nodes[dst]).__name__ in _SC_NODE_TO_TYPE
        ]
        if not dst_names:
            continue

        weights, bias = weight_data
        for dst_name in dst_names:
            connections.append(
                ConnectionSpec(
                    src=src_name,
                    dst=dst_name,
                    weights=weights,
                    bias=bias,
                    delay_steps=1,
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
        len(populations),
        len(connections),
        graph.total_neurons,
        graph.total_synapses,
    )

    return graph
