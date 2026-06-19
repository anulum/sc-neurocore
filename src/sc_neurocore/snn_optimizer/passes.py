# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SNN optimization passes

"""Composable optimization passes for SNN computation graphs.

Like LLVM but for spiking networks: dead neuron elimination, layer fusion,
redundancy elimination. Each pass operates on an SNNGraph IR, transforms
it, and reports what changed.

No SNN framework has an optimizer pass manager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class LayerNode:
    """One layer in the SNN computation graph."""

    name: str
    n_inputs: int
    n_neurons: int
    weights: np.ndarray[Any, Any]
    neuron_type: str = "LIF"
    firing_rates: np.ndarray[Any, Any] | None = None

    @property
    def n_params(self) -> int:
        return self.weights.size


@dataclass
class SNNGraph:
    """SNN computation graph: sequence of layers."""

    layers: list[LayerNode] = field(default_factory=list)

    @property
    def total_params(self) -> int:
        return sum(layer.n_params for layer in self.layers)

    @property
    def total_neurons(self) -> int:
        return sum(layer.n_neurons for layer in self.layers)

    def copy(self) -> SNNGraph:
        return SNNGraph(
            layers=[
                LayerNode(
                    name=l.name,
                    n_inputs=l.n_inputs,
                    n_neurons=l.n_neurons,
                    weights=l.weights.copy(),
                    neuron_type=l.neuron_type,
                    firing_rates=l.firing_rates.copy() if l.firing_rates is not None else None,
                )
                for l in self.layers
            ]
        )


@dataclass
class PassResult:
    """Result of one optimization pass."""

    name: str
    neurons_removed: int = 0
    layers_fused: int = 0
    params_before: int = 0
    params_after: int = 0


@dataclass
class OptimizationReport:
    """Report from running all optimization passes."""

    pass_results: list[PassResult] = field(default_factory=list)
    params_before: int = 0
    params_after: int = 0
    neurons_before: int = 0
    neurons_after: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.params_after == 0:  # pragma: no cover
            return 0.0
        return self.params_before / self.params_after

    def summary(self) -> str:
        lines = [
            f"SNN Optimizer: {self.params_before} -> {self.params_after} params "
            f"({self.compression_ratio:.2f}x compression)",
            f"  Neurons: {self.neurons_before} -> {self.neurons_after}",
        ]
        for pr in self.pass_results:
            lines.append(
                f"  [{pr.name}] removed {pr.neurons_removed} neurons, "
                f"fused {pr.layers_fused} layers"
            )
        return "\n".join(lines)


def dead_neuron_elimination(graph: SNNGraph, threshold: float = 0.001) -> PassResult:
    """Remove neurons that never fire (firing rate below threshold).

    Requires firing_rates to be set on each layer (from profiling).
    Removes rows from weight matrices and corresponding columns from
    the next layer's weight matrix.
    """
    result = PassResult(name="dead_neuron_elimination", params_before=graph.total_params)
    total_removed = 0

    for i, layer in enumerate(graph.layers):
        if layer.firing_rates is None:
            continue

        keep_mask = layer.firing_rates > threshold
        if keep_mask.all():
            continue

        n_removed = int((~keep_mask).sum())
        total_removed += n_removed

        # Remove dead neurons from this layer
        layer.weights = layer.weights[keep_mask]
        layer.n_neurons = layer.weights.shape[0]
        if layer.firing_rates is not None:
            layer.firing_rates = layer.firing_rates[keep_mask]

        # Remove corresponding input columns from next layer
        if i + 1 < len(graph.layers):
            next_layer = graph.layers[i + 1]
            next_layer.weights = next_layer.weights[:, keep_mask]
            next_layer.n_inputs = next_layer.weights.shape[1]

    result.neurons_removed = total_removed
    result.params_after = graph.total_params
    return result


def layer_fusion(graph: SNNGraph) -> PassResult:
    """Fuse adjacent linear layers with compatible dimensions.

    If two consecutive layers have no nonlinearity between them
    (both LIF with same type), fuse W2 @ W1 into a single layer.
    Caveat: this is valid only when the intermediate layer has
    effectively linear behavior (high threshold, no spikes).
    """
    result = PassResult(name="layer_fusion", params_before=graph.total_params)
    fused = 0

    i = 0
    while i < len(graph.layers) - 1:
        curr = graph.layers[i]
        nxt = graph.layers[i + 1]

        # Only fuse if intermediate layer has negligible firing
        can_fuse = (
            curr.firing_rates is not None
            and curr.firing_rates.max() < 0.01
            and curr.neuron_type == nxt.neuron_type
        )

        if can_fuse:
            fused_weights = nxt.weights @ curr.weights
            fused_node = LayerNode(
                name=f"{curr.name}+{nxt.name}",
                n_inputs=curr.n_inputs,
                n_neurons=nxt.n_neurons,
                weights=fused_weights,
                neuron_type=nxt.neuron_type,
                firing_rates=nxt.firing_rates,
            )
            graph.layers[i] = fused_node
            graph.layers.pop(i + 1)
            fused += 1
        else:
            i += 1

    result.layers_fused = fused
    result.params_after = graph.total_params
    return result


def redundancy_elimination(graph: SNNGraph, correlation_threshold: float = 0.99) -> PassResult:
    """Merge neurons with near-identical weight vectors.

    If two neurons in the same layer have weight correlation > threshold,
    merge them: keep one, remove the other, scale outgoing weights by 2.
    """
    result = PassResult(name="redundancy_elimination", params_before=graph.total_params)
    total_removed = 0

    for i, layer in enumerate(graph.layers):
        if layer.n_neurons < 2:
            continue

        W = layer.weights
        keep = np.ones(layer.n_neurons, dtype=bool)
        merged_into: dict[int, int] = {}

        for a in range(layer.n_neurons):
            if not keep[a]:
                continue
            for b in range(a + 1, layer.n_neurons):
                if not keep[b]:  # pragma: no cover
                    continue
                norms = np.linalg.norm(W[a]) * np.linalg.norm(W[b])
                if norms < 1e-10:  # pragma: no cover
                    continue
                corr = np.dot(W[a], W[b]) / norms
                if corr > correlation_threshold:
                    keep[b] = False
                    merged_into[b] = a
                    total_removed += 1

        if total_removed == 0:
            continue

        # Scale weights of kept neurons that absorbed others
        for removed, keeper in merged_into.items():
            W[keeper] = (W[keeper] + W[removed]) / 2.0

        layer.weights = W[keep]
        layer.n_neurons = layer.weights.shape[0]
        if layer.firing_rates is not None:
            layer.firing_rates = layer.firing_rates[keep]

        if i + 1 < len(graph.layers):
            nxt = graph.layers[i + 1]
            # Sum columns of removed neurons into their keepers
            new_next_w = nxt.weights[:, keep]
            nxt.weights = new_next_w
            nxt.n_inputs = new_next_w.shape[1]

    result.neurons_removed = total_removed
    result.params_after = graph.total_params
    return result


def optimize(
    graph: SNNGraph,
    passes: list[str] | None = None,
) -> tuple[SNNGraph, OptimizationReport]:
    """Run optimization passes on an SNN graph.

    Parameters
    ----------
    graph : SNNGraph
    passes : list of str, optional
        Pass names to run. Default: all passes.
        Options: 'dead_neuron_elimination', 'layer_fusion', 'redundancy_elimination'

    Returns
    -------
    (optimized_graph, OptimizationReport)
    """
    if passes is None:
        passes = ["dead_neuron_elimination", "redundancy_elimination", "layer_fusion"]

    pass_map = {
        "dead_neuron_elimination": dead_neuron_elimination,
        "layer_fusion": layer_fusion,
        "redundancy_elimination": redundancy_elimination,
    }

    report = OptimizationReport(
        params_before=graph.total_params,
        neurons_before=graph.total_neurons,
    )

    optimized = graph.copy()
    for pass_name in passes:
        fn = pass_map.get(pass_name)
        if fn is None:
            continue
        result = fn(optimized)
        report.pass_results.append(result)

    report.params_after = optimized.total_params
    report.neurons_after = optimized.total_neurons
    return optimized, report
