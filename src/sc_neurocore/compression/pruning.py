# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SNN weight and structural pruning

"""Weight, structural, and stochastic-aware pruning for SNN model compression.

Weight pruning: zero out weights below a magnitude threshold.
Structural pruning: remove entire neurons that fire below an
activity threshold, reducing layer width.
Stochastic pruning: score weights by bitstream contribution —
how many popcount bits they contribute per inference. SC-specific.

All methods reduce FPGA resource usage when combined with
Projection(weight_threshold=) for runtime sparsity exploitation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PruningReport:
    """Results of a pruning operation."""

    original_params: int
    pruned_params: int
    remaining_params: int
    sparsity: float
    original_neurons: int = 0
    pruned_neurons: int = 0


def prune_weights(
    weights: list[np.ndarray[Any, Any]],
    threshold: float = 0.01,
    method: str = "magnitude",
) -> tuple[list[np.ndarray[Any, Any]], PruningReport]:
    """Prune small weights from layer weight matrices.

    Parameters
    ----------
    weights : list of ndarray
        Weight matrices for each layer.
    threshold : float
        Pruning threshold. Weights with |w| <= threshold are zeroed.
    method : str
        'magnitude' (default): prune by absolute value.
        'percentile': treat threshold as percentile (0-100) of weight
        magnitudes to prune.

    Returns
    -------
    (pruned_weights, PruningReport)
    """
    pruned = []
    total_original = 0
    total_pruned = 0

    for w in weights:
        total_original += w.size
        w_copy = w.copy()

        if method == "percentile":
            abs_w = np.abs(w_copy)
            cutoff = np.percentile(abs_w[abs_w > 0], threshold) if np.any(abs_w > 0) else 0.0
            mask = abs_w <= cutoff
        else:
            mask = np.abs(w_copy) <= threshold

        w_copy[mask] = 0.0
        total_pruned += int(mask.sum())
        pruned.append(w_copy)

    remaining = total_original - total_pruned
    sparsity = total_pruned / max(total_original, 1)

    return pruned, PruningReport(
        original_params=total_original,
        pruned_params=total_pruned,
        remaining_params=remaining,
        sparsity=sparsity,
    )


def prune_neurons(
    weights: list[np.ndarray[Any, Any]],
    firing_rates: list[np.ndarray[Any, Any]] | None = None,
    activity_threshold: float = 0.001,
) -> tuple[list[np.ndarray[Any, Any]], PruningReport]:
    """Structural pruning: remove neurons with low firing rates.

    Removes entire rows from weight matrices (output neurons) and
    corresponding columns from the next layer's weight matrix (input
    connections). Reduces layer width, not just sparsity.

    Parameters
    ----------
    weights : list of ndarray
        Weight matrices [W1, W2, ...] where W_i has shape (n_out, n_in).
    firing_rates : list of ndarray, optional
        Per-neuron firing rates for each layer. If None, uses output
        weight magnitude as a proxy for importance.
    activity_threshold : float
        Neurons with firing rate (or weight norm) below this are pruned.

    Returns
    -------
    (pruned_weights, PruningReport)
    """
    n_layers = len(weights)
    pruned_weights = [w.copy() for w in weights]
    total_neurons = sum(w.shape[0] for w in weights)
    neurons_pruned = 0

    for i in range(n_layers):
        w = pruned_weights[i]
        n_out = w.shape[0]

        if firing_rates is not None and i < len(firing_rates):
            importance = firing_rates[i]
        else:
            importance = np.linalg.norm(w, axis=1)

        keep_mask = importance > activity_threshold
        if keep_mask.all():
            continue

        n_removed = int((~keep_mask).sum())
        neurons_pruned += n_removed

        pruned_weights[i] = w[keep_mask]

        if i + 1 < n_layers:
            pruned_weights[i + 1] = pruned_weights[i + 1][:, keep_mask]

    total_remaining = total_neurons - neurons_pruned

    original_params = sum(w.size for w in weights)
    remaining_params = sum(w.size for w in pruned_weights)

    return pruned_weights, PruningReport(
        original_params=original_params,
        pruned_params=original_params - remaining_params,
        remaining_params=remaining_params,
        sparsity=(original_params - remaining_params) / max(original_params, 1),
        original_neurons=total_neurons,
        pruned_neurons=neurons_pruned,
    )


def prune_stochastic(
    weights: list[np.ndarray[Any, Any]],
    bitstream_length: int = 256,
    min_popcount_bits: float = 1.0,
) -> tuple[list[np.ndarray[Any, Any]], PruningReport]:
    """Stochastic-aware pruning: score weights by bitstream contribution.

    In SC networks, weight w encodes probability p = clip(|w|, 0, 1).
    The expected popcount contribution per inference is:
        contribution = min(p, 1-p) * bitstream_length

    Weights that produce nearly-deterministic bitstreams (p near 0 or 1)
    contribute almost nothing to computation — they can be replaced with
    constant 0/1 gates, saving AND+popcount hardware.

    Parameters
    ----------
    weights : list of ndarray
        Weight matrices (values in [0, 1] for unipolar SC).
    bitstream_length : int
        Bitstream length (L). Longer streams = more bits per weight.
    min_popcount_bits : float
        Minimum expected popcount contribution to keep a weight.
        Weights contributing fewer bits than this are zeroed.

    Returns
    -------
    (pruned_weights, PruningReport)
    """
    pruned = []
    total_original = 0
    total_pruned = 0

    for w in weights:
        total_original += w.size
        w_copy = w.copy()

        # SC probability: clip to [0, 1]
        p = np.clip(np.abs(w_copy), 0.0, 1.0)
        # Expected popcount contribution: min(p, 1-p) * L
        # This is the "unpredictable" fraction of the bitstream
        contribution = np.minimum(p, 1.0 - p) * bitstream_length

        mask = contribution < min_popcount_bits
        w_copy[mask] = 0.0
        total_pruned += int(mask.sum())
        pruned.append(w_copy)

    remaining = total_original - total_pruned
    sparsity = total_pruned / max(total_original, 1)

    return pruned, PruningReport(
        original_params=total_original,
        pruned_params=total_pruned,
        remaining_params=remaining,
        sparsity=sparsity,
    )
