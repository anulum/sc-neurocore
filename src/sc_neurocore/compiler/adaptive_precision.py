# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adaptive per-layer bitstream length assignment

"""Per-layer adaptive bitstream length for mixed-precision SC networks.

Different layers tolerate different amounts of SC quantization noise.
Shallow layers (close to input) can use short bitstreams (L=64) for speed,
while deep layers (close to output) need longer bitstreams (L=1024) for
precision. Uniform L wastes throughput on shallow layers.

This module:
1. Analyzes per-layer sensitivity to bitstream length via sweeps
2. Assigns optimal L_i per layer using Hoeffding bounds or empirical calibration
3. Outputs a precision map for the compiler to generate per-layer Verilog
   with different bitstream lengths

Reference: Sim & Lee 2019 — "Adjustable Sequence Length for SC NNs"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sc_neurocore.utils.bitstreams import adaptive_length


@dataclass
class LayerPrecision:
    """Bitstream length assignment for one layer."""

    layer_index: int
    name: str
    bitstream_length: int
    error_bound: float
    sensitivity: float


def analyze_sensitivity(
    layer_weights: list[np.ndarray],
    lengths: list[int] | None = None,
    n_trials: int = 100,
    seed: int = 42,
) -> list[float]:
    """Measure per-layer sensitivity to bitstream length reduction.

    For each layer, compute mean output error across trial inputs
    when reducing bitstream length from max to min. Layers with high
    sensitivity need longer bitstreams.

    Parameters
    ----------
    layer_weights : list of ndarray
        Weight matrices for each layer.
    lengths : list of int
        Bitstream lengths to sweep (default: [32, 64, 128, 256, 512, 1024]).
    n_trials : int
        Number of random input trials.
    seed : int
        Random seed.

    Returns
    -------
    list of float
        Per-layer sensitivity scores (higher = needs longer bitstream).
    """
    if lengths is None:
        lengths = [32, 64, 128, 256, 512, 1024]

    rng = np.random.RandomState(seed)
    sensitivities = []

    for w in layer_weights:
        n_in = w.shape[1] if w.ndim == 2 else w.shape[0]
        errors = []

        for _ in range(n_trials):
            x = rng.random(n_in)
            exact = x @ w.T if w.ndim == 2 else x * w

            length_errors = []
            for L in lengths:
                # SC computation: encode as bitstream, AND-multiply, popcount
                sc_results = []
                for trial in range(5):
                    bits_x = (rng.random((L, n_in)) < x).astype(np.float64)
                    if w.ndim == 2:
                        n_out = w.shape[0]
                        bits_w = np.zeros((L, n_out, n_in))
                        for j in range(n_out):
                            w_prob = np.clip(w[j], 0, 1)
                            bits_w[:, j, :] = (rng.random((L, n_in)) < w_prob).astype(np.float64)
                        and_result = bits_x[:, np.newaxis, :] * bits_w
                        sc_out = and_result.sum(axis=(0, 2)) / L
                    else:  # pragma: no cover — scalar weight path
                        w_prob = np.clip(w, 0, 1)
                        bits_w = (rng.random((L,)) < w_prob).astype(np.float64)
                        sc_out = (bits_x.mean(axis=0) * bits_w).mean()
                    sc_results.append(sc_out)

                sc_mean = np.mean(sc_results, axis=0)
                err = np.mean(np.abs(sc_mean - np.clip(exact, 0, None)))
                length_errors.append(err)

            # Sensitivity = how much error changes across length range
            sensitivity = max(length_errors) - min(length_errors) if length_errors else 0.0
            errors.append(sensitivity)

        sensitivities.append(float(np.mean(errors)))

    return sensitivities


def assign_lengths(
    layer_weights: list[np.ndarray],
    layer_names: list[str] | None = None,
    total_budget: int | None = None,
    min_length: int = 32,
    max_length: int = 1024,
    target_error: float = 0.01,
    method: str = "hoeffding",
) -> list[LayerPrecision]:
    """Assign per-layer bitstream lengths under a total budget.

    Parameters
    ----------
    layer_weights : list of ndarray
        Weight matrices for each layer.
    layer_names : list of str, optional
        Human-readable layer names.
    total_budget : int, optional
        Total bitstream cycles budget. If None, each layer gets its own
        minimum length for target_error.
    min_length, max_length : int
        Bounds on per-layer bitstream length.
    target_error : float
        Target per-layer accuracy (probability tolerance).
    method : str
        'hoeffding' uses Hoeffding bound, 'sensitivity' uses empirical sweep.

    Returns
    -------
    list of LayerPrecision
        Per-layer bitstream length assignments.
    """
    n_layers = len(layer_weights)
    if layer_names is None:
        layer_names = [f"layer_{i}" for i in range(n_layers)]

    if method == "hoeffding":
        assignments = []
        for i, (w, name) in enumerate(zip(layer_weights, layer_names)):
            fan_in = w.shape[1] if w.ndim == 2 else 1
            # Per-synapse error epsilon, aggregated over fan_in synapses
            per_syn_eps = target_error / max(1, np.sqrt(fan_in))
            L = adaptive_length(p=0.5, epsilon=per_syn_eps, confidence=0.95)
            L = int(np.clip(L, min_length, max_length))
            # Round up to power of 2 for hardware efficiency
            L = int(2 ** np.ceil(np.log2(max(L, min_length))))
            L = min(L, max_length)
            bound = 0.5 / np.sqrt(L) if L > 0 else 1.0
            assignments.append(
                LayerPrecision(
                    layer_index=i,
                    name=name,
                    bitstream_length=L,
                    error_bound=bound,
                    sensitivity=0.0,
                )
            )
        return assignments

    # Sensitivity-based assignment
    sensitivities = analyze_sensitivity(layer_weights)
    total_sens = sum(sensitivities) or 1.0

    if total_budget is None:  # pragma: no cover
        total_budget = max_length * n_layers

    assignments = []
    for i, (w, name, sens) in enumerate(zip(layer_weights, layer_names, sensitivities)):
        # Allocate budget proportional to sensitivity
        fraction = sens / total_sens
        L = int(fraction * total_budget / n_layers * n_layers)
        L = int(np.clip(L, min_length, max_length))
        L = int(2 ** np.ceil(np.log2(max(L, min_length))))
        L = min(L, max_length)
        bound = 0.5 / np.sqrt(L) if L > 0 else 1.0
        assignments.append(
            LayerPrecision(
                layer_index=i,
                name=name,
                bitstream_length=L,
                error_bound=bound,
                sensitivity=sens,
            )
        )

    return assignments
