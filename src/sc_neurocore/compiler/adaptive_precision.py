# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
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
from typing import Any

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


@dataclass(frozen=True)
class SynapsePrecision:
    """Precision assignment and conservative error bound for one synapse."""

    layer_index: int
    layer_name: str
    output_index: int
    input_index: int
    bit_width: int
    bitstream_length: int
    sensitivity: float
    quantization_error_bound: float
    stochastic_error_bound: float
    total_error_bound: float

    def to_dict(self) -> dict[str, int | float | str]:
        """Return a JSON-serialisable precision-plan row."""
        return {
            "layer_index": self.layer_index,
            "layer_name": self.layer_name,
            "output_index": self.output_index,
            "input_index": self.input_index,
            "bit_width": self.bit_width,
            "bitstream_length": self.bitstream_length,
            "sensitivity": self.sensitivity,
            "quantization_error_bound": self.quantization_error_bound,
            "stochastic_error_bound": self.stochastic_error_bound,
            "total_error_bound": self.total_error_bound,
        }


def analyze_sensitivity(
    layer_weights: list[np.ndarray[Any, Any]],
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


def assign_synapse_precisions(
    layer_weights: list[np.ndarray[Any, Any]],
    layer_names: list[str] | None = None,
    sensitivity_maps: list[np.ndarray[Any, Any]] | None = None,
    target_error: float = 0.01,
    min_bits: int = 4,
    max_bits: int = 16,
    min_length: int = 32,
    max_length: int = 4096,
    confidence: float = 0.95,
) -> list[SynapsePrecision]:
    """Assign per-synapse bit widths and SC lengths with error bounds.

    The bound is intentionally conservative: each synapse gets a local target
    derived from the global target, quantisation error is bounded by half an
    integer-quantisation step scaled by sensitivity, and stochastic sampling
    error uses the existing Hoeffding bitstream-length helper.
    """
    if target_error <= 0:
        raise ValueError("target_error must be positive")
    if min_bits < 1 or max_bits < min_bits:
        raise ValueError("bit-width bounds must satisfy 1 <= min_bits <= max_bits")
    if min_length < 1 or max_length < min_length:
        raise ValueError("length bounds must satisfy 1 <= min_length <= max_length")

    n_layers = len(layer_weights)
    if layer_names is None:
        layer_names = [f"layer_{i}" for i in range(n_layers)]
    if len(layer_names) != n_layers:
        raise ValueError("layer_names length must match layer_weights")
    if sensitivity_maps is not None and len(sensitivity_maps) != n_layers:
        raise ValueError("sensitivity_maps length must match layer_weights")

    total_synapses = sum(int(np.asarray(w).size) for w in layer_weights)
    local_target = target_error / max(1.0, float(np.sqrt(total_synapses)))
    assignments: list[SynapsePrecision] = []

    for layer_index, (weights, name) in enumerate(zip(layer_weights, layer_names)):
        w = np.asarray(weights, dtype=float)
        if w.ndim not in {1, 2}:
            raise ValueError("layer weights must be 1D or 2D arrays")
        matrix: np.ndarray[Any, Any] = w.reshape(1, -1) if w.ndim == 1 else w

        if sensitivity_maps is None:
            max_abs = float(np.max(np.abs(matrix))) if matrix.size else 0.0
            sensitivity = np.abs(matrix) / max(max_abs, 1e-12)
        else:
            sensitivity_raw = np.asarray(sensitivity_maps[layer_index], dtype=float)
            if sensitivity_raw.shape != w.shape:
                raise ValueError("each sensitivity map must match its layer weight shape")
            sensitivity = (
                sensitivity_raw.reshape(1, -1) if sensitivity_raw.ndim == 1 else sensitivity_raw
            )
            if np.any(sensitivity < 0) or not np.all(np.isfinite(sensitivity)):
                raise ValueError("sensitivity maps must contain finite non-negative values")

        for output_index in range(matrix.shape[0]):
            for input_index in range(matrix.shape[1]):
                sens = float(sensitivity[output_index, input_index])
                bit_width = _select_bit_width(sens, local_target, min_bits, max_bits)
                quant_bound = _quantization_error_bound(sens, bit_width)
                remaining = max(local_target - quant_bound, local_target * 0.25)
                if sens <= 0:
                    length = min_length
                    stochastic_bound = 0.0
                else:
                    epsilon = max(remaining / sens, 1e-12)
                    length = adaptive_length(
                        p=0.5,
                        epsilon=epsilon,
                        confidence=confidence,
                        min_length=min_length,
                        max_length=max_length,
                    )
                    stochastic_bound = sens * _hoeffding_radius(length, confidence)

                assignments.append(
                    SynapsePrecision(
                        layer_index=layer_index,
                        layer_name=name,
                        output_index=output_index,
                        input_index=input_index,
                        bit_width=bit_width,
                        bitstream_length=length,
                        sensitivity=sens,
                        quantization_error_bound=quant_bound,
                        stochastic_error_bound=stochastic_bound,
                        total_error_bound=quant_bound + stochastic_bound,
                    )
                )

    return assignments


def precision_plan_manifest(assignments: list[SynapsePrecision]) -> dict[str, Any]:
    """Build a deterministic manifest for a per-synapse precision plan."""
    rows = [assignment.to_dict() for assignment in assignments]
    return {
        "schema": "sc-neurocore.adaptive_precision_plan.v1",
        "granularity": "synapse",
        "num_synapses": len(assignments),
        "max_total_error_bound": max(
            (assignment.total_error_bound for assignment in assignments),
            default=0.0,
        ),
        "assignments": rows,
    }


def _select_bit_width(sensitivity: float, target: float, min_bits: int, max_bits: int) -> int:
    for bits in range(min_bits, max_bits + 1):
        if _quantization_error_bound(sensitivity, bits) <= target:
            return bits
    return max_bits


def _quantization_error_bound(sensitivity: float, bits: int) -> float:
    levels = max(1, 2**bits - 1)
    return float(sensitivity) * 0.5 / levels


def _hoeffding_radius(length: int, confidence: float) -> float:
    delta = 1.0 - confidence
    if delta <= 0:
        raise ValueError("confidence must be < 1.0")
    return float(np.sqrt(-np.log(delta / 2.0) / (2.0 * length)))


def assign_lengths(
    layer_weights: list[np.ndarray[Any, Any]],
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
