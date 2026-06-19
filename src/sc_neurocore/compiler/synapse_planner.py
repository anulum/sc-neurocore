# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Synapse precision planner

"""Per-synapse bit width and bitstream length assignment."""

from __future__ import annotations

from typing import Any

import numpy as np

from sc_neurocore.utils.bitstreams import adaptive_length

from .synapse_precision import SynapsePrecision


def _select_bit_width(sensitivity: float, target: float, min_bits: int, max_bits: int) -> int:
    for bits in range(min_bits, max_bits + 1):
        if _quantization_error_bound(sensitivity, bits) <= target:
            return bits
    return max_bits


def _quantization_error_bound(sensitivity: float, bits: int) -> float:
    levels: int = max(1, 2**bits - 1)
    return float(sensitivity) * 0.5 / levels


def _hoeffding_radius(length: int, confidence: float) -> float:
    delta = 1.0 - confidence
    if delta <= 0:
        raise ValueError("confidence must be < 1.0")
    return float(np.sqrt(-np.log(delta / 2.0) / (2.0 * length)))


def _precision_cost_summary(assignments: list[SynapsePrecision]) -> dict[str, float]:
    if not assignments:
        return {
            "estimated_lut_cost": 0.0,
            "uniform_length_reference_cost": 0.0,
            "estimated_lut_savings_vs_uniform_length": 0.0,
            "estimated_lut_savings_fraction_vs_uniform_length": 0.0,
        }
    max_length = max(assignment.bitstream_length for assignment in assignments)
    log2_max = float(np.log2(max_length))
    estimated = float(
        sum(
            assignment.bit_width * np.log2(assignment.bitstream_length)
            for assignment in assignments
        )
    )
    uniform_reference = float(sum(assignment.bit_width * log2_max for assignment in assignments))
    savings = uniform_reference - estimated
    savings_fraction = 0.0 if uniform_reference <= 0 else savings / uniform_reference
    return {
        "estimated_lut_cost": estimated,
        "uniform_length_reference_cost": uniform_reference,
        "estimated_lut_savings_vs_uniform_length": savings,
        "estimated_lut_savings_fraction_vs_uniform_length": savings_fraction,
    }


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
    """Assign per-synapse bit widths and SC lengths with error bounds."""
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
