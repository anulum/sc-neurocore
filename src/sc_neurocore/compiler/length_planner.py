# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Layer length planner

"""Validated per-layer bitstream-length planning for SC networks."""

from __future__ import annotations

from typing import Any

import numpy as np

from sc_neurocore.utils.bitstreams import adaptive_length

from .layer_precision import LayerPrecision
from .sensitivity_analysis import analyze_sensitivity


def assign_lengths(
    layer_weights: list[np.ndarray[Any, Any]],
    layer_names: list[str] | None = None,
    total_budget: int | None = None,
    min_length: int = 32,
    max_length: int = 1024,
    target_error: float = 0.01,
    method: str = "hoeffding",
) -> list[LayerPrecision]:
    """Assign per-layer bitstream lengths under a target error budget.

    Parameters
    ----------
    layer_weights:
        One- or two-dimensional finite weight tensors, one tensor per layer.
        One-dimensional tensors are treated as single-output layers.
    layer_names:
        Optional non-empty layer names. When provided, the list length must match
        `layer_weights` exactly.
    total_budget:
        Optional aggregate bitstream-length budget for sensitivity planning.
        When omitted, sensitivity planning uses `max_length * n_layers`.
    min_length:
        Minimum bitstream length assigned to any layer.
    max_length:
        Maximum bitstream length assigned to any layer.
    target_error:
        Positive per-layer target error used by the Hoeffding planner.
    method:
        Planning method. `hoeffding` uses analytic Hoeffding lengths;
        `sensitivity` and `proportional` allocate from sensitivity scores.

    Returns
    -------
    list[LayerPrecision]
        Validated layer precision rows in input-layer order.

    Raises
    ------
    ValueError
        If planner bounds, method, names, or weight tensors are invalid.
    """
    _validate_planner_bounds(min_length, max_length, target_error)
    if method not in {"hoeffding", "sensitivity", "proportional"}:
        raise ValueError("method must be one of: hoeffding, sensitivity, proportional")

    validated_weights = [_as_weight_array(weights) for weights in layer_weights]
    n_layers = len(validated_weights)
    if layer_names is None:
        layer_names = [f"layer_{i}" for i in range(n_layers)]
    elif len(layer_names) != n_layers:
        raise ValueError("layer_names length must match layer_weights")

    if method == "hoeffding":
        assignments: list[LayerPrecision] = []
        for i, (w, name) in enumerate(zip(validated_weights, layer_names)):
            fan_in = w.shape[1] if w.ndim == 2 else 1
            per_syn_eps = target_error / max(1, np.sqrt(fan_in))
            L = adaptive_length(p=0.5, epsilon=per_syn_eps, confidence=0.95)
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
                    sensitivity=0.0,
                )
            )
        return assignments

    sensitivities = analyze_sensitivity(validated_weights)
    total_sens = sum(sensitivities) or 1.0

    if total_budget is None:
        total_budget = max_length * n_layers
    elif total_budget <= 0:
        raise ValueError("total_budget must be positive when provided")

    assignments = []
    for i, (w, name, sens) in enumerate(zip(validated_weights, layer_names, sensitivities)):
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


def _validate_planner_bounds(min_length: int, max_length: int, target_error: float) -> None:
    """Validate scalar bounds shared by layer-length planners."""
    if min_length < 1 or max_length < min_length:
        raise ValueError("length bounds must satisfy 1 <= min_length <= max_length")
    if not np.isfinite(target_error) or target_error <= 0.0:
        raise ValueError("target_error must be finite and positive")


def _as_weight_array(weights: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Return a finite 1D/2D layer-weight array for planning."""
    array = np.asarray(weights, dtype=float)
    if array.ndim not in {1, 2}:
        raise ValueError("layer weights must be 1D or 2D arrays")
    if array.size == 0:
        raise ValueError("layer weights must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError("layer weights must contain finite values")
    return array
