# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Layer length planner

"""Assign per-layer bitstream lengths for SC networks."""

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
    """Assign per-layer bitstream lengths under a total budget."""
    n_layers = len(layer_weights)
    if layer_names is None:
        layer_names = [f"layer_{i}" for i in range(n_layers)]

    if method == "hoeffding":
        assignments = []
        for i, (w, name) in enumerate(zip(layer_weights, layer_names)):
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

    sensitivities = analyze_sensitivity(layer_weights)
    total_sens = sum(sensitivities) or 1.0

    if total_budget is None:
        total_budget = max_length * n_layers

    assignments = []
    for i, (w, name, sens) in enumerate(zip(layer_weights, layer_names, sensitivities)):
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
