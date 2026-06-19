# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Auto-tune precision

"""Studio-facing API for automatic precision/resource optimization."""

from __future__ import annotations

from typing import Any

import numpy as np

from .synapse_planner import _precision_cost_summary, assign_synapse_precisions


def precision_plan_manifest(assignments: list[Any]) -> dict[str, Any]:
    """Build a deterministic manifest for a per-synapse precision plan."""
    rows = [assignment.to_dict() for assignment in assignments]
    cost_summary = _precision_cost_summary(assignments)
    return {
        "schema": "sc-neurocore.adaptive_precision_plan.v1",
        "granularity": "synapse",
        "num_synapses": len(assignments),
        "max_total_error_bound": max(
            (assignment.total_error_bound for assignment in assignments),
            default=0.0,
        ),
        "cost_summary": cost_summary,
        "assignments": rows,
    }


def auto_tune_synapse_precisions(
    layer_weights: list[np.ndarray[Any, Any]],
    *,
    layer_names: list[str] | None = None,
    target_error_percent: float = 0.1,
    min_bits: int = 4,
    max_bits: int = 16,
    min_length: int = 32,
    max_length: int = 4096,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Auto-tune per-synapse precision for an explicit percent error target."""
    if target_error_percent <= 0:
        raise ValueError("target_error_percent must be positive")
    target_error = target_error_percent / 100.0
    assignments = assign_synapse_precisions(
        layer_weights,
        layer_names=layer_names,
        target_error=target_error,
        min_bits=min_bits,
        max_bits=max_bits,
        min_length=min_length,
        max_length=max_length,
        confidence=confidence,
    )
    manifest = precision_plan_manifest(assignments)
    manifest["api_surface"] = {
        "action_id": "auto_tune_adaptive_precision",
        "target_error_percent": target_error_percent,
        "target_error_fraction": target_error,
        "objective": "minimal_luts_under_error_target",
        "cost_metric": "sum(bit_width * log2(bitstream_length))",
        "estimated_lut_cost": manifest["cost_summary"]["estimated_lut_cost"],
        "uniform_length_reference_cost": manifest["cost_summary"]["uniform_length_reference_cost"],
        "estimated_lut_savings_vs_uniform_length": manifest["cost_summary"][
            "estimated_lut_savings_vs_uniform_length"
        ],
    }
    return manifest
