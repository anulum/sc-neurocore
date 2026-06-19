# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adaptive runtime validation

"""Validation and coercion logic for dual-datapath parameters."""

from __future__ import annotations

import math
from typing import Any

from .manifest_gen import _precision_label, _precision_manifest
from .quantizer import BlockFloatingMode, QFormat, parse_precision_format


def _coerce_precision(
    precision: str | None,
    *,
    default_width: int,
    default_frac: int,
    tag: str,
    parameter_count: int | None = None,
) -> tuple[int, int, str, dict[str, Any], QFormat | BlockFloatingMode]:
    """Resolve concrete fixed-point datapath parameters and telemetry metadata."""
    if precision is None:
        q = QFormat(default_width - default_frac, default_frac)
        return (
            default_width,
            default_frac,
            q.q_label,
            _precision_manifest(
                q,
                source=f"{tag}:fallback",
                resolved_width=default_width,
                emitted_fraction=default_frac,
                kind="fixed",
                parameter_count=parameter_count,
            ),
            q,
        )

    try:
        parsed = parse_precision_format(precision)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{tag} precision must be a fixed Q-format or block-floating format"
        ) from exc
    if isinstance(parsed, QFormat):
        width = parsed.total_bits
        fraction = parsed.fraction_bits
        return (
            width,
            fraction,
            _precision_label(parsed, source=precision),
            _precision_manifest(
                parsed,
                source=precision,
                resolved_width=width,
                emitted_fraction=fraction,
                kind="fixed",
                parameter_count=parameter_count,
            ),
            parsed,
        )

    width = parsed.mantissa_bits
    fraction = parsed.emit_fraction
    return (
        width,
        fraction,
        _precision_label(parsed, source=precision),
        _precision_manifest(
            parsed,
            source=precision,
            resolved_width=width,
            emitted_fraction=fraction,
            kind="block_floating",
            parameter_count=parameter_count,
        ),
        parsed,
    )


def _validate_lp_hp(lp_width: int, lp_frac: int, hp_width: int, hp_frac: int) -> None:
    """Validate that the LP/HP pair is sensible."""
    if lp_width >= hp_width:
        raise ValueError(
            f"LP data_width ({lp_width}) must be strictly less than HP data_width ({hp_width})"
        )
    if lp_frac < 1:
        raise ValueError(f"LP fraction ({lp_frac}) must be >= 1")
    if hp_frac < 1:
        raise ValueError(f"HP fraction ({hp_frac}) must be >= 1")
    if lp_width < 2:
        raise ValueError(f"LP data_width ({lp_width}) must be >= 2")


def _validate_hysteresis(
    threshold_up_pct: float,
    threshold_down_pct: float,
    max_lp_code: int,
) -> None:
    """Validate adaptive-precision hysteresis thresholds."""
    if not math.isfinite(threshold_up_pct) or not math.isfinite(threshold_down_pct):
        raise ValueError("Threshold percentages must be finite")

    if not (0.0 < threshold_up_pct < 1.0):
        raise ValueError("threshold_up_pct must satisfy 0 < threshold_up_pct < 1")

    if not (0.0 < threshold_down_pct < threshold_up_pct):
        raise ValueError(
            "threshold_down_pct must satisfy 0 < threshold_down_pct < threshold_up_pct"
        )

    quantized_up = int(threshold_up_pct * max_lp_code)
    quantized_down = int(threshold_down_pct * max_lp_code)
    if not (1 <= quantized_down < quantized_up < max_lp_code):
        raise ValueError("Quantised threshold codes must satisfy 1 <= down < up < max_lp_code")
