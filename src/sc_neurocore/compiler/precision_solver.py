# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Precision solver

"""Constraint-based solver for optimal per-variable precision."""

from __future__ import annotations

import math

from .precision_config import PrecisionConfig
from .mixed_precision_spec import MixedPrecisionSpec


def _min_bits_for_range(lo: float, hi: float, signed: bool = True) -> int:
    """Compute minimum integer bits to cover a value range."""
    abs_max = max(abs(lo), abs(hi))
    if abs_max == 0:
        return 1
    return math.ceil(math.log2(abs_max + 1)) + (1 if signed else 0)


def _min_frac_for_resolution(resolution: float) -> int:
    """Compute minimum fractional bits for a target resolution."""
    if resolution <= 0:
        return 16  # Default to high precision
    return math.ceil(math.log2(1.0 / resolution))


def solve_precision(
    bounds: dict[str, tuple[float, float]],
    *,
    min_resolution: dict[str, float] | None = None,
    max_total_bits: int | None = None,
    signed: bool = True,
    align_to: int = 1,
) -> MixedPrecisionSpec:
    """Automatically solve for optimal per-variable precision.

    Uses a constraint-based approach to select data widths and fractions.

    Parameters
    ----------
    bounds : dict
        Mapping from variable name to (min, max) value bounds.
    min_resolution : dict, optional
        Mapping from variable name to minimum required resolution.
        Defaults to 0.01 for all variables.
    max_total_bits : int, optional
        If set, the solver will reduce fractional bits to fit.
    signed : bool
        Whether to use signed formats.
    align_to : int
        Align each variable's data_width to this multiple.

    Returns
    -------
    MixedPrecisionSpec
        Optimal per-variable precision configuration.
    """
    if min_resolution is None:
        min_resolution = {v: 0.01 for v in bounds}

    configs: dict[str, PrecisionConfig] = {}

    for var, (lo, hi) in bounds.items():
        int_bits = _min_bits_for_range(lo, hi, signed)
        res = min_resolution.get(var, 0.01)
        frac_bits = _min_frac_for_resolution(res)

        # Sign bit
        sign_bits = 1 if signed else 0
        total = sign_bits + int_bits + frac_bits

        # Align
        if align_to > 1:
            total = math.ceil(total / align_to) * align_to

        configs[var] = PrecisionConfig(
            data_width=total,
            fraction=frac_bits,
            signed=signed,
        )

    spec = MixedPrecisionSpec(configs)

    # If total exceeds budget, iteratively reduce least-sensitive fractions
    if max_total_bits is not None:
        while spec.total_bits > max_total_bits:
            # Find the variable with the most fractional bits — reduce it
            worst = max(
                spec.var_configs.keys(),
                key=lambda v: spec.var_configs[v].fraction,
            )
            old = spec.var_configs[worst]
            if old.fraction <= 1:
                break  # Can't reduce further
            new_frac = old.fraction - 1
            new_dw = old.data_width - 1
            if align_to > 1:
                new_dw = math.ceil(new_dw / align_to) * align_to
            spec.var_configs[worst] = PrecisionConfig(
                data_width=new_dw,
                fraction=new_frac,
                signed=signed,
            )

    return spec
