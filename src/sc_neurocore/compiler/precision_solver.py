# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Precision solver

"""Heuristic solver for per-variable fixed-point precision budgets."""

from __future__ import annotations

import math

from .mixed_precision_spec import MixedPrecisionSpec
from .precision_config import BlockFloatingPrecisionConfig, PrecisionConfig


def _min_bits_for_range(lo: float, hi: float, signed: bool = True) -> int:
    """Return a conservative integer-bit count for a closed value range."""
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ValueError("range bounds must be finite")
    if lo > hi:
        raise ValueError("range lower bound must not exceed upper bound")
    if signed:
        bits = 1
        while lo < -(1 << (bits - 1)) or hi >= (1 << (bits - 1)):
            bits += 1
        return bits
    if lo < 0:
        raise ValueError("unsigned precision cannot represent a negative range")
    bits = 1
    while hi >= (1 << bits):
        bits += 1
    return bits


def _min_frac_for_resolution(resolution: float) -> int:
    """Return fractional bits needed for a target resolution quantum."""
    if not math.isfinite(resolution) or resolution <= 0:
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
    """Solve a deterministic per-variable fixed-point precision assignment.

    The solver derives integer bits from each value range and fractional bits
    from the requested resolution. Optional total-bit budgets reduce the largest
    fractional fields first until the budget is met or every reduced variable is
    already at one fractional bit.

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
        Per-variable precision configuration derived from range, resolution,
        alignment, signedness, and optional total-bit budget constraints.
    """
    if not isinstance(align_to, int) or isinstance(align_to, bool):
        raise TypeError("align_to must be an integer")
    if align_to < 1:
        raise ValueError("align_to must be positive")
    if max_total_bits is not None:
        if not isinstance(max_total_bits, int) or isinstance(max_total_bits, bool):
            raise TypeError("max_total_bits must be an integer or None")
        if max_total_bits < 1:
            raise ValueError("max_total_bits must be positive")

    if min_resolution is None:
        min_resolution = {v: 0.01 for v in bounds}

    configs: dict[str, PrecisionConfig | BlockFloatingPrecisionConfig] = {}
    integer_widths: dict[str, int] = {}

    for var, (lo, hi) in bounds.items():
        int_bits = _min_bits_for_range(lo, hi, signed)
        integer_widths[var] = int_bits
        res = min_resolution.get(var, 0.01)
        frac_bits = _min_frac_for_resolution(res)

        total = int_bits + frac_bits

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
            new_dw = integer_widths[worst] + new_frac
            if align_to > 1:
                new_dw = math.ceil(new_dw / align_to) * align_to
            spec.var_configs[worst] = PrecisionConfig(
                data_width=new_dw,
                fraction=new_frac,
                signed=signed,
            )

    return spec
