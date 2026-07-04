# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for the compiler precision solver

"""Focused contracts for compiler precision solver range and budget heuristics."""

from __future__ import annotations

import math
from typing import Any, cast

import pytest

from sc_neurocore.compiler.precision_solver import (
    _min_bits_for_range,
    _min_frac_for_resolution,
    solve_precision,
)


def test_min_bits_for_range_handles_zero_signed_and_unsigned_ranges() -> None:
    """Integer-bit estimates cover zero, signed, and unsigned range contracts."""
    assert _min_bits_for_range(0.0, 0.0) == 1
    assert _min_bits_for_range(-3.0, 7.0) == 4
    assert _min_bits_for_range(-128.0, 127.0) == 8
    assert _min_bits_for_range(0.0, 7.0, signed=False) == 3


def test_min_bits_for_range_rejects_impossible_or_non_finite_ranges() -> None:
    """Integer-bit estimates fail closed for invalid or impossible ranges."""
    with pytest.raises(ValueError, match="finite"):
        _min_bits_for_range(math.inf, 1.0)
    with pytest.raises(ValueError, match="lower bound"):
        _min_bits_for_range(2.0, 1.0)
    with pytest.raises(ValueError, match="unsigned"):
        _min_bits_for_range(-1.0, 1.0, signed=False)


def test_min_frac_for_resolution_handles_fallback_and_positive_quantum() -> None:
    """Fraction-bit estimates cover invalid fallback and positive resolution paths."""
    assert _min_frac_for_resolution(0.0) == 16
    assert _min_frac_for_resolution(-0.125) == 16
    assert _min_frac_for_resolution(math.nan) == 16
    assert _min_frac_for_resolution(0.125) == 3
    assert _min_frac_for_resolution(0.2) == 3


def test_solve_precision_defaults_resolution_and_preserves_unsigned_ranges() -> None:
    """Default resolutions and unsigned ranges produce representable configs."""
    spec = solve_precision(
        {"flag": (0.0, 1.0), "counter": (0.0, 7.0)},
        min_resolution={"flag": 0.5},
        signed=False,
    )

    flag = spec.get("flag")
    counter = spec.get("counter")

    assert flag.signed is False
    assert flag.resolution <= 0.5
    assert flag.can_represent(1.0)
    assert counter.signed is False
    assert counter.resolution <= 0.01
    assert counter.can_represent(7.0)


def test_solve_precision_aligns_widths_without_reducing_requested_resolution() -> None:
    """Alignment rounds datapath widths while preserving requested fractions."""
    spec = solve_precision(
        {"slow": (-10.0, 10.0), "coarse": (-1.0, 1.0)},
        min_resolution={"slow": 0.125, "coarse": 0.5},
        align_to=2,
    )

    slow = spec.get("slow")
    coarse = spec.get("coarse")

    assert slow.data_width % 2 == 0
    assert coarse.data_width % 2 == 0
    assert slow.fraction == 3
    assert coarse.fraction == 1


def test_solve_precision_rejects_invalid_budget_and_alignment_arguments() -> None:
    """Solver options fail closed before precision rows are emitted."""
    bounds = {"v": (-1.0, 1.0)}

    with pytest.raises(TypeError, match="align_to"):
        solve_precision(bounds, align_to=cast(Any, True))
    with pytest.raises(TypeError, match="align_to"):
        solve_precision(bounds, align_to=cast(Any, 1.5))
    with pytest.raises(ValueError, match="align_to"):
        solve_precision(bounds, align_to=0)
    with pytest.raises(TypeError, match="max_total_bits"):
        solve_precision(bounds, max_total_bits=cast(Any, True))
    with pytest.raises(TypeError, match="max_total_bits"):
        solve_precision(bounds, max_total_bits=cast(Any, 1.5))
    with pytest.raises(ValueError, match="max_total_bits"):
        solve_precision(bounds, max_total_bits=0)


def test_solve_precision_reduces_largest_fractional_field_to_fit_budget() -> None:
    """Unaligned total-bit budgets reduce the highest-resolution variable first."""
    spec = solve_precision(
        {"slow": (-10.0, 10.0), "coarse": (-1.0, 1.0)},
        min_resolution={"slow": 0.125, "coarse": 0.5},
        max_total_bits=10,
    )

    slow = spec.get("slow")
    coarse = spec.get("coarse")

    assert slow.fraction == 2
    assert coarse.fraction == 1
    assert spec.total_bits == 10


def test_solve_precision_recomputes_aligned_widths_from_integer_floor() -> None:
    """Aligned budget reductions shrink the datapath when the next fraction fits."""
    spec = solve_precision(
        {"slow": (-10.0, 10.0), "coarse": (-1.0, 1.0)},
        min_resolution={"slow": 0.125, "coarse": 0.5},
        max_total_bits=10,
        align_to=2,
    )

    slow = spec.get("slow")
    coarse = spec.get("coarse")

    assert slow.data_width == 6
    assert slow.fraction == 1
    assert coarse.data_width == 4
    assert coarse.fraction == 1
    assert spec.total_bits == 10


def test_solve_precision_stops_when_budget_would_remove_last_fractional_bit() -> None:
    """A too-tight budget leaves one fractional bit instead of removing resolution."""
    spec = solve_precision({"v": (-10.0, 10.0)}, max_total_bits=5)
    config = spec.get("v")

    assert config.fraction == 1
    assert config.data_width == 6
    assert spec.total_bits > 5
    assert config.can_represent(10.0)
