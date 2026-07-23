# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestConstraintSolver from former test_bus_mixed_precision.py

"""Focused suite: TestConstraintSolver from former test_bus_mixed_precision.py."""

from __future__ import annotations

from tests.bus_mixed_precision_support import *  # noqa: F403

class TestConstraintSolver:
    """Test the automatic precision constraint solver."""

    def test_basic_solve(self) -> None:
        """Should produce valid configs from bounds."""
        spec = solve_precision(
            bounds={"v": (-128, 127), "u": (-10, 10)},
        )
        assert spec.get("v").can_represent(-128)
        assert spec.get("v").can_represent(127)
        assert spec.get("u").can_represent(-10)
        assert spec.get("u").can_represent(10)

    def test_resolution_honoured(self) -> None:
        """Requested resolution should be achievable."""
        spec = solve_precision(
            bounds={"v": (-1, 1)},
            min_resolution={"v": 0.001},
        )
        assert spec.get("v").resolution <= 0.001

    def test_budget_constraint(self) -> None:
        """Should reduce precision to fit bit budget."""
        spec = solve_precision(
            bounds={"v": (-128, 127), "u": (-10, 10)},
            max_total_bits=24,
        )
        assert spec.total_bits <= 24

    def test_alignment(self) -> None:
        """Byte alignment should round up data widths."""
        spec = solve_precision(
            bounds={"v": (-1, 1)},
            min_resolution={"v": 0.01},
            align_to=8,
        )
        assert spec.get("v").data_width % 8 == 0

    def test_single_variable(self) -> None:
        """Should work with a single variable."""
        spec = solve_precision(
            bounds={"x": (0, 255)},
            min_resolution={"x": 0.1},
        )
        assert spec.get("x").can_represent(255)

    def test_mixed_ranges(self) -> None:
        """Variables with very different ranges get different widths."""
        spec = solve_precision(
            bounds={"v": (-32768, 32767), "flag": (0, 1)},
            min_resolution={"v": 0.01, "flag": 0.5},
        )
        assert spec.get("v").data_width > spec.get("flag").data_width
