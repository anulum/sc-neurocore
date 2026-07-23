# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSamplePoints from former test_expr_lut_tables.py

"""Focused suite: TestSamplePoints from former test_expr_lut_tables.py."""

from __future__ import annotations

from tests.expr_lut_tables_support import *  # noqa: F403

class TestSamplePoints:
    def test_length_and_endpoints(self) -> None:
        pts = tables.symmetric_sample_points()
        assert len(pts) == 256
        assert pts[0] == -16.0
        assert pts[128] == 0.0
        assert pts[-1] == -16.0 + 255 * 0.125

    def test_deterministic(self) -> None:
        assert tables.symmetric_sample_points() == tables.symmetric_sample_points()

    def test_log_grid_is_positive_power_of_two_geometry(self) -> None:
        points = tables.log_sample_points()
        assert len(points) == tables.LOG_LUT_SIZE == 256
        assert points[0] == tables.LOG_LUT_MIN == 1.0 / 256.0
        assert points[1] - points[0] == tables.LOG_LUT_STEP == 1.0 / 32.0
        assert points[-1] < tables.LOG_LUT_MIN + tables.LOG_LUT_SIZE * tables.LOG_LUT_STEP

    def test_sqrt_grid_is_non_negative_half_unit_geometry(self) -> None:
        points = tables.sqrt_sample_points()
        assert len(points) == tables.SQRT_LUT_SIZE == 16
        assert points[0] == tables.SQRT_LUT_MIN == 0.0
        assert points[1] - points[0] == tables.SQRT_LUT_STEP == 0.5
        assert points[-1] == 7.5
