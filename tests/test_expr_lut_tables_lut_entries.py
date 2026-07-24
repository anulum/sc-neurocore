# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLutEntries from former test_expr_lut_tables.py

"""Focused suite: TestLutEntries from former test_expr_lut_tables.py."""

from __future__ import annotations

from tests.expr_lut_tables_support import *  # noqa: F403


class TestLutEntries:
    def test_symmetric_luts_have_256_entries(self) -> None:
        assert len(tables.exp_lut_entries(16, 8)) == 256
        assert len(tables.tanh_lut_entries(8)) == 256
        assert len(tables.cosh_lut_entries(16, 8)) == 256
        assert len(tables.exprel_lut_entries(16, 8)) == 256
        assert len(tables.sigmoid_lut_entries(8)) == 256
        assert len(tables.sin_lut_entries(8)) == 256
        assert len(tables.cos_lut_entries(8)) == 256
        assert len(tables.cbrt_lut_entries(8)) == 256

    def test_log_and_sqrt_lut_sizes(self) -> None:
        assert len(tables.log_lut_entries(8)) == 256
        assert len(tables.sqrt_lut_entries(8)) == 16

    def test_log_entries_match_positive_grid(self) -> None:
        entries = tables.log_lut_entries(8)
        points = tables.log_sample_points()
        assert entries[0] == int(round(math.log(1.0 / 256.0) * 256))
        assert entries[128] == int(round(math.log(points[128]) * 256))
        assert entries[-1] == int(round(math.log(points[-1]) * 256))

    def test_sqrt_entries_match_non_negative_grid(self) -> None:
        entries = tables.sqrt_lut_entries(8)
        points = tables.sqrt_sample_points()
        assert entries[0] == 0
        assert entries[2] == 256
        assert entries[8] == 512
        assert entries[-1] == int(round(math.sqrt(points[-1]) * 256))

    def test_exp_zero_point_and_saturation(self) -> None:
        exp = tables.exp_lut_entries(16, 8)
        # x = 0 at index 128 -> exp(0) * 256 = 256.
        assert exp[128] == 256
        # Large positive x saturates to the signed 16-bit max.
        assert exp[255] == (1 << 15) - 1

    def test_saturation_cap_tracks_width(self) -> None:
        # An 8-bit word saturates far lower than a 16-bit word.
        exp8 = tables.exp_lut_entries(8, 4)
        assert max(exp8) == (1 << 7) - 1

    def test_tanh_is_bounded(self) -> None:
        tanh = tables.tanh_lut_entries(8)
        assert max(tanh) <= (1 << 8)
        assert min(tanh) >= -(1 << 8)

    def test_exprel_limit_at_zero_is_one(self) -> None:
        # exprel(0) = 1 -> 1 * 256 = 256 at index 128.
        assert tables.exprel_lut_entries(16, 8)[128] == 256

    def test_all_entries_are_int(self) -> None:
        for entry in tables.exp_lut_entries(16, 8):
            assert isinstance(entry, int)
        for entry in tables.sqrt_lut_entries(8):
            assert isinstance(entry, int)

    def test_cbrt_is_odd_symmetric(self) -> None:
        cbrt = tables.cbrt_lut_entries(8)
        # index 128 is x=0; index 128+k and 128-k are negatives of each other.
        assert cbrt[128] == 0
        assert cbrt[128 + 40] == -cbrt[128 - 40]

    def test_sin_matches_reference(self) -> None:
        sin = tables.sin_lut_entries(8)
        pts = tables.symmetric_sample_points()
        assert sin[200] == int(round(math.sin(pts[200]) * 256))
