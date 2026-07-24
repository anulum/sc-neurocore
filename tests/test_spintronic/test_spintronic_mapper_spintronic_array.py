# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpintronicArray from former test_spintronic_mapper.py

"""Focused suite: TestSpintronicArray from former test_spintronic_mapper.py."""

from __future__ import annotations

from spintronic_mapper_support import *  # noqa: F403


class TestSpintronicArray:
    def test_creation(self):
        arr = SpintronicArray(4, 8)
        assert arr.total_cells == 32

    def test_total_area(self):
        arr = SpintronicArray(4, 4)
        assert arr.total_area_um2 > 0

    def test_program_and_read(self):
        arr = SpintronicArray(
            2,
            3,
            variability=VariabilityModel(
                width_sigma_pct=0,
                length_sigma_pct=0,
                ku_sigma_pct=0,
                dmi_sigma_pct=0,
                damping_sigma_pct=0,
                ms_sigma_pct=0,
            ),
        )
        w = np.array([[100, 200, 50], [250, 10, 180]], dtype=np.int32)
        arr.program_weights(w)
        rb = arr.read_weights()
        np.testing.assert_array_equal(rb, w)

    def test_state_from_weight(self):
        arr = SpintronicArray(
            1,
            2,
            variability=VariabilityModel(
                width_sigma_pct=0,
                length_sigma_pct=0,
                ku_sigma_pct=0,
                dmi_sigma_pct=0,
                damping_sigma_pct=0,
                ms_sigma_pct=0,
            ),
        )
        w = np.array([[50, 200]], dtype=np.int32)
        arr.program_weights(w)
        assert arr.cells[0][0].state == 0  # w=50 < 128 → P
        assert arr.cells[0][1].state == 1  # w=200 > 128 → AP
