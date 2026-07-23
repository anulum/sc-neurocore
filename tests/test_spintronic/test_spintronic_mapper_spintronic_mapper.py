# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpintronicMapper from former test_spintronic_mapper.py

"""Focused suite: TestSpintronicMapper from former test_spintronic_mapper.py."""

from __future__ import annotations

from spintronic_mapper_support import *  # noqa: F403

class TestSpintronicMapper:
    def test_map_network(self):
        mapper = SpintronicMapper()
        w = np.random.default_rng(42).integers(0, 256, (8, 16), dtype=np.int32)
        arr, result = mapper.map_network(w)
        assert result.array_rows == 8
        assert result.array_cols == 16
        assert result.total_energy_fj > 0

    def test_all_techs(self):
        w = np.ones((4, 4), dtype=np.int32) * 128
        for tech in SpintronicTech:
            mapper = SpintronicMapper(tech=tech)
            arr, result = mapper.map_network(w)
            assert result.tech == tech
            assert result.total_area_um2 > 0

    def test_monte_carlo_yield(self):
        mapper = SpintronicMapper()
        w = np.ones((4, 4), dtype=np.int32) * 128
        yld = mapper.monte_carlo_yield(w, n_trials=50, tolerance_q88=128)
        assert 0.0 <= yld <= 1.0

    def test_yield_high_tolerance(self):
        mapper = SpintronicMapper()
        w = np.ones((4, 4), dtype=np.int32) * 128
        yld = mapper.monte_carlo_yield(w, n_trials=20, tolerance_q88=256)
        assert yld == 1.0  # very high tolerance → 100% yield
