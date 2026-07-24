# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDeterminism from former test_cortical_column.py

"""Focused suite: TestDeterminism from former test_cortical_column.py."""

from __future__ import annotations

from tests.cortical_column_support import *  # noqa: F403


class TestDeterminism:
    def test_same_seed_same_state(self):
        a = CorticalColumn(scale=0.02, scale_correction=False, delay_distribution=False, seed=99)
        b = CorticalColumn(scale=0.02, scale_correction=False, delay_distribution=False, seed=99)
        for p in POPULATIONS:
            np.testing.assert_array_equal(a.v[p], b.v[p])

    def test_same_seed_same_rasters(self):
        a = CorticalColumn(scale=0.02, scale_correction=False, delay_distribution=False, seed=7)
        b = CorticalColumn(scale=0.02, scale_correction=False, delay_distribution=False, seed=7)
        ra = a.simulate(duration_ms=20.0, dt=0.1)
        rb = b.simulate(duration_ms=20.0, dt=0.1)
        for p in POPULATIONS:
            np.testing.assert_array_equal(ra[p], rb[p])

    def test_different_seed_different_state(self):
        a = CorticalColumn(scale=0.02, scale_correction=False, delay_distribution=False, seed=1)
        b = CorticalColumn(scale=0.02, scale_correction=False, delay_distribution=False, seed=2)
        # At least one population must have different initial voltages.
        differs = any(not np.array_equal(a.v[p], b.v[p]) for p in POPULATIONS)
        assert differs

    def test_global_numpy_seed_does_not_leak(self):
        np.random.seed(0)
        a = CorticalColumn(scale=0.02, scale_correction=False, delay_distribution=False, seed=42)
        ra = a.simulate(duration_ms=10.0, dt=0.1)
        np.random.seed(99999)
        b = CorticalColumn(scale=0.02, scale_correction=False, delay_distribution=False, seed=42)
        rb = b.simulate(duration_ms=10.0, dt=0.1)
        for p in POPULATIONS:
            np.testing.assert_array_equal(ra[p], rb[p])
