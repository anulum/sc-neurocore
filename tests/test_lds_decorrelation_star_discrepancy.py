# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestStarDiscrepancy from former test_lds_decorrelation.py

"""Focused suite: TestStarDiscrepancy from former test_lds_decorrelation.py."""

from __future__ import annotations

from tests.lds_decorrelation_support import *  # noqa: F403

class TestStarDiscrepancy:
    def test_sobol_lower_discrepancy_than_random(self):
        """Sobol samples should have lower discrepancy than random."""
        sobol = qmc.Sobol(d=2, seed=42).random(256)
        rand = np.random.RandomState(42).uniform(0, 1, (256, 2))
        disc_sobol = star_discrepancy_estimate(sobol, n_test=1000)
        disc_rand = star_discrepancy_estimate(rand, n_test=1000)
        assert disc_sobol < disc_rand

    def test_discrepancy_nonnegative(self):
        samples = np.random.uniform(0, 1, (100, 3))
        disc = star_discrepancy_estimate(samples, n_test=500)
        assert disc >= 0.0
