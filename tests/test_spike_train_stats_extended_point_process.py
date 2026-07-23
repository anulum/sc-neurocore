# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPointProcess from former test_spike_train_stats_extended.py

"""Focused suite: TestPointProcess from former test_spike_train_stats_extended.py."""

from __future__ import annotations

from tests.spike_train_stats_extended_support import *  # noqa: F403

class TestPointProcess:
    def test_conditional_intensity(self, poisson_train):
        ci = conditional_intensity(poisson_train)
        assert ci.size == poisson_train.size
        assert np.all(ci >= 0)

    def test_isi_hazard_function(self, poisson_train):
        h, centers = isi_hazard_function(poisson_train)
        assert h.size == centers.size
        assert h.size > 0

    def test_isi_survivor_function(self, poisson_train):
        s, centers = isi_survivor_function(poisson_train)
        assert s.size > 0
        assert s[0] >= s[-1]

    def test_renewal_density(self, poisson_train):
        d, centers = renewal_density(poisson_train)
        assert d.size == centers.size
