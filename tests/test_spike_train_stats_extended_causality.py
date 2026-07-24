# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCausality from former test_spike_train_stats_extended.py

"""Focused suite: TestCausality from former test_spike_train_stats_extended.py."""

from __future__ import annotations

from tests.spike_train_stats_extended_support import *  # noqa: F403


class TestCausality:
    def test_pairwise_granger(self, two_trains):
        a, b = two_trains
        gc = pairwise_granger_causality(a, b)
        assert np.isfinite(gc)

    def test_conditional_granger(self, two_trains):
        a, b = two_trains
        c = np.zeros_like(a)
        c[::50] = 1
        gc = conditional_granger_causality(a, b, c)
        assert np.isfinite(gc)

    def test_spectral_granger(self, population):
        gc = spectral_granger_causality(population[:3])
        assert gc.shape[0] == 3
        assert gc.shape[1] == 3

    def test_partial_directed_coherence(self, population):
        pdc = partial_directed_coherence(population[:3])
        assert pdc.shape[0] == 3
        assert np.all(pdc >= 0)

    def test_directed_transfer_function(self, population):
        dtf = directed_transfer_function(population[:3])
        assert dtf.shape[0] == 3
        assert np.all(dtf >= 0)
