# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBrainScaleSAnalysis from former test_model_brainscales_adex.py

"""Focused suite: TestBrainScaleSAnalysis from former test_model_brainscales_adex.py."""

from __future__ import annotations

from tests.model_brainscales_adex_support import *  # noqa: F403

class TestBrainScaleSAnalysis:
    def _get_binary_train(self):
        n = BrainScaleSAdExNeuron()
        train = np.zeros(10_000, dtype=np.int8)
        for t in range(10_000):
            train[t] = n.step(25.0)
        return train

    def test_firing_rate(self):
        train = self._get_binary_train()
        rate = firing_rate(train, dt=0.0001)
        assert rate >= 0

    def test_spike_count(self):
        train = self._get_binary_train()
        assert spike_count(train) >= 0

    def test_isi(self):
        train = self._get_binary_train()
        intervals = isi(train, dt=0.0001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))
