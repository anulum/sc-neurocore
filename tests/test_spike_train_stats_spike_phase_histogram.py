# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikePhaseHistogram from former test_spike_train_stats.py

"""Focused suite: TestSpikePhaseHistogram from former test_spike_train_stats.py."""

from __future__ import annotations

from tests.spike_train_stats_support import *  # noqa: F403


class TestSpikePhaseHistogram:
    def test_shape(self):
        lfp = np.sin(2 * np.pi * 10 * np.arange(5000) * 0.001)
        train = _poisson_train(100.0, 5.0)[:5000]
        hist, centers = spike_phase_histogram(train, lfp, n_bins=18)
        assert hist.size == 18
        assert centers.size == 18
