# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSurrogateTrialShuffle from former test_spike_train_stats.py

"""Focused suite: TestSurrogateTrialShuffle from former test_spike_train_stats.py."""

from __future__ import annotations

from tests.spike_train_stats_support import *  # noqa: F403


class TestSurrogateTrialShuffle:
    def test_preserves_trials(self):
        trains = [_poisson_train(50.0, 0.2, seed=i) for i in range(5)]
        shuffled = surrogate_trial_shuffle(trains, seed=1)
        assert len(shuffled) == 5
        sums_orig = sorted(t.sum() for t in trains)
        sums_shuf = sorted(t.sum() for t in shuffled)
        assert sums_orig == sums_shuf
