# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPhaseLockingValue from former test_spike_train_stats.py

"""Focused suite: TestPhaseLockingValue from former test_spike_train_stats.py."""

from __future__ import annotations

from tests.spike_train_stats_support import *  # noqa: F403


class TestPhaseLockingValue:
    def test_locked(self):
        lfp = np.sin(2 * np.pi * 10 * np.arange(10000) * 0.001)
        train = np.zeros(10000, dtype=np.uint8)
        peaks = np.where(lfp > 0.99)[0]
        train[peaks] = 1
        plv = phase_locking_value(train, lfp)
        assert plv > 0.5

    def test_random_low(self):
        lfp = np.sin(2 * np.pi * 10 * np.arange(5000) * 0.001)
        train = _poisson_train(50.0, 5.0)[:5000]
        plv = phase_locking_value(train, lfp)
        assert plv < 0.5
