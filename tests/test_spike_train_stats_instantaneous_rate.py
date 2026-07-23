# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestInstantaneousRate from former test_spike_train_stats.py

"""Focused suite: TestInstantaneousRate from former test_spike_train_stats.py."""

from __future__ import annotations

from tests.spike_train_stats_support import *  # noqa: F403

class TestInstantaneousRate:
    def test_gaussian(self):
        train = _poisson_train(100.0, 1.0)
        rate = instantaneous_rate(train, kernel="gaussian", sigma_ms=20.0)
        assert rate.size == train.size
        assert rate.mean() > 0

    def test_exponential(self):
        rate = instantaneous_rate(_poisson_train(50.0, 0.5), kernel="exponential")
        assert rate.size > 0

    def test_rectangular(self):
        rate = instantaneous_rate(_poisson_train(50.0, 0.5), kernel="rectangular")
        assert rate.size > 0
