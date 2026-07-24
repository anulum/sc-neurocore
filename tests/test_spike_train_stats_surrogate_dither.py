# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSurrogateDither from former test_spike_train_stats.py

"""Focused suite: TestSurrogateDither from former test_spike_train_stats.py."""

from __future__ import annotations

from tests.spike_train_stats_support import *  # noqa: F403


class TestSurrogateDither:
    def test_preserves_count(self):
        train = _poisson_train(50.0, 0.5)
        surr = surrogate_dither(train, dither_ms=3.0, seed=1)
        assert abs(int(surr.sum()) - int(train.sum())) <= 5
