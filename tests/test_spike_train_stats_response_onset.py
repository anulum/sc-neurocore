# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestResponseOnset from former test_spike_train_stats.py

"""Focused suite: TestResponseOnset from former test_spike_train_stats.py."""

from __future__ import annotations

from tests.spike_train_stats_support import *  # noqa: F403


class TestResponseOnset:
    def test_detects(self):
        train = np.zeros(500, dtype=np.uint8)
        train[200:210] = 1
        onset = response_onset(train, baseline_steps=150)
        assert 0.15 < onset < 0.25

    def test_no_response(self):
        assert np.isnan(response_onset(np.zeros(200, dtype=np.uint8), baseline_steps=100))
