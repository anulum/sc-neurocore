# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTemporalWindow from former test_explainability.py

"""Focused suite: TestTemporalWindow from former test_explainability.py."""

from __future__ import annotations

from explainability_support import *  # noqa: F403


class TestTemporalWindow:
    def test_add_and_query(self):
        tw = TemporalWindow()
        tree = SpikeDecisionTree()
        n0 = tree.add_decision("n0", np.ones(8, dtype=np.uint8), 4, timestep=0)
        n1 = tree.add_decision("n1", np.zeros(8, dtype=np.uint8), 4, timestep=1)
        tw.add(n0)
        tw.add(n1)
        assert tw.num_timesteps == 2
        assert tw.spike_rate_at(0) == 1.0
        assert tw.spike_rate_at(1) == 0.0

    def test_peak_timestep(self):
        tw = TemporalWindow()
        tree = SpikeDecisionTree()
        for t in range(3):
            for _ in range(3):
                bs = np.ones(8, dtype=np.uint8) if t == 1 else np.zeros(8, dtype=np.uint8)
                n = tree.add_decision(f"n_{t}", bs, 4, timestep=t)
                tw.add(n)
        assert tw.peak_timestep() == 1

    def test_active_timesteps(self):
        tw = TemporalWindow()
        tree = SpikeDecisionTree()
        for t in [0, 5, 10]:
            n = tree.add_decision(f"n_{t}", np.ones(8, dtype=np.uint8), 4, timestep=t)
            tw.add(n)
        assert tw.active_timesteps() == [0, 5, 10]

    def test_empty_timestep_rate_zero(self):
        tw = TemporalWindow()
        assert tw.spike_rate_at(999) == 0.0
