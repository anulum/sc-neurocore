# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRewardModulatedLearner from former test_learning_advanced.py

"""Focused suite: TestRewardModulatedLearner from former test_learning_advanced.py."""

from __future__ import annotations

from tests.learning_advanced_support import *  # noqa: F403

class TestRewardModulatedLearner:
    def test_positive_reward_does_not_crash(self):
        """R-STDP step with positive reward should execute without error.
        Weight changes depend on spike coincidence detection (voltage > 0.9)
        which may not trigger with default LIF parameters."""
        net, proj = _make_small_network()
        rstdp = RewardModulatedLearner(net, tau_reward=100.0)
        net.run(duration=0.05, dt=0.001)
        rstdp.step(reward=10.0)  # should not raise

    def test_zero_reward_minimal_change(self):
        net, proj = _make_small_network()
        rstdp = RewardModulatedLearner(net, tau_reward=100.0)
        w_before = proj.data.copy()
        net.run(duration=0.01, dt=0.001)
        rstdp.step(reward=0.0)
        dw = proj.data - w_before
        np.testing.assert_allclose(dw, 0.0, atol=1e-10)
