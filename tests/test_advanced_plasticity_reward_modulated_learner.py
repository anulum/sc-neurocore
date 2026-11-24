# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRewardModulatedLearner from former test_advanced_plasticity.py

"""Focused suite: TestRewardModulatedLearner from former test_advanced_plasticity.py."""

from __future__ import annotations

from sc_neurocore.network.projection import Projection
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
import numpy as np
from sc_neurocore.learning.advanced import (
    RewardModulatedLearner,
)


class TestRewardModulatedLearner:
    """Reward-modulated STDP learning on a projection."""

    def test_step_runs(
        self, simple_net: tuple[Network, Population, Population, Projection]
    ) -> None:
        """A learning step keeps the weight shape."""
        net, _, _, proj = simple_net
        learner = RewardModulatedLearner(net, tau_reward=50.0)
        w_before = proj.data.copy()
        learner.step(reward=1.0)
        assert proj.data is not None
        assert proj.data.shape == w_before.shape

    def test_weights_non_negative(
        self, simple_net: tuple[Network, Population, Population, Projection]
    ) -> None:
        """Learning keeps the weights non-negative."""
        net, _, _, proj = simple_net
        learner = RewardModulatedLearner(net, tau_reward=10.0)
        for _ in range(20):
            learner.step(reward=-5.0)
        assert np.all(proj.data >= 0)
