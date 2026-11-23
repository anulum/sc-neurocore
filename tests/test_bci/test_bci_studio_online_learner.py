# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestOnlineLearner from former test_bci_studio.py

"""Focused suite: TestOnlineLearner from former test_bci_studio.py."""

from __future__ import annotations

import numpy as np
import unittest
from sc_neurocore.bci_studio.bci_studio import (
    OnlineLearner,
)


class TestOnlineLearner(unittest.TestCase):
    """Reward-modulated weight updates of the online learner."""

    def setUp(self) -> None:
        """Create a 64-weight online learner."""
        self.learner = OnlineLearner(num_weights=64, lr=0.1)

    def test_initial_weights(self) -> None:
        """All weights start at one."""
        np.testing.assert_array_equal(self.learner.weights, np.ones(64, dtype=np.float32))

    def test_positive_reward_potentiates(self) -> None:
        """A positive reward keeps an active weight above nine tenths of its value."""
        spikes = np.zeros(64, dtype=np.uint8)
        spikes[:10] = 1
        old_w = self.learner.weights[0]
        self.learner.step(spikes, reward=1.0)
        # Spiking channels should be potentiated (after decay)
        self.assertGreater(self.learner.weights[0], old_w * 0.9)

    def test_negative_reward_depresses(self) -> None:
        """A negative reward lowers an active weight below one."""
        spikes = np.zeros(64, dtype=np.uint8)
        spikes[:10] = 1
        self.learner.step(spikes, reward=-1.0)
        # Spiking channels with negative reward get depressed
        self.assertLess(self.learner.weights[0], 1.0)

    def test_weights_clipped(self) -> None:
        """Repeated reward keeps every weight between its bounds."""
        for _ in range(100):
            spikes = np.ones(64, dtype=np.uint8)
            self.learner.step(spikes, reward=1.0)
        self.assertTrue(np.all(self.learner.weights <= 10.0))
        self.assertTrue(np.all(self.learner.weights >= 0.01))

    def test_update_counter(self) -> None:
        """Each step counts as one update."""
        spikes = np.zeros(64, dtype=np.uint8)
        self.learner.step(spikes, reward=0.0)
        self.assertEqual(self.learner.updates, 1)
