# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestOnlineLearner from former test_bci_studio.py

"""Focused suite: TestOnlineLearner from former test_bci_studio.py."""

from __future__ import annotations

from bci_studio_support import *  # noqa: F403


class TestOnlineLearner(unittest.TestCase):
    def setUp(self):
        self.learner = OnlineLearner(num_weights=64, lr=0.1)

    def test_initial_weights(self):
        np.testing.assert_array_equal(self.learner.weights, np.ones(64, dtype=np.float32))

    def test_positive_reward_potentiates(self):
        spikes = np.zeros(64, dtype=np.uint8)
        spikes[:10] = 1
        old_w = self.learner.weights[0]
        self.learner.step(spikes, reward=1.0)
        # Spiking channels should be potentiated (after decay)
        self.assertGreater(self.learner.weights[0], old_w * 0.9)

    def test_negative_reward_depresses(self):
        spikes = np.zeros(64, dtype=np.uint8)
        spikes[:10] = 1
        self.learner.step(spikes, reward=-1.0)
        # Spiking channels with negative reward get depressed
        self.assertLess(self.learner.weights[0], 1.0)

    def test_weights_clipped(self):
        for _ in range(100):
            spikes = np.ones(64, dtype=np.uint8)
            self.learner.step(spikes, reward=1.0)
        self.assertTrue(np.all(self.learner.weights <= 10.0))
        self.assertTrue(np.all(self.learner.weights >= 0.01))

    def test_update_counter(self):
        spikes = np.zeros(64, dtype=np.uint8)
        self.learner.step(spikes, reward=0.0)
        self.assertEqual(self.learner.updates, 1)
