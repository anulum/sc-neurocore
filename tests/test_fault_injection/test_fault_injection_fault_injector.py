# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFaultInjector from former test_fault_injection.py

"""Focused suite: TestFaultInjector from former test_fault_injection.py."""

from __future__ import annotations

from fault_injection_support import *  # noqa: F403

class TestFaultInjector(unittest.TestCase):
    def setUp(self):
        self.injector = FaultInjector(seed=42)

    def test_bit_flip_at_high_ber(self):
        bs = np.ones(1000, dtype=np.uint8)
        corrupted, flipped = self.injector.inject(bs, FaultModel.BIT_FLIP, ber=0.5)
        self.assertGreater(flipped, 100)
        self.assertLess(flipped, 900)

    def test_bit_flip_at_zero_ber(self):
        bs = np.ones(100, dtype=np.uint8)
        corrupted, flipped = self.injector.inject(bs, FaultModel.BIT_FLIP, ber=0.0)
        self.assertEqual(flipped, 0)
        np.testing.assert_array_equal(bs, corrupted)

    def test_stuck_at_0(self):
        bs = np.ones(1000, dtype=np.uint8)
        corrupted, affected = self.injector.inject(bs, FaultModel.STUCK_AT_0, ber=0.1)
        self.assertGreater(affected, 0)
        self.assertEqual(int(np.sum(corrupted)), 1000 - affected)

    def test_stuck_at_1(self):
        bs = np.zeros(1000, dtype=np.uint8)
        corrupted, affected = self.injector.inject(bs, FaultModel.STUCK_AT_1, ber=0.1)
        self.assertGreater(affected, 0)
        self.assertEqual(int(np.sum(corrupted)), affected)

    def test_gaussian_noise(self):
        bs = np.ones(1000, dtype=np.uint8)
        corrupted, changed = self.injector.inject(bs, FaultModel.GAUSSIAN_NOISE, ber=0.3)
        self.assertGreater(changed, 0)

    def test_dropout(self):
        bs = np.ones(1000, dtype=np.uint8)
        corrupted, affected = self.injector.inject(bs, FaultModel.DROPOUT, ber=0.2)
        self.assertGreater(affected, 0)
        self.assertLess(int(np.sum(corrupted)), 1000)

    def test_deterministic_injection(self):
        bs = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        corrupted = self.injector.inject_at_positions(bs, [0, 4])
        expected = np.array([0, 0, 1, 1, 1], dtype=np.uint8)
        np.testing.assert_array_equal(corrupted, expected)

    def test_inject_preserves_length(self):
        bs = np.ones(512, dtype=np.uint8)
        corrupted, _ = self.injector.inject(bs, FaultModel.BIT_FLIP, ber=0.01)
        self.assertEqual(len(corrupted), 512)
