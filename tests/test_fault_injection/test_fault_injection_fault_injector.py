# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFaultInjector from former test_fault_injection.py

"""Focused suite: TestFaultInjector from former test_fault_injection.py."""

from __future__ import annotations

import numpy as np
import unittest
from sc_neurocore.fault_injection.fault_injection import (
    FaultInjector,
    FaultModel,
)


class TestFaultInjector(unittest.TestCase):
    """Each fault model of the fault injector on bitstreams."""

    def setUp(self) -> None:
        """Create a seeded fault injector."""
        self.injector = FaultInjector(seed=42)

    def test_bit_flip_at_high_ber(self) -> None:
        """A bit-error rate of one half flips a middling share of the bits."""
        bs = np.ones(1000, dtype=np.uint8)
        corrupted, flipped = self.injector.inject(bs, FaultModel.BIT_FLIP, ber=0.5)
        self.assertGreater(flipped, 100)
        self.assertLess(flipped, 900)

    def test_bit_flip_at_zero_ber(self) -> None:
        """A bit-error rate of zero flips nothing and leaves the stream unchanged."""
        bs = np.ones(100, dtype=np.uint8)
        corrupted, flipped = self.injector.inject(bs, FaultModel.BIT_FLIP, ber=0.0)
        self.assertEqual(flipped, 0)
        np.testing.assert_array_equal(bs, corrupted)

    def test_stuck_at_0(self) -> None:
        """Stuck-at-zero faults clear exactly the bits they affect."""
        bs = np.ones(1000, dtype=np.uint8)
        corrupted, affected = self.injector.inject(bs, FaultModel.STUCK_AT_0, ber=0.1)
        self.assertGreater(affected, 0)
        self.assertEqual(int(np.sum(corrupted)), 1000 - affected)

    def test_stuck_at_1(self) -> None:
        """Stuck-at-one faults set exactly the bits they affect."""
        bs = np.zeros(1000, dtype=np.uint8)
        corrupted, affected = self.injector.inject(bs, FaultModel.STUCK_AT_1, ber=0.1)
        self.assertGreater(affected, 0)
        self.assertEqual(int(np.sum(corrupted)), affected)

    def test_gaussian_noise(self) -> None:
        """Gaussian noise changes some bits of the stream."""
        bs = np.ones(1000, dtype=np.uint8)
        corrupted, changed = self.injector.inject(bs, FaultModel.GAUSSIAN_NOISE, ber=0.3)
        self.assertGreater(changed, 0)

    def test_dropout(self) -> None:
        """Dropout clears some bits of an all-ones stream."""
        bs = np.ones(1000, dtype=np.uint8)
        corrupted, affected = self.injector.inject(bs, FaultModel.DROPOUT, ber=0.2)
        self.assertGreater(affected, 0)
        self.assertLess(int(np.sum(corrupted)), 1000)

    def test_deterministic_injection(self) -> None:
        """Injecting at given positions flips exactly those bits."""
        bs = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        corrupted = self.injector.inject_at_positions(bs, [0, 4])
        expected = np.array([0, 0, 1, 1, 1], dtype=np.uint8)
        np.testing.assert_array_equal(corrupted, expected)

    def test_inject_preserves_length(self) -> None:
        """Injection keeps the stream length."""
        bs = np.ones(512, dtype=np.uint8)
        corrupted, _ = self.injector.inject(bs, FaultModel.BIT_FLIP, ber=0.01)
        self.assertEqual(len(corrupted), 512)
