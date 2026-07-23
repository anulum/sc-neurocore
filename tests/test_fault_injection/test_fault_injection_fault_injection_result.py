# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFaultInjectionResult from former test_fault_injection.py

"""Focused suite: TestFaultInjectionResult from former test_fault_injection.py."""

from __future__ import annotations

from fault_injection_support import *  # noqa: F403

class TestFaultInjectionResult(unittest.TestCase):
    def test_probability_calculation(self):
        r = FaultInjectionResult(
            original_popcount=500,
            corrupted_popcount=480,
            bits_flipped=20,
            bitstream_length=1000,
        )
        self.assertAlmostEqual(r.probability_original, 0.5)
        self.assertAlmostEqual(r.probability_corrupted, 0.48)
        self.assertAlmostEqual(r.absolute_error, 0.02)

    def test_zero_length_safety(self):
        r = FaultInjectionResult(0, 0, 0, 0)
        self.assertEqual(r.probability_original, 0.0)
        self.assertEqual(r.absolute_error, 0.0)
