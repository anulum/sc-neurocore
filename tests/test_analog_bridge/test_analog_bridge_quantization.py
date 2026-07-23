# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQuantization from former test_analog_bridge.py

"""Focused suite: TestQuantization from former test_analog_bridge.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent))
from analog_bridge_support import *  # noqa: F403

class TestQuantization(unittest.TestCase):
    """DAC quantization checks for conductance ranges."""

    def setUp(self) -> None:
        """Create a bridge with a ten-bit DAC range."""
        self.bridge = AnalogBridge(g_range=(0.0, 50.0), v_range=(-80.0, -40.0), dac_res=10)

    def test_quantize_min(self) -> None:
        """The range minimum maps to DAC code zero."""
        dac, actual = self.bridge._quantize(0.0, 0.0, 50.0)
        self.assertEqual(dac, 0)
        self.assertAlmostEqual(actual, 0.0, places=2)

    def test_quantize_max(self) -> None:
        """The range maximum maps to the largest DAC code."""
        dac, actual = self.bridge._quantize(50.0, 0.0, 50.0)
        self.assertEqual(dac, 1023)
        self.assertAlmostEqual(actual, 50.0, places=2)

    def test_quantize_midpoint(self) -> None:
        """Mid-range values round back to the expected analog value."""
        dac, actual = self.bridge._quantize(25.0, 0.0, 50.0)
        self.assertAlmostEqual(actual, 25.0, delta=0.1)

    def test_quantize_clamp_below(self) -> None:
        """Values below the range clamp to DAC code zero."""
        dac, actual = self.bridge._quantize(-10.0, 0.0, 50.0)
        self.assertEqual(dac, 0)

    def test_quantize_clamp_above(self) -> None:
        """Values above the range clamp to the largest DAC code."""
        dac, actual = self.bridge._quantize(100.0, 0.0, 50.0)
        self.assertEqual(dac, 1023)
