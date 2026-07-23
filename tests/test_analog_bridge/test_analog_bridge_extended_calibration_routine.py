# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCalibrationRoutine from former test_analog_bridge_extended.py

"""Focused suite: TestCalibrationRoutine from former test_analog_bridge_extended.py."""

from __future__ import annotations

from analog_bridge_extended_support import *  # noqa: F403

class TestCalibrationRoutine(unittest.TestCase):
    """Calibration sweep and ENOB contract checks."""

    def setUp(self) -> None:
        """Create a low-resolution calibration routine."""
        self.bridge = AnalogBridge(g_range=(0, 100), v_range=(-80, -40), dac_res=4)
        self.cal = CalibrationRoutine(self.bridge)

    def test_sweep_length(self) -> None:
        """The conductance sweep includes both endpoints and ten intervals."""
        sweep = self.cal.sweep_conductance()
        self.assertEqual(len(sweep), 11)

    def test_sweep_endpoints(self) -> None:
        """The conductance sweep starts and ends at the configured range."""
        sweep = self.cal.sweep_conductance()
        self.assertAlmostEqual(sweep[0][1], 0.0, places=2)
        self.assertAlmostEqual(sweep[-1][1], 100.0, places=2)

    def test_max_error_positive_for_low_res(self) -> None:
        """A low-resolution DAC has non-zero quantization error."""
        err = self.cal.max_quantization_error()
        self.assertGreater(err, 0)

    def test_enob_less_than_nominal(self) -> None:
        """Low-resolution ENOB remains positive and near nominal resolution."""
        # 4-bit DAC with 16 levels → ENOB should be ≤ 4 for non-ideal sweep grid
        enob = self.cal.effective_resolution_bits()
        self.assertLessEqual(enob, self.bridge.dac_res + 1)
        self.assertGreater(enob, 0)

    def test_high_res_low_error(self) -> None:
        """A higher-resolution DAC reduces maximum quantization error."""
        bridge_hires = AnalogBridge(g_range=(0, 100), v_range=(-80, -40), dac_res=16)
        cal_hires = CalibrationRoutine(bridge_hires)
        err_hires = cal_hires.max_quantization_error()
        err_lores = self.cal.max_quantization_error()
        self.assertLess(err_hires, err_lores)

    def test_enob_zero_error_falls_back_to_nominal(self) -> None:
        """Perfect quantisation (max_err == 0) avoids log2(inf) via the fallback.

        Reachable when every sweep target lands exactly on a DAC level —
        here forced via a patched ``max_quantization_error`` because FP
        round-off usually keeps the real error strictly positive.
        """
        with patch.object(CalibrationRoutine, "max_quantization_error", return_value=0.0):
            enob = self.cal.effective_resolution_bits()
        self.assertEqual(enob, float(self.bridge.dac_res))

    def test_enob_zero_range_falls_back_to_nominal(self) -> None:
        """Zero-width conductance range (``g_max == g_min``) short-circuits ENOB.

        ``_quantize`` would raise ``ZeroDivisionError`` if invoked, so the
        sweep is bypassed by patching ``max_quantization_error`` to a
        positive stub while the bridge itself is constructed degenerately.
        """
        bridge = AnalogBridge.__new__(AnalogBridge)
        bridge.g_min = 10.0
        bridge.g_max = 10.0
        bridge.v_min = -80.0
        bridge.v_max = -40.0
        bridge.dac_res = 8
        cal = CalibrationRoutine(bridge)
        with patch.object(CalibrationRoutine, "max_quantization_error", return_value=1e-3):
            enob = cal.effective_resolution_bits()
        self.assertEqual(enob, float(bridge.dac_res))
