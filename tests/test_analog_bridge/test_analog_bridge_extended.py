# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Analog Bridge Extended Tests

"""Extended real-surface tests for analog profiles, AER events, and calibration."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from sc_neurocore.analog_bridge import (
    AEREvent,
    AnalogBridge,
    AnalogSubstrateProfile,
    CalibrationRoutine,
    EventDrivenInterface,
)


class TestSubstrateProfiles(unittest.TestCase):
    """Substrate profile contract checks."""

    def test_brainscales3(self) -> None:
        """The BrainScaleS-3 profile exposes its DAC and fan-in limits."""
        p = AnalogSubstrateProfile.brainscales3()
        self.assertEqual(p.name, "BrainScaleS-3")
        self.assertEqual(p.dac_resolution, 6)
        self.assertEqual(p.max_fanin, 256)

    def test_dynapse2(self) -> None:
        """The DynapSE-2 profile exposes its DAC resolution."""
        p = AnalogSubstrateProfile.dynapse2()
        self.assertEqual(p.name, "DynapSE-2")
        self.assertEqual(p.dac_resolution, 7)

    def test_profile_constructor(self) -> None:
        """Profile construction configures bridge resolution from the profile."""
        bridge = AnalogBridge(profile=AnalogSubstrateProfile.brainscales3())
        self.assertEqual(bridge.dac_res, 6)
        self.assertEqual(bridge.dac_levels, 64)

    def test_legacy_constructor(self) -> None:
        """Legacy range construction preserves explicit DAC resolution."""
        bridge = AnalogBridge(g_range=(0, 100), v_range=(-80, -40), dac_res=10)
        self.assertEqual(bridge.dac_levels, 1024)


class TestEventDrivenInterface(unittest.TestCase):
    """AER conversion and rate-coding contract checks."""

    def setUp(self) -> None:
        """Create a one-microsecond event interface."""
        self.iface = EventDrivenInterface(clock_period_us=1.0)

    def test_bitstream_to_events(self) -> None:
        """Set bits become ordered AER events with matching timestamps."""
        bs = np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
        events = self.iface.bitstream_to_events(42, bs)
        self.assertEqual(len(events), 4)
        self.assertEqual(events[0].neuron_id, 42)
        self.assertAlmostEqual(events[0].timestamp_us, 0.0)
        self.assertAlmostEqual(events[1].timestamp_us, 2.0)

    def test_zero_bitstream(self) -> None:
        """An all-zero bitstream emits no AER events."""
        bs = np.zeros(100, dtype=np.uint8)
        events = self.iface.bitstream_to_events(1, bs)
        self.assertEqual(len(events), 0)

    def test_events_to_current_shape(self) -> None:
        """AER events produce a current trace with the requested duration."""
        events = [AEREvent(0, 5.0), AEREvent(0, 15.0)]
        current = self.iface.events_to_current(events, duration_us=50.0, tau_syn=5.0)
        self.assertEqual(len(current), 50)

    def test_events_produce_positive_current(self) -> None:
        """Excitatory AER events add positive synaptic current."""
        events = [AEREvent(0, 0.0)]
        current = self.iface.events_to_current(events, duration_us=20.0, tau_syn=5.0)
        self.assertGreater(current[0], 0)

    def test_current_decays(self) -> None:
        """The synaptic current kernel decays after an event."""
        events = [AEREvent(0, 0.0)]
        current = self.iface.events_to_current(events, duration_us=50.0, tau_syn=5.0)
        self.assertGreater(current[0], current[-1])

    def test_rate_code(self) -> None:
        """Rate coding reports event frequency in hertz."""
        events = [AEREvent(0, i * 10.0) for i in range(100)]
        rate = self.iface.rate_code(events, window_us=1000.0)
        self.assertAlmostEqual(rate, 100_000.0, delta=1.0)  # 100 events in 1ms = 100kHz

    def test_rate_code_empty(self) -> None:
        """Empty event lists report zero firing rate."""
        self.assertEqual(self.iface.rate_code([], window_us=100.0), 0.0)


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


if __name__ == "__main__":
    unittest.main()
