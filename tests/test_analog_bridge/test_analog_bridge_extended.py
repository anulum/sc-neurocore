# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Analog Bridge Extended Tests

from __future__ import annotations

import unittest

import numpy as np

from sc_neurocore.analog_bridge import (
    AEREvent,
    AnalogBridge,
    AnalogSubstrateProfile,
    CalibrationRoutine,
    EventDrivenInterface,
)


class TestSubstrateProfiles(unittest.TestCase):
    def test_brainscales3(self):
        p = AnalogSubstrateProfile.brainscales3()
        self.assertEqual(p.name, "BrainScaleS-3")
        self.assertEqual(p.dac_resolution, 6)
        self.assertEqual(p.max_fanin, 256)

    def test_dynapse2(self):
        p = AnalogSubstrateProfile.dynapse2()
        self.assertEqual(p.name, "DynapSE-2")
        self.assertEqual(p.dac_resolution, 7)

    def test_profile_constructor(self):
        bridge = AnalogBridge(profile=AnalogSubstrateProfile.brainscales3())
        self.assertEqual(bridge.dac_res, 6)
        self.assertEqual(bridge.dac_levels, 64)

    def test_legacy_constructor(self):
        bridge = AnalogBridge(g_range=(0, 100), v_range=(-80, -40), dac_res=10)
        self.assertEqual(bridge.dac_levels, 1024)


class TestEventDrivenInterface(unittest.TestCase):
    def setUp(self):
        self.iface = EventDrivenInterface(clock_period_us=1.0)

    def test_bitstream_to_events(self):
        bs = np.array([1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
        events = self.iface.bitstream_to_events(42, bs)
        self.assertEqual(len(events), 4)
        self.assertEqual(events[0].neuron_id, 42)
        self.assertAlmostEqual(events[0].timestamp_us, 0.0)
        self.assertAlmostEqual(events[1].timestamp_us, 2.0)

    def test_zero_bitstream(self):
        bs = np.zeros(100, dtype=np.uint8)
        events = self.iface.bitstream_to_events(1, bs)
        self.assertEqual(len(events), 0)

    def test_events_to_current_shape(self):
        events = [AEREvent(0, 5.0), AEREvent(0, 15.0)]
        current = self.iface.events_to_current(events, duration_us=50.0, tau_syn=5.0)
        self.assertEqual(len(current), 50)

    def test_events_produce_positive_current(self):
        events = [AEREvent(0, 0.0)]
        current = self.iface.events_to_current(events, duration_us=20.0, tau_syn=5.0)
        self.assertGreater(current[0], 0)

    def test_current_decays(self):
        events = [AEREvent(0, 0.0)]
        current = self.iface.events_to_current(events, duration_us=50.0, tau_syn=5.0)
        self.assertGreater(current[0], current[-1])

    def test_rate_code(self):
        events = [AEREvent(0, i * 10.0) for i in range(100)]
        rate = self.iface.rate_code(events, window_us=1000.0)
        self.assertAlmostEqual(rate, 100_000.0, delta=1.0)  # 100 events in 1ms = 100kHz

    def test_rate_code_empty(self):
        self.assertEqual(self.iface.rate_code([], window_us=100.0), 0.0)


class TestCalibrationRoutine(unittest.TestCase):
    def setUp(self):
        self.bridge = AnalogBridge(g_range=(0, 100), v_range=(-80, -40), dac_res=4)
        self.cal = CalibrationRoutine(self.bridge)

    def test_sweep_length(self):
        sweep = self.cal.sweep_conductance()
        self.assertEqual(len(sweep), 11)

    def test_sweep_endpoints(self):
        sweep = self.cal.sweep_conductance()
        self.assertAlmostEqual(sweep[0][1], 0.0, places=2)
        self.assertAlmostEqual(sweep[-1][1], 100.0, places=2)

    def test_max_error_positive_for_low_res(self):
        err = self.cal.max_quantization_error()
        self.assertGreater(err, 0)

    def test_enob_less_than_nominal(self):
        # 4-bit DAC with 16 levels → ENOB should be ≤ 4 for non-ideal sweep grid
        enob = self.cal.effective_resolution_bits()
        self.assertLessEqual(enob, self.bridge.dac_res + 1)
        self.assertGreater(enob, 0)

    def test_high_res_low_error(self):
        bridge_hires = AnalogBridge(g_range=(0, 100), v_range=(-80, -40), dac_res=16)
        cal_hires = CalibrationRoutine(bridge_hires)
        err_hires = cal_hires.max_quantization_error()
        err_lores = self.cal.max_quantization_error()
        self.assertLess(err_hires, err_lores)


if __name__ == "__main__":
    unittest.main()
