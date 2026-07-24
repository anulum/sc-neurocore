# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAging from former test_intelligence_verification_and_safety.py

"""Focused suite: TestAging from former test_intelligence_verification_and_safety.py."""

from __future__ import annotations

from tests.intelligence_verification_and_safety_support import *  # noqa: F403


class TestAging(unittest.TestCase):
    def test_degradation_increases(self):
        r5 = predict_aging(250.0, years=5.0, temperature_c=25.0)
        r10 = predict_aging(250.0, years=10.0, temperature_c=25.0)
        self.assertGreater(r10.degradation_pct, r5.degradation_pct)
        self.assertLess(r10.degraded_fmax_mhz, r5.degraded_fmax_mhz)

    def test_high_temp_worse(self):
        r_cool = predict_aging(250.0, temperature_c=25.0)
        r_hot = predict_aging(250.0, temperature_c=125.0)
        self.assertGreater(r_hot.degradation_pct, r_cool.degradation_pct)

    def test_dominant_mechanism(self):
        r = predict_aging(250.0)
        self.assertIn(r.dominant_mechanism, ("NBTI", "HCI"))

    def test_high_voltage_stress_can_make_hci_dominant(self):
        r = predict_aging(250.0, voltage_v=1.4, temperature_c=25.0, years=10.0)
        self.assertGreater(r.hci_degradation_pct, r.nbti_degradation_pct)
        self.assertEqual(r.dominant_mechanism, "HCI")

    def test_predict_aging_rejects_invalid_inputs(self):
        invalid_cases = [
            ({"initial_fmax_mhz": 0.0}, "initial_fmax_mhz"),
            ({"initial_fmax_mhz": 250.0, "voltage_v": 0.0}, "voltage_v"),
            ({"initial_fmax_mhz": 250.0, "temperature_c": -274.0}, "temperature_c"),
            ({"initial_fmax_mhz": 250.0, "years": -1.0}, "years"),
        ]
        for kwargs, message in invalid_cases:
            with self.assertRaisesRegex(ValueError, message):
                predict_aging(**kwargs)
