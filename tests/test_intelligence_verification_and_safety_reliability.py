# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestReliability from former test_intelligence_verification_and_safety.py

"""Focused suite: TestReliability from former test_intelligence_verification_and_safety.py."""

from __future__ import annotations

from tests.intelligence_verification_and_safety_support import *  # noqa: F403


class TestReliability(unittest.TestCase):
    def test_predict_reliability_rejects_invalid_inputs(self):
        invalid_cases = [
            ({"voltage_v": 0.0}, "voltage_v"),
            ({"temperature_c": -274.0}, "temperature_c"),
            ({"node_nm": 0}, "node_nm"),
            ({"base_mttf_hours": 0.0}, "base_mttf_hours"),
        ]
        for kwargs, message in invalid_cases:
            with self.assertRaisesRegex(ValueError, message):
                predict_reliability(**kwargs)

    def test_predict_reliability_hotter_voltage_stressed_mttf_is_lower(self):
        nominal = predict_reliability(voltage_v=0.9, temperature_c=85.0)
        stressed = predict_reliability(voltage_v=1.05, temperature_c=105.0)
        self.assertLess(stressed.mttf_hours, nominal.mttf_hours)

    def test_predict_reliability_reports_per_mechanism_mttf(self):
        stressed = predict_reliability(voltage_v=1.3, temperature_c=105.0)
        self.assertIn("NBTI", stressed.mechanism_mttf_hours)
        self.assertIn("HCI", stressed.mechanism_mttf_hours)
        self.assertIn("TDDB", stressed.mechanism_mttf_hours)
        self.assertEqual(
            stressed.failure_mode,
            min(stressed.mechanism_mttf_hours, key=stressed.mechanism_mttf_hours.get),
        )
