# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDVFS from former test_intelligence_power_and_thermal.py

"""Focused suite: TestDVFS from former test_intelligence_power_and_thermal.py."""

from __future__ import annotations

from tests.intelligence_power_and_thermal_support import *  # noqa: F403


class TestDVFS(unittest.TestCase):
    def test_default(self):
        v = generate_dvfs_controller("sc_lif")
        self.assertIn("module sc_lif_dvfs_ctrl", v)
        self.assertIn("spike_rate", v)
        self.assertIn("OP_0", v)
        self.assertIn("endmodule", v)

    def test_custom_points(self):
        v = generate_dvfs_controller(
            "sc_hh",
            operating_points=[
                {"voltage_mv": 600, "freq_mhz": 50},
                {"voltage_mv": 1200, "freq_mhz": 800},
            ],
        )
        self.assertIn("50", v)
        self.assertIn("800", v)
