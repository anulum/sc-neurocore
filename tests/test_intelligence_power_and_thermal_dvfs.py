# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDVFS from former test_intelligence_power_and_thermal.py

"""Focused suite: TestDVFS from former test_intelligence_power_and_thermal.py."""

from __future__ import annotations

import unittest
from sc_neurocore.compiler.intelligence import (
    generate_dvfs_controller,
)


class TestDVFS(unittest.TestCase):
    """Generating a DVFS controller."""

    def test_default(self) -> None:
        """The default controller tracks the spike rate."""
        v = generate_dvfs_controller("sc_lif")
        self.assertIn("module sc_lif_dvfs_ctrl", v)
        self.assertIn("spike_rate", v)
        self.assertIn("OP_0", v)
        self.assertIn("endmodule", v)

    def test_custom_points(self) -> None:
        """Custom operating points are used."""
        v = generate_dvfs_controller(
            "sc_hh",
            operating_points=[
                {"voltage_mv": 600, "freq_mhz": 50},
                {"voltage_mv": 1200, "freq_mhz": 800},
            ],
        )
        self.assertIn("50", v)
        self.assertIn("800", v)
