# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHardwareCosts from former test_energy_accounting.py

"""Focused suite: TestHardwareCosts from former test_energy_accounting.py."""

from __future__ import annotations

from tests.energy_accounting_support import *  # noqa: F403


class TestHardwareCosts:
    def test_builtin_targets(self):
        assert len(HARDWARE_COSTS) >= 5
        assert "loihi2" in HARDWARE_COSTS
        assert "akida" in HARDWARE_COSTS
        assert "analog_28nm" in HARDWARE_COSTS
