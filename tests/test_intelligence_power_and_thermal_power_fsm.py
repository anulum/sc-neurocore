# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPowerFSM from former test_intelligence_power_and_thermal.py

"""Focused suite: TestPowerFSM from former test_intelligence_power_and_thermal.py."""

from __future__ import annotations

from tests.intelligence_power_and_thermal_support import *  # noqa: F403

class TestPowerFSM:
    def test_default(self):
        from sc_neurocore.compiler.intelligence import generate_power_state_machine

        v = generate_power_state_machine("sc_lif")
        assert "ACTIVE" in v
        assert "HIBERNATE" in v
        assert "power_fsm" in v

    def test_custom_states(self):
        from sc_neurocore.compiler.intelligence import generate_power_state_machine

        v = generate_power_state_machine("sc_lif", states=["ON", "OFF"])
        assert "ON" in v
        assert "OFF" in v
