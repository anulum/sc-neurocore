# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPowerIntent from former test_intelligence_power_and_thermal.py

"""Focused suite: TestPowerIntent from former test_intelligence_power_and_thermal.py."""

from __future__ import annotations

from tests.intelligence_power_and_thermal_support import *  # noqa: F403


class TestPowerIntent:
    def test_upf_output(self):
        from sc_neurocore.compiler.intelligence import generate_power_intent

        upf = generate_power_intent("sc_lif")
        assert "set_scope sc_lif" in upf
        assert "PD_NEURON_0" in upf
        assert "set_isolation" in upf

    def test_num_domains(self):
        from sc_neurocore.compiler.intelligence import generate_power_intent

        upf = generate_power_intent("sc_lif", num_domains=4)
        assert "PD_NEURON_3" in upf
