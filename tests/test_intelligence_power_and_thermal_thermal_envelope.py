# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestThermalEnvelope from former test_intelligence_power_and_thermal.py

"""Focused suite: TestThermalEnvelope from former test_intelligence_power_and_thermal.py."""

from __future__ import annotations

from tests.intelligence_power_and_thermal_support import *  # noqa: F403


class TestThermalEnvelope:
    def test_pass(self):
        from sc_neurocore.compiler.intelligence import (
            estimate_thermal_envelope,
        )

        t = estimate_thermal_envelope(power_mw=100, theta_ja=25)
        assert t.pass_fail == "PASS"
        assert t.t_junction == 27.5  # 25 + 0.1*25

    def test_fail(self):
        from sc_neurocore.compiler.intelligence import (
            estimate_thermal_envelope,
        )

        t = estimate_thermal_envelope(
            power_mw=5000,
            theta_ja=30,
            t_junction_max=100,
        )
        assert t.pass_fail == "FAIL"
        assert t.thermal_margin < 0

    def test_margin(self):
        from sc_neurocore.compiler.intelligence import (
            estimate_thermal_envelope,
        )

        t = estimate_thermal_envelope(power_mw=0)
        assert t.thermal_margin == 100.0  # 125 - 25
