# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestThermalEnvelope from former test_intelligence_power_and_thermal.py

"""Focused suite: TestThermalEnvelope from former test_intelligence_power_and_thermal.py."""

from __future__ import annotations


class TestThermalEnvelope:
    """Checking the thermal envelope."""

    def test_pass(self) -> None:
        """A low power passes with its junction temperature."""
        from sc_neurocore.compiler.intelligence import (
            estimate_thermal_envelope,
        )

        t = estimate_thermal_envelope(power_mw=100, theta_ja=25)
        assert t.pass_fail == "PASS"
        assert t.t_junction == 27.5  # 25 + 0.1*25

    def test_fail(self) -> None:
        """Too much power fails with a negative margin."""
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

    def test_margin(self) -> None:
        """The margin is the limit minus the ambient."""
        from sc_neurocore.compiler.intelligence import (
            estimate_thermal_envelope,
        )

        t = estimate_thermal_envelope(power_mw=0)
        assert t.thermal_margin == 100.0  # 125 - 25
