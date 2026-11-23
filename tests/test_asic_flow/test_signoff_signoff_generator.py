# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSignoffGenerator from former test_signoff.py

"""Focused suite: TestSignoffGenerator from former test_signoff.py."""

from __future__ import annotations

from sc_neurocore.asic_flow.asic_flow import (
    DesignParams,
    PDKConfig,
    PDKType,
    SignoffGenerator,
)


class TestSignoffGenerator:
    """Signoff scripts and pass or fail decisions."""

    def test_sta_script(self) -> None:
        """The STA script reports checks and total negative slack."""
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        script = SignoffGenerator.generate_sta_script(pdk, design)
        assert "report_checks" in script
        assert "report_tns" in script

    def test_drc_open_source(self) -> None:
        """The open-source DRC runs KLayout."""
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        script = SignoffGenerator.generate_drc_script(pdk, design)
        assert "klayout" in script.lower() or "KLayout" in script

    def test_drc_commercial(self) -> None:
        """The commercial DRC defers to the vendor flow."""
        pdk = PDKConfig.from_pdk_type(PDKType.TSMC28)
        design = DesignParams()
        script = SignoffGenerator.generate_drc_script(pdk, design)
        assert "vendor" in script.lower()

    def test_lvs_open_source(self) -> None:
        """The open-source LVS runs Netgen."""
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        script = SignoffGenerator.generate_lvs_script(pdk, design)
        assert "netgen" in script.lower()

    def test_timing_pass(self) -> None:
        """Met timing passes the STA check."""
        result = SignoffGenerator.evaluate_timing(wns=0.5, tns=0.0, clock_period_ns=10.0)
        assert result.passed is True
        assert result.check_name == "STA"

    def test_timing_fail(self) -> None:
        """Negative slack fails the STA check."""
        result = SignoffGenerator.evaluate_timing(wns=-0.3, tns=-5.0, clock_period_ns=10.0)
        assert result.passed is False

    def test_power_pass(self) -> None:
        """Power within the budget passes the power check."""
        result = SignoffGenerator.evaluate_power(3.0, 0.5, 10.0)
        assert result.passed is True

    def test_power_fail(self) -> None:
        """Power over the budget fails the power check."""
        result = SignoffGenerator.evaluate_power(8.0, 3.0, 10.0)
        assert result.passed is False

    def test_area_pass(self) -> None:
        """A utilisation of 60 percent passes the area check."""
        result = SignoffGenerator.evaluate_area(5000, 150000, 250000)
        assert result.passed is True

    def test_area_fail(self) -> None:
        """A utilisation of 96 percent, over 85, fails."""
        result = SignoffGenerator.evaluate_area(50000, 240000, 250000)
        assert result.passed is False
