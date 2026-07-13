# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ASIC signoff tests

"""Exercise signoff decks, PVT corners, OCV, and summary decisions."""

from __future__ import annotations

from sc_neurocore.asic_flow.asic_flow import (
    CornerType,
    DRCViolation,
    DesignParams,
    MultiCornerAnalysis,
    OCVConfig,
    PDKConfig,
    PDKType,
    PVTCorner,
    SignoffCheckResult,
    SignoffGenerator,
    SignoffSummary,
)


class TestSignoffGenerator:
    def test_sta_script(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        script = SignoffGenerator.generate_sta_script(pdk, design)
        assert "report_checks" in script
        assert "report_tns" in script

    def test_drc_open_source(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        script = SignoffGenerator.generate_drc_script(pdk, design)
        assert "klayout" in script.lower() or "KLayout" in script

    def test_drc_commercial(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.TSMC28)
        design = DesignParams()
        script = SignoffGenerator.generate_drc_script(pdk, design)
        assert "vendor" in script.lower()

    def test_lvs_open_source(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        script = SignoffGenerator.generate_lvs_script(pdk, design)
        assert "netgen" in script.lower()

    def test_timing_pass(self) -> None:
        result = SignoffGenerator.evaluate_timing(wns=0.5, tns=0.0, clock_period_ns=10.0)
        assert result.passed is True
        assert result.check_name == "STA"

    def test_timing_fail(self) -> None:
        result = SignoffGenerator.evaluate_timing(wns=-0.3, tns=-5.0, clock_period_ns=10.0)
        assert result.passed is False

    def test_power_pass(self) -> None:
        result = SignoffGenerator.evaluate_power(3.0, 0.5, 10.0)
        assert result.passed is True

    def test_power_fail(self) -> None:
        result = SignoffGenerator.evaluate_power(8.0, 3.0, 10.0)
        assert result.passed is False

    def test_area_pass(self) -> None:
        result = SignoffGenerator.evaluate_area(5000, 150000, 250000)
        assert result.passed is True

    def test_area_fail(self) -> None:
        result = SignoffGenerator.evaluate_area(50000, 240000, 250000)
        assert result.passed is False


class TestMultiCornerAnalysis:
    def test_generates_all_corners(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        script = MultiCornerAnalysis.generate(pdk, design)
        assert "ss_125C" in script
        assert "ff_" in script

    def test_custom_corners(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        corners = [PVTCorner(CornerType.TT, 25.0, 1.8, "_tt_025C_1v80")]
        script = MultiCornerAnalysis.generate(pdk, design, corners)
        assert "tt_25C" in script

    def test_worst_slack(self) -> None:
        wns = {"tt": 0.5, "ss": -0.2, "ff": 1.0}
        corner, slack = MultiCornerAnalysis.worst_slack(wns)
        assert corner == "ss"
        assert slack == -0.2

    def test_empty_slack_map_returns_neutral_sentinel(self) -> None:
        """No reported corners produce the documented neutral sentinel."""
        assert MultiCornerAnalysis.worst_slack({}) == ("none", 0.0)

    def test_corner_without_suffix_uses_configured_liberty(self) -> None:
        """A custom corner without a suffix leaves the Liberty path unchanged."""
        pdk = PDKConfig.from_pdk_type(PDKType.CUSTOM)
        pdk.liberty_file = "custom.lib"
        corner = PVTCorner(CornerType.TT, 25.0, 1.8)

        script = MultiCornerAnalysis.generate(pdk, DesignParams(), [corner])

        assert "read_liberty custom.lib" in script

    def test_pvt_corner_label(self) -> None:
        c = PVTCorner(CornerType.SS, 125.0, 1.62)
        assert "ss" in c.label
        assert "125" in c.label


class TestOCVConfig:
    def test_default(self) -> None:
        ocv = OCVConfig()
        frag = ocv.generate_sdc_fragment()
        assert "set_timing_derate" in frag
        assert "0.950" in frag

    def test_conservative(self) -> None:
        ocv = OCVConfig.conservative()
        assert ocv.data_cell_early < 0.95
        assert ocv.data_cell_late > 1.05


class TestSignoffSummary:
    def test_all_pass(self) -> None:
        s = SignoffSummary(
            timing=SignoffCheckResult("STA", True, ""),
            power=SignoffCheckResult("Power", True, ""),
            area=SignoffCheckResult("Area", True, ""),
            lvs_match=True,
        )
        assert s.all_pass

    def test_drc_failure(self) -> None:
        s = SignoffSummary(
            timing=SignoffCheckResult("STA", True, ""),
            power=SignoffCheckResult("Power", True, ""),
            area=SignoffCheckResult("Area", True, ""),
            drc_violations=[DRCViolation("min_width", 5, "error")],
            lvs_match=True,
        )
        assert not s.drc_clean
        assert not s.all_pass

    def test_to_dict(self) -> None:
        s = SignoffSummary(
            timing=SignoffCheckResult("STA", True, "ok"),
            power=SignoffCheckResult("Power", True, "ok"),
            area=SignoffCheckResult("Area", True, "ok"),
        )
        d = s.to_dict()
        assert "timing" in d
        assert "all_pass" in d
