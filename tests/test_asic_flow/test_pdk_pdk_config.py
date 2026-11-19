# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Focused PDK preset and root-binding tests

"""Verify PDK preset geometry, voltage, Liberty tokens, and root binding."""

from __future__ import annotations

from sc_neurocore.asic_flow.asic_flow import PDKConfig, PDKType


class TestPDKConfig:
    """Validate maintained process-design-kit presets and path substitution."""

    def test_sky130_preset(self) -> None:
        """Sky130 preset configures 1.8 V 130 nm nominal paths and corner token."""
        cfg = PDKConfig.from_pdk_type(PDKType.SKY130)
        assert "sky130" in cfg.liberty_file
        assert cfg.voltage_v == 1.8
        assert cfg.temperature_c == 25.0
        assert cfg.corner == "tt"
        assert cfg.min_feature_nm == 130
        assert cfg.nominal_liberty_suffix == "_tt_025C_1v80"

    def test_gf180_preset(self) -> None:
        """GF180MCU preset configures 3.3 V 180 nm nominal paths and corner token."""
        cfg = PDKConfig.from_pdk_type(PDKType.GF180MCU)
        assert "gf180" in cfg.liberty_file
        assert cfg.tech_lef.endswith(".tlef")
        assert cfg.voltage_v == 3.3
        assert cfg.temperature_c == 25.0
        assert cfg.corner == "tt"
        assert cfg.nominal_liberty_suffix == "_tt_025C_3v30"

    def test_tsmc28_preset(self) -> None:
        """TSMC28 preset records its slow 0.81 V 125 C library without a tt token."""
        cfg = PDKConfig.from_pdk_type(PDKType.TSMC28)
        assert cfg.min_feature_nm == 28
        assert cfg.metal_layers == 10
        assert cfg.voltage_v == 0.81
        assert cfg.temperature_c == 125.0
        assert cfg.corner == "ss"
        assert cfg.nominal_liberty_suffix == ""

    def test_intel16_preset(self) -> None:
        """Intel16 preset configures 16 nm geometry without a corner infix."""
        cfg = PDKConfig.from_pdk_type(PDKType.INTEL16)
        assert cfg.min_feature_nm == 16
        assert cfg.nominal_liberty_suffix == ""

    def test_custom_preset(self) -> None:
        """Custom preset starts with empty file paths and empty nominal token."""
        cfg = PDKConfig.from_pdk_type(PDKType.CUSTOM)
        assert cfg.liberty_file == ""
        assert cfg.nominal_liberty_suffix == ""

    def test_is_open_source(self) -> None:
        """Only Sky130 and GF180MCU report maintained open-source file maps."""
        assert PDKConfig.from_pdk_type(PDKType.SKY130).is_open_source is True
        assert PDKConfig.from_pdk_type(PDKType.TSMC28).is_open_source is False

    def test_all_pdks(self) -> None:
        """Every declared PDK enum member builds a positive feature size."""
        for pdk in PDKType:
            cfg = PDKConfig.from_pdk_type(pdk)
            assert cfg.min_feature_nm > 0

    def test_bind_pdk_root(self) -> None:
        """Binding PDK_ROOT expands template paths and preserves corner tokens."""
        cfg = PDKConfig.from_pdk_type(PDKType.GF180MCU).with_pdk_root("/opt/pdk")
        assert cfg.liberty_file.startswith("/opt/pdk/gf180mcuD")
        assert "$PDK_ROOT" not in cfg.lef_file
        assert cfg.nominal_liberty_suffix == "_tt_025C_3v30"
