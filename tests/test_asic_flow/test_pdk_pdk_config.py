# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPDKConfig from former test_pdk.py

"""Focused suite: TestPDKConfig from former test_pdk.py."""

from __future__ import annotations

from tests.test_asic_flow.pdk_support import *  # noqa: F403


class TestPDKConfig:
    def test_sky130_preset(self) -> None:
        cfg = PDKConfig.from_pdk_type(PDKType.SKY130)
        assert "sky130" in cfg.liberty_file
        assert cfg.voltage_v == 1.8
        assert cfg.min_feature_nm == 130

    def test_gf180_preset(self) -> None:
        cfg = PDKConfig.from_pdk_type(PDKType.GF180MCU)
        assert "gf180" in cfg.liberty_file
        assert cfg.tech_lef.endswith(".tlef")
        assert cfg.voltage_v == 3.3

    def test_tsmc28_preset(self) -> None:
        cfg = PDKConfig.from_pdk_type(PDKType.TSMC28)
        assert cfg.min_feature_nm == 28
        assert cfg.metal_layers == 10

    def test_intel16_preset(self) -> None:
        cfg = PDKConfig.from_pdk_type(PDKType.INTEL16)
        assert cfg.min_feature_nm == 16

    def test_custom_preset(self) -> None:
        cfg = PDKConfig.from_pdk_type(PDKType.CUSTOM)
        assert cfg.liberty_file == ""

    def test_is_open_source(self) -> None:
        assert PDKConfig.from_pdk_type(PDKType.SKY130).is_open_source is True
        assert PDKConfig.from_pdk_type(PDKType.TSMC28).is_open_source is False

    def test_all_pdks(self) -> None:
        for pdk in PDKType:
            cfg = PDKConfig.from_pdk_type(pdk)
            assert cfg.min_feature_nm > 0

    def test_bind_pdk_root(self) -> None:
        cfg = PDKConfig.from_pdk_type(PDKType.SKY130).with_pdk_root("/opt/pdk")
        assert cfg.liberty_file.startswith("/opt/pdk/sky130A")
        assert "$PDK_ROOT" not in cfg.lef_file
