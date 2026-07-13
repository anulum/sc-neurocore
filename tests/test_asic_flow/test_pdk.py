# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ASIC PDK resolution and validation tests

"""Exercise PDK presets, filesystem resolution, and validation contracts."""

from __future__ import annotations

from pathlib import Path

from sc_neurocore.asic_flow.asic_flow import (
    OpenSourcePDKResolver,
    PDKConfig,
    PDKResolution,
    PDKType,
    ResolvedPDKFiles,
    validate_pdk,
    validate_pdk_installation,
)


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


class TestOpenSourcePDKResolver:
    def test_resolves_sky130_manifest(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        resolution = OpenSourcePDKResolver.resolve(pdk, pdk_root="/opt/pdk")
        assert isinstance(resolution, PDKResolution)
        assert isinstance(resolution.files, ResolvedPDKFiles)
        assert resolution.pdk.liberty_file.startswith("/opt/pdk/sky130A")
        assert "sky130.lydrc" in resolution.files.drc_deck

    def test_resolves_gf180_manifest(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.GF180MCU)
        resolution = OpenSourcePDKResolver.resolve(pdk, pdk_root="/opt/pdk")
        assert resolution.pdk.tech_lef.endswith(".tlef")
        assert "gf180mcuD_setup.tcl" in resolution.files.lvs_setup

    def test_reports_missing_required_files(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        resolution = OpenSourcePDKResolver.resolve(
            pdk, pdk_root="/definitely/missing/pdk", require_existing=True
        )
        assert not resolution.usable_for_synthesis
        assert "liberty_file" in resolution.missing_required

    def test_accepts_minimal_existing_synthesis_files(self, tmp_path: Path) -> None:
        root = tmp_path / "pdk"
        paths = [
            root / "sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib",
            root / "sky130A/libs.ref/sky130_fd_sc_hd/lef/sky130_fd_sc_hd.lef",
            root / "sky130A/libs.ref/sky130_fd_sc_hd/techlef/sky130_fd_sc_hd__nom.tlef",
        ]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test-pdk-file\n", encoding="utf-8")

        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        resolution = OpenSourcePDKResolver.resolve(pdk, pdk_root=str(root), require_existing=True)
        assert resolution.usable_for_synthesis
        assert not resolution.usable_for_signoff

    def test_vendor_process_resolves_without_open_source_signoff_decks(self) -> None:
        """Commercial file inputs remain intact and gain no invented deck paths."""
        pdk = PDKConfig(
            pdk_type=PDKType.TSMC28,
            liberty_file="lib/tsmc28.lib",
            lef_file="lef/tsmc28.lef",
            tech_lef="lef/tsmc28.tech.lef",
        )

        resolution = OpenSourcePDKResolver.resolve(pdk)

        assert resolution.files.required_paths() == {
            "liberty_file": "lib/tsmc28.lib",
            "lef_file": "lef/tsmc28.lef",
            "tech_lef": "lef/tsmc28.tech.lef",
        }
        assert resolution.files.optional_paths() == {
            "setup_tcl": "",
            "drc_deck": "",
            "lvs_setup": "",
        }


class TestPDKValidation:
    def test_valid_sky130(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        result = validate_pdk(pdk)
        assert result.valid

    def test_invalid_broken_pdk(self) -> None:
        pdk = PDKConfig(pdk_type=PDKType.SKY130, liberty_file="", lef_file="", voltage_v=0.0)
        result = validate_pdk(pdk)
        assert not result.valid
        assert len(result.errors) >= 2

    def test_custom_pdk_no_file_check(self) -> None:
        pdk = PDKConfig(
            pdk_type=PDKType.CUSTOM, liberty_file="", voltage_v=1.8, clock_period_ns=10.0
        )
        result = validate_pdk(pdk)
        assert result.valid

    def test_installation_check_reports_missing_pdk(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        result = validate_pdk_installation(pdk, pdk_root="/definitely/missing/pdk")
        assert not result.valid
        assert any("liberty_file not found" in err for err in result.errors)

    def test_installation_check_can_require_signoff_files(self, tmp_path: Path) -> None:
        root = tmp_path / "pdk"
        paths = [
            root / "sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib",
            root / "sky130A/libs.ref/sky130_fd_sc_hd/lef/sky130_fd_sc_hd.lef",
            root / "sky130A/libs.ref/sky130_fd_sc_hd/techlef/sky130_fd_sc_hd__nom.tlef",
        ]
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test-pdk-file\n", encoding="utf-8")

        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        synth_only = validate_pdk_installation(pdk, pdk_root=str(root))
        signoff = validate_pdk_installation(pdk, pdk_root=str(root), require_signoff=True)
        assert synth_only.valid
        assert synth_only.warnings
        assert not signoff.valid

    def test_validation_reports_all_invalid_physical_fields(self) -> None:
        """Missing decks and invalid physical inputs are reported together."""
        pdk = PDKConfig(
            pdk_type=PDKType.SKY130,
            liberty_file="",
            lef_file="",
            tech_lef="",
            clock_period_ns=0.0,
            voltage_v=0.0,
            metal_layers=2,
        )

        result = validate_pdk(pdk)

        assert result.errors == [
            "liberty_file is empty",
            "lef_file is empty",
            "tech_lef is empty",
            "clock_period_ns must be positive, got 0.0",
            "voltage_v must be positive, got 0.0",
        ]
        assert result.warnings == ["only 2 metal layers — may limit routing"]
