# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPDKValidation from former test_pdk.py

"""Focused suite: TestPDKValidation from former test_pdk.py."""

from __future__ import annotations

from tests.test_asic_flow.pdk_support import *  # noqa: F403


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
