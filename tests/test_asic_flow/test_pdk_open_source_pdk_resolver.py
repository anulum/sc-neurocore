# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestOpenSourcePDKResolver from former test_pdk.py

"""Focused suite: TestOpenSourcePDKResolver from former test_pdk.py."""

from __future__ import annotations

from tests.test_asic_flow.pdk_support import *  # noqa: F403


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
