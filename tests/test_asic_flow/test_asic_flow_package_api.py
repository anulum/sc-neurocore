# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ASIC flow package API tests

"""Package-facade tests for the ASIC flow public API."""

from __future__ import annotations

import json
from pathlib import Path

import sc_neurocore.asic_flow as asic_flow
from sc_neurocore.asic_flow.asic_flow import ASICFlowBundle, generate_asic_flow_bundle


def test_asic_flow_package_exports_bundle_api() -> None:
    """Package entry point re-exports the one-command ASIC bundle API."""
    assert asic_flow.__tier__ == "industrial"
    assert asic_flow.__all__ == ["ASICFlowBundle", "generate_asic_flow_bundle"]
    assert asic_flow.ASICFlowBundle is ASICFlowBundle
    assert asic_flow.generate_asic_flow_bundle is generate_asic_flow_bundle


def test_asic_flow_package_import_generates_evidence_manifest(tmp_path: Path) -> None:
    """Package-level bundle generation writes the evidence manifest."""
    bundle = asic_flow.generate_asic_flow_bundle(
        tmp_path / "sky130_demo",
        pdk_type="sky130",
        pdk_root="/opt/pdks",
    )
    manifest = json.loads(Path(bundle.manifest_path).read_text(encoding="utf-8"))

    assert isinstance(bundle, asic_flow.ASICFlowBundle)
    assert manifest["schema"] == "sc-neurocore.asic_flow_manifest.v1"
    assert manifest["claim_status"]["scripts_generated"] is True
    assert manifest["claim_status"]["external_eda_executed"] is False
    assert manifest["claim_status"]["physical_ppa_claim_allowed"] is False
    assert set(bundle.file_paths) == {
        "Makefile",
        "constraints.sdc",
        "drc_check.py",
        "floorplan.tcl",
        "gdsii_export.sh",
        "lvs_check.sh",
        "pnr.tcl",
        "sta.tcl",
        "synth.tcl",
    }
