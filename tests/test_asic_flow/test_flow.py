# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ASIC bundle orchestration tests

"""Exercise complete deck bundles and evidence manifests through public APIs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sc_neurocore.asic_flow.asic_flow import (
    ASICFlowBundle,
    ASICFlowGenerator,
    ASICFlowOutput,
    DesignParams,
    PDKConfig,
    PDKType,
    generate_asic_flow_bundle,
)


class TestASICFlowGenerator:
    def test_full_flow(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(
            top_module="sc_lif_neuron",
            rtl_files=["sc_lif_neuron.v"],
        )
        gen = ASICFlowGenerator()
        output = gen.generate(pdk, design)
        assert isinstance(output, ASICFlowOutput)
        assert "synth" in output.synth_tcl.lower()
        assert "create_clock" in output.sdc
        assert "Makefile" in output.filelist

    def test_all_pdks(self) -> None:
        gen = ASICFlowGenerator()
        for pdk_type in PDKType:
            pdk = PDKConfig.from_pdk_type(pdk_type)
            design = DesignParams()
            output = gen.generate(pdk, design)
            assert len(output.filelist) > 0

    def test_makefile(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams(top_module="test_module")
        gen = ASICFlowGenerator()
        output = gen.generate(pdk, design)
        assert "test_module" in output.makefile
        assert "yosys" in output.makefile
        assert "openroad" in output.makefile

    def test_output_dict(self) -> None:
        pdk = PDKConfig.from_pdk_type(PDKType.SKY130)
        design = DesignParams()
        gen = ASICFlowGenerator()
        output = gen.generate(pdk, design)
        d = output.to_dict()
        assert "synth.tcl" in d
        assert "Makefile" in d
        assert len(d) == 9

    def test_one_command_bundle_writes_manifest(self, tmp_path: Path) -> None:
        design = DesignParams(top_module="edge_snn", rtl_files=["edge_snn.sv"])
        bundle = generate_asic_flow_bundle(
            tmp_path,
            pdk_type="sky130",
            design=design,
            pdk_root="/opt/pdks",
            n_neurons=32,
            n_synapses=512,
            bitstream_width=128,
            n_aer_ports=8,
        )

        assert isinstance(bundle, ASICFlowBundle)
        assert (tmp_path / "synth.tcl").is_file()
        assert (tmp_path / "Makefile").is_file()
        assert (tmp_path / "asic_flow_manifest.json").is_file()
        assert bundle.estimate.gate_count > 0
        manifest = (tmp_path / "asic_flow_manifest.json").read_text(encoding="utf-8")
        assert '"schema": "sc-neurocore.asic_flow_manifest.v1"' in manifest
        assert '"external_eda_executed": false' in manifest
        assert '"physical_ppa_claim_allowed": false' in manifest
        assert '"formal_evidence_attached": false' in manifest
        assert '"formal_evidence_complete_for_claim": false' in manifest
        assert "edge_snn" in manifest

    def test_one_command_bundle_reports_missing_required_pdk_files(self, tmp_path: Path) -> None:
        missing_root = tmp_path / "missing_pdk_root"
        bundle = generate_asic_flow_bundle(
            tmp_path / "out",
            pdk_type=PDKType.GF180MCU,
            pdk_root=str(missing_root),
            require_pdk_files=True,
        )

        assert bundle.pdk_resolution.usable_for_synthesis is False
        assert set(bundle.pdk_resolution.missing_required) == {
            "liberty_file",
            "lef_file",
            "tech_lef",
        }

    def test_one_command_bundle_records_formal_evidence_status(self, tmp_path: Path) -> None:
        bundle = generate_asic_flow_bundle(
            tmp_path / "out",
            pdk_type=PDKType.SKY130,
            formal_evidence_artifacts=["formal/sc_top.sby", "formal/report.json"],
        )

        manifest = json.loads(Path(bundle.manifest_path).read_text(encoding="utf-8"))
        assert manifest["formal_evidence"]["attached"] is True
        assert manifest["formal_evidence"]["complete_for_claim"] is True
        assert manifest["formal_evidence"]["artifacts"] == [
            "formal/report.json",
            "formal/sc_top.sby",
        ]

    def test_bundle_serialisation_preserves_public_paths(self, tmp_path: Path) -> None:
        """The public serialiser exposes the exact generated bundle paths."""
        bundle = generate_asic_flow_bundle(tmp_path, pdk_type="sky130")

        payload = bundle.to_dict()

        assert payload["output_dir"] == bundle.output_dir
        assert payload["manifest_path"] == bundle.manifest_path
        assert payload["file_paths"] == dict(bundle.file_paths)

    def test_bundle_rejects_an_unknown_process_name(self, tmp_path: Path) -> None:
        """The public bundle API fails closed before writing an unknown PDK deck."""
        with pytest.raises(ValueError, match="unknown PDK type"):
            generate_asic_flow_bundle(tmp_path, pdk_type="not-a-real-pdk")

        assert list(tmp_path.iterdir()) == []
