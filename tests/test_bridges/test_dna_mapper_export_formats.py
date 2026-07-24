# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExportFormats from former test_dna_mapper.py

"""Focused suite: TestExportFormats from former test_dna_mapper.py."""

from __future__ import annotations

from dna_mapper_support import *  # noqa: F403


class TestExportFormats:
    """Verify exported file format correctness."""

    def test_genbank_export(
        self,
        simple_and_circuit: DNACircuitDesign,
        tmp_path: Path,
    ) -> None:
        path = str(tmp_path / "test.gb")
        export_genbank(simple_and_circuit, path)
        content = Path(path).read_text()
        assert "LOCUS" in content
        assert "ORIGIN" in content
        assert "//" in content
        assert "synthetic construct" in content

    def test_fasta_export(
        self,
        simple_and_circuit: DNACircuitDesign,
        tmp_path: Path,
    ) -> None:
        path = str(tmp_path / "test.fasta")
        export_fasta(simple_and_circuit, path)
        content = Path(path).read_text()
        lines = content.strip().split("\n")
        fasta_headers = [l for l in lines if l.startswith(">")]
        assert len(fasta_headers) >= 1

    def test_nupack_export(
        self,
        simple_and_circuit: DNACircuitDesign,
        tmp_path: Path,
    ) -> None:
        path = str(tmp_path / "test.nupack")
        export_nupack_input(simple_and_circuit, path)
        content = Path(path).read_text()
        assert "material = dna" in content
        assert "strand" in content

    def test_json_export(
        self,
        simple_and_circuit: DNACircuitDesign,
        tmp_path: Path,
    ) -> None:
        path = str(tmp_path / "test.json")
        export_json(simple_and_circuit, path)
        data = json.loads(Path(path).read_text())
        assert data["name"] == "simple_and"
        assert data["total_gates"] == 1
        assert "gates" in data
        assert len(data["gates"]) == 1

    def test_json_round_trip_fields(
        self,
        nand_circuit: DNACircuitDesign,
        tmp_path: Path,
    ) -> None:
        path = str(tmp_path / "nand.json")
        export_json(nand_circuit, path)
        data = json.loads(Path(path).read_text())
        for gate in data["gates"]:
            assert "gate_type" in gate
            assert "strands" in gate
            for strand in gate["strands"]:
                assert "sequence" in strand
                assert "gc_content" in strand
                assert "delta_g_37" in strand
