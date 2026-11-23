# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPlateLayout from former test_dna_mapper.py

"""Focused suite: TestPlateLayout from former test_dna_mapper.py."""

from __future__ import annotations

from sc_neurocore.bridges.dna_mapper import (
    DNACircuitDesign,
    DNAStrand,
    PlateLayout,
)


class TestPlateLayout:
    """96-well plate layout generation."""

    def test_layout_produces_plates(self, simple_and_circuit: DNACircuitDesign) -> None:
        """The layout fills at least one plate."""
        pl = PlateLayout()
        result = pl.layout(simple_and_circuit)
        assert result["n_plates"] >= 1
        assert result["n_unique_oligos"] > 0

    def test_well_format(self, simple_and_circuit: DNACircuitDesign) -> None:
        """Wells are named by row letter and two digits."""
        pl = PlateLayout()
        result = pl.layout(simple_and_circuit)
        for plate in result["plates"]:
            for entry in plate:
                assert len(entry["well"]) == 3  # e.g. A01
                assert entry["well"][0] in "ABCDEFGH"

    def test_csv_manifest(self, simple_and_circuit: DNACircuitDesign) -> None:
        """The manifest is a CSV with a header and rows."""
        pl = PlateLayout()
        result = pl.layout(simple_and_circuit)
        csv = result["manifest_csv"]
        assert csv.startswith("Well,Name,Sequence,Length")
        lines = csv.strip().split("\n")
        assert len(lines) > 1

    def test_layout_splits_across_multiple_plates(self) -> None:
        """Many oligos spill over three plates in well order."""
        design = DNACircuitDesign(
            name="multi_plate",
            input_strands=[
                DNAStrand(name=f"s{i}", sequence=f"ACGTACGTACGT{i % 10}", role="signal")
                for i in range(5)
            ],
        )
        pl = PlateLayout(n_wells=2)

        result = pl.layout(design)

        assert result["n_plates"] == 3
        assert [entry["well"] for plate in result["plates"] for entry in plate] == [
            "A01",
            "A02",
            "A01",
            "A02",
            "A01",
        ]
