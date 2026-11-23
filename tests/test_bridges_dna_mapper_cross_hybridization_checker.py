# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCrossHybridizationChecker from former test_bridges_dna_mapper.py

"""Focused suite: TestCrossHybridizationChecker from former test_bridges_dna_mapper.py."""

from __future__ import annotations

from sc_neurocore.bridges.dna_mapper import (
    CrossHybridizationChecker,
    DNACircuitDesign,
    DNAStrand,
    StrandDisplacementCompiler,
)


class TestCrossHybridizationChecker:
    """Smoke test of the cross-hybridisation check."""

    def test_check_returns_list(self) -> None:
        """The result is a list; its contents are not checked."""
        compiler = StrandDisplacementCompiler()
        gate = compiler.compile_and("a", "b", "out")
        design = DNACircuitDesign(
            name="test",
            gates=[gate],
            input_strands=[
                DNAStrand(name="a", sequence="ACGTACGT"),
                DNAStrand(name="b", sequence="TGCATGCA"),
            ],
            output_strands=[DNAStrand(name="out", sequence="AGCTAGCT")],
        )

        result = CrossHybridizationChecker().check(design)

        assert isinstance(result, list)
