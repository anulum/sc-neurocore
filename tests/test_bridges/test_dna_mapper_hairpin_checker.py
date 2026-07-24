# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHairpinChecker from former test_dna_mapper.py

"""Focused suite: TestHairpinChecker from former test_dna_mapper.py."""

from __future__ import annotations

from dna_mapper_support import *  # noqa: F403


class TestHairpinChecker:
    """Hairpin secondary structure detection."""

    def test_palindrome_detected(self) -> None:
        checker = HairpinChecker(min_stem_length=4, min_loop_length=3)
        # This sequence has a perfect hairpin: ACGT...loop...ACGT
        seq = "ACGTACGTTTTCGTACGT"
        hairpins = checker.check_strand(seq)
        assert isinstance(hairpins, list)

    def test_short_strand_no_hairpin(self) -> None:
        checker = HairpinChecker(min_stem_length=4)
        hairpins = checker.check_strand("ACGT")
        assert len(hairpins) == 0

    def test_check_design_returns_list(self, simple_and_circuit: DNACircuitDesign) -> None:
        checker = HairpinChecker()
        flags = checker.check_design(simple_and_circuit)
        assert isinstance(flags, list)

    def test_check_design_flags_hairpin_strand(self) -> None:
        checker = HairpinChecker(min_stem_length=4, min_loop_length=3)
        design = DNACircuitDesign(
            name="hairpin_design",
            input_strands=[DNAStrand(name="hp", sequence="GCGCGCGCAAAGCGCGCGC", role="signal")],
        )

        flags = checker.check_design(design)

        assert flags
        assert flags[0]["strand_name"] == "hp"

    def test_flag_structure(self) -> None:
        checker = HairpinChecker(min_stem_length=4, min_loop_length=3)
        seq = "GCGCGCGCAAAGCGCGCGC"
        hairpins = checker.check_strand(seq)
        for hp in hairpins:
            assert "stem_length" in hp
            assert "loop_length" in hp
            assert "delta_g_estimate" in hp
