# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDNAStrand from former test_dna_mapper.py

"""Focused suite: TestDNAStrand from former test_dna_mapper.py."""

from __future__ import annotations

from dna_mapper_support import *  # noqa: F403

class TestDNAStrand:
    """Strand data class properties."""

    def test_length(self) -> None:
        s = DNAStrand(name="x", sequence="ACGTACGT")
        assert s.length == 8

    def test_gc_content(self) -> None:
        s = DNAStrand(name="x", sequence="GCGCGC")
        assert s.gc_content == 1.0
        s2 = DNAStrand(name="y", sequence="ATATAT")
        assert s2.gc_content == 0.0

    def test_complement(self) -> None:
        s = DNAStrand(name="x", sequence="ACGT")
        assert s.complement == "ACGT"  # palindromic

    def test_max_homopolymer_run(self) -> None:
        s = DNAStrand(name="x", sequence="AACCCGT")
        assert s.max_homopolymer_run == 3

    def test_delta_g_negative(self) -> None:
        s = DNAStrand(name="x", sequence="GCGCGCGCGC")
        dg = s.delta_g_37()
        assert dg < 0, "GC-rich sequence should have negative ΔG"

    def test_melting_temperature_gc_rich_higher(self) -> None:
        gc_rich = DNAStrand(name="gc", sequence="GCGCGCGCGCGCGCGC")
        at_rich = DNAStrand(name="at", sequence="ATATATATATATATATAT")
        assert gc_rich.melting_temperature() > at_rich.melting_temperature()

    def test_melting_temperature_depends_on_salt_and_strand_concentration(self) -> None:
        strand = DNAStrand(name="thermo", sequence="ACGTTGCAACGTTGCA")
        low_salt = strand.melting_temperature(na_conc_M=0.05, strand_conc_M=2.5e-7)
        high_salt = strand.melting_temperature(na_conc_M=1.0, strand_conc_M=2.5e-7)
        high_conc = strand.melting_temperature(na_conc_M=0.05, strand_conc_M=2.5e-6)

        assert high_salt > low_salt
        assert high_conc > low_salt

    def test_melting_temperature_rejects_invalid_conditions(self) -> None:
        strand = DNAStrand(name="thermo", sequence="ACGTTGCA")
        with pytest.raises(ValueError, match="na_conc_M must be finite and positive"):
            strand.melting_temperature(na_conc_M=0.0)
        with pytest.raises(ValueError, match="strand_conc_M must be finite and positive"):
            strand.melting_temperature(strand_conc_M=0.0)

    def test_empty_strand(self) -> None:
        s = DNAStrand(name="empty", sequence="")
        assert s.length == 0
        assert s.gc_content == 0.0
        assert s.max_homopolymer_run == 0

    def test_short_strand_has_zero_delta_g_and_rejects_tm(self) -> None:
        s = DNAStrand(name="short", sequence="A")

        assert s.delta_g_37() == 0.0
        with pytest.raises(ValueError, match="at least two nucleotides"):
            s.melting_temperature()
