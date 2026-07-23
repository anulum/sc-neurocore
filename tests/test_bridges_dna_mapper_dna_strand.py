# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDNAStrand from former test_bridges_dna_mapper.py

"""Focused suite: TestDNAStrand from former test_bridges_dna_mapper.py."""

from __future__ import annotations

from tests.bridges_dna_mapper_support import *  # noqa: F403

class TestDNAStrand:
    def test_gc_content_all_gc(self) -> None:
        s = DNAStrand(name="s1", sequence="GCGC")
        assert s.gc_content == pytest.approx(1.0)

    def test_gc_content_all_at(self) -> None:
        s = DNAStrand(name="s2", sequence="ATAT")
        assert s.gc_content == pytest.approx(0.0)

    def test_gc_content_mixed(self) -> None:
        s = DNAStrand(name="s3", sequence="ATGC")
        assert s.gc_content == pytest.approx(0.5)

    def test_complement(self) -> None:
        s = DNAStrand(name="s4", sequence="ATCG")
        comp = s.complement
        assert isinstance(comp, str)
        assert len(comp) == 4
        assert set(comp).issubset(set("ATCG"))

    def test_max_homopolymer_run(self) -> None:
        s = DNAStrand(name="s5", sequence="AAAAATCG")
        assert s.max_homopolymer_run >= 5

    def test_delta_g_37(self) -> None:
        s = DNAStrand(name="s6", sequence="GCGCGCGCGCGC")
        dg = s.delta_g_37()
        assert isinstance(dg, float)
