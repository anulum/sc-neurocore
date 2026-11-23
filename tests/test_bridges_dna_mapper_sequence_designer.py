# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSequenceDesigner from former test_bridges_dna_mapper.py

"""Focused suite: TestSequenceDesigner from former test_bridges_dna_mapper.py."""

from __future__ import annotations

from sc_neurocore.bridges.dna_mapper import (
    SequenceDesigner,
)


class TestSequenceDesigner:
    """Designing DNA sequences."""

    def test_generate_returns_valid_dna(self) -> None:
        """A generated sequence has the asked length of bases."""
        designer = SequenceDesigner(seed=42)
        seq = designer.generate(length=20)
        assert len(seq) == 20
        assert all(c in "ATCG" for c in seq)

    def test_generate_complement(self) -> None:
        """The complement has the sequence's length."""
        designer = SequenceDesigner(seed=42)
        seq = designer.generate(length=15)
        comp = designer.generate_complement(seq)
        assert len(comp) == len(seq)

    def test_generate_toehold(self) -> None:
        """A toehold is a DNA string."""
        designer = SequenceDesigner(seed=42)
        th = designer.generate_toehold()
        assert isinstance(th, str)
        assert all(c in "ATCG" for c in th)

    def test_deterministic_with_seed(self) -> None:
        """The same seed designs the same sequence."""
        d1 = SequenceDesigner(seed=123)
        d2 = SequenceDesigner(seed=123)
        assert d1.generate(length=30) == d2.generate(length=30)

    def test_generate_recognition(self) -> None:
        """A recognition domain is a string."""
        designer = SequenceDesigner(seed=42)
        rec = designer.generate_recognition()
        assert isinstance(rec, str)
