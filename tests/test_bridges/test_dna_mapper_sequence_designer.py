# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSequenceDesigner from former test_dna_mapper.py

"""Focused suite: TestSequenceDesigner from former test_dna_mapper.py."""

from __future__ import annotations

from dna_mapper_support import *  # noqa: F403

class TestSequenceDesigner:
    """Sequence generation constraint satisfaction."""

    def test_gc_content_within_bounds(self, designer: SequenceDesigner) -> None:
        for i in range(20):
            seq = designer.generate(30, f"test_gc_{i}")
            gc = sum(1 for c in seq if c in "GC") / len(seq)
            assert _GC_TARGET_LOW - 0.1 <= gc <= _GC_TARGET_HIGH + 0.1, (
                f"Sequence {i} GC={gc:.3f} outside bounds"
            )

    def test_no_excessive_homopolymer(self, designer: SequenceDesigner) -> None:
        for i in range(20):
            seq = designer.generate(50, f"test_homo_{i}")
            max_run = 1
            cur_run = 1
            for j in range(1, len(seq)):
                if seq[j] == seq[j - 1]:
                    cur_run += 1
                    max_run = max(max_run, cur_run)
                else:
                    cur_run = 1
            assert max_run <= _MAX_HOMOPOLYMER + 1, f"Sequence {i} has homopolymer run {max_run}"

    def test_deterministic_with_same_seed(self) -> None:
        d1 = SequenceDesigner(seed=99)
        d2 = SequenceDesigner(seed=99)
        seq1 = d1.generate(20, "x")
        seq2 = d2.generate(20, "x")
        assert seq1 == seq2

    def test_different_seeds_different_sequences(self) -> None:
        d1 = SequenceDesigner(seed=1)
        d2 = SequenceDesigner(seed=2)
        seq1 = d1.generate(20, "x")
        seq2 = d2.generate(20, "x")
        assert seq1 != seq2

    def test_complement_is_watson_crick(self, designer: SequenceDesigner) -> None:
        seq = designer.generate(20, "comp_test")
        comp = designer.generate_complement(seq)
        pairs = {"A": "T", "T": "A", "C": "G", "G": "C"}
        for i, c in enumerate(seq):
            expected = pairs[c]
            actual = comp[len(seq) - 1 - i]
            assert actual == expected, f"Position {i}: {c} → {actual}, expected {expected}"

    def test_toehold_length(self, designer: SequenceDesigner) -> None:
        th = designer.generate_toehold("test")
        assert len(th) == 6

    def test_recognition_length(self, designer: SequenceDesigner) -> None:
        rec = designer.generate_recognition("test")
        assert len(rec) == 15

    def test_orthogonality_low_overlap(self, designer: SequenceDesigner) -> None:
        sequences = [designer.generate(20, f"ortho_{i}") for i in range(10)]
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                overlap = sum(1 for a, b in zip(sequences[i], sequences[j]) if a == b)
                similarity = overlap / 20
                assert similarity < 0.90, f"Sequences {i} and {j} too similar: {similarity:.2f}"

    def test_sequence_scoring_penalizes_adverse_homopolymer_rng(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class ConstantRng:
            def integers(self, low: int, high: int) -> int:
                return low

            def choice(self, nucs: list[str], p: list[float]) -> str:
                return "A"

        monkeypatch.setattr(np.random, "default_rng", lambda seed=None: ConstantRng())

        seq = SequenceDesigner(seed=42).generate(4, "forced_homopolymer")

        assert seq == "AAAA"

    def test_sequence_generation_recovers_when_constraints_exhaust_weights(self) -> None:
        seq = SequenceDesigner(seed=42, max_homopolymer=0).generate(4, "zero_run_budget")

        assert len(seq) == 4
        assert set(seq).issubset({"A", "C", "G", "T"})
