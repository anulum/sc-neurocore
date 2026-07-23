# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSymbolEncoder from former test_predictive_coding.py

"""Focused suite: TestSymbolEncoder from former test_predictive_coding.py."""

from __future__ import annotations

from predictive_coding_support import *  # noqa: F403

class TestSymbolEncoder:
    def test_deterministic(self):
        enc1 = SymbolEncoder(42)
        enc2 = SymbolEncoder(42)
        a = enc1.encode("hello")
        b = enc2.encode("hello")
        assert np.array_equal(a.data, b.data)

    def test_different_symbols_orthogonal(self):
        enc = SymbolEncoder(42)
        a = enc.encode("cat")
        b = enc.encode("dog")
        assert abs(a.similarity(b)) < 0.15

    def test_vocabulary_size(self):
        enc = SymbolEncoder(42)
        enc.encode("a")
        enc.encode("b")
        enc.encode("a")
        assert enc.vocabulary_size == 2

    def test_sequence_order_matters(self):
        enc = SymbolEncoder(42)
        ab = enc.encode_sequence(["A", "B"])
        ba = enc.encode_sequence(["B", "A"])
        assert abs(ab.similarity(ba)) < 0.2

    def test_sequence_single_symbol(self):
        enc = SymbolEncoder(42)
        single = enc.encode("X")
        seq = enc.encode_sequence(["X"])
        assert np.array_equal(single.data, seq.data)
