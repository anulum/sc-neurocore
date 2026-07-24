# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDNAEncoder from former test_bio.py

"""Focused suite: TestDNAEncoder from former test_bio.py."""

from __future__ import annotations

from tests.bio_support import *  # noqa: F403


class TestDNAEncoder:
    def test_encode_basic(self):
        enc = DNAEncoder(mutation_rate=0.0)
        bits = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)
        dna = enc.encode(bits)
        assert dna == "ACGT"

    def test_decode_lossless(self):
        enc = DNAEncoder(mutation_rate=0.0)
        bits = np.array([1, 0, 0, 1, 1, 1, 0, 0], dtype=np.uint8)
        dna = enc.encode(bits)
        recovered = enc.decode(dna)
        np.testing.assert_array_equal(bits, recovered)

    def test_odd_length_padded(self):
        enc = DNAEncoder(mutation_rate=0.0)
        bits = np.array([1, 0, 1], dtype=np.uint8)
        dna = enc.encode(bits)
        assert len(dna) == 2

    def test_roundtrip_even(self):
        np.random.seed(42)
        enc = DNAEncoder(mutation_rate=0.0)
        bits = np.random.randint(0, 2, 20).astype(np.uint8)
        dna = enc.encode(bits)
        recovered = enc.decode(dna)
        np.testing.assert_array_equal(bits, recovered)
