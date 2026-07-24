# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGF4ErrorCorrection from former test_dna_mapper.py

"""Focused suite: TestGF4ErrorCorrection from former test_dna_mapper.py."""

from __future__ import annotations

from dna_mapper_support import *  # noqa: F403


class TestGF4ErrorCorrection:
    """Reed-Solomon over GF(4) for DNA error correction."""

    def test_encode_increases_length(self) -> None:
        ec = GF4ErrorCorrection(n_parity=4, block_size=12)
        original = "ACGTACGTACGT"
        encoded = ec.encode(original)
        assert len(encoded) > len(original)

    def test_round_trip_no_errors(self) -> None:
        ec = GF4ErrorCorrection(n_parity=4, block_size=12)
        original = "ACGTACGTACGT"
        encoded = ec.encode(original)
        decoded, corrections = ec.decode(encoded)
        assert decoded == original
        assert corrections == 0

    def test_detects_single_error(self) -> None:
        ec = GF4ErrorCorrection(n_parity=4, block_size=12)
        original = "ACGTACGTACGT"
        encoded = ec.encode(original)
        mutated = list(encoded)
        mutated[3] = "T" if mutated[3] != "T" else "A"
        mutated_str = "".join(mutated)
        _, corrections = ec.decode(mutated_str)
        assert corrections >= 1

    def test_multiple_blocks(self) -> None:
        ec = GF4ErrorCorrection(n_parity=4, block_size=12)
        original = "ACGTACGTACGT" * 5
        encoded = ec.encode(original)
        decoded, corrections = ec.decode(encoded)
        assert decoded == original
        assert corrections == 0
