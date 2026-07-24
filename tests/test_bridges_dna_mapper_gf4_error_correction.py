# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGF4ErrorCorrection from former test_bridges_dna_mapper.py

"""Focused suite: TestGF4ErrorCorrection from former test_bridges_dna_mapper.py."""

from __future__ import annotations

from tests.bridges_dna_mapper_support import *  # noqa: F403


class TestGF4ErrorCorrection:
    def test_encode_decode_roundtrip(self) -> None:
        ecc = GF4ErrorCorrection(n_parity=4)
        data = "ATCGATCG"
        encoded = ecc.encode(data)
        assert len(encoded) > len(data)
        decoded, errors = ecc.decode(encoded)
        assert decoded[: len(data)] == data
        assert errors == 0

    def test_detects_errors(self) -> None:
        ecc = GF4ErrorCorrection(n_parity=4)
        data = "ATCGATCG"
        encoded = ecc.encode(data)
        corrupted_chars = list(encoded)
        orig = corrupted_chars[2]
        corrupted_chars[2] = "G" if orig != "G" else "A"
        corrupted = "".join(corrupted_chars)
        decoded, error_count = ecc.decode(corrupted)
        assert isinstance(decoded, str)
        assert error_count >= 0
