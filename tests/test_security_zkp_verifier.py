# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestZKPVerifier from former test_security.py

"""Focused suite: TestZKPVerifier from former test_security.py."""

from __future__ import annotations

from tests.security_support import *  # noqa: F403

class TestZKPVerifier:
    """Test suite for Zero-Knowledge Proof spike verification."""

    def test_commit_generates_hash(self):
        """Commitment should generate a valid SHA256 hash."""
        bitstream = np.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=np.uint8)
        commitment = ZKPVerifier.commit(bitstream)

        assert isinstance(commitment, str)
        assert len(commitment) == 64  # SHA256 hex digest length
        assert all(c in "0123456789abcdef" for c in commitment)

    def test_commit_deterministic(self):
        """Same input should produce same commitment."""
        bitstream = np.array([1, 0, 1, 1], dtype=np.uint8)
        c1 = ZKPVerifier.commit(bitstream)
        c2 = ZKPVerifier.commit(bitstream)

        assert c1 == c2

    def test_commit_different_inputs_different_hashes(self):
        """Different inputs should produce different commitments."""
        bs1 = np.array([1, 0, 1, 1], dtype=np.uint8)
        bs2 = np.array([1, 0, 1, 0], dtype=np.uint8)

        c1 = ZKPVerifier.commit(bs1)
        c2 = ZKPVerifier.commit(bs2)

        assert c1 != c2

    def test_generate_challenge_deterministic(self):
        """Challenge should be deterministic based on commitment."""
        commitment = "abc123def456abc123def456abc123def456abc123def456abc123def456abcd"
        ch1 = ZKPVerifier.generate_challenge(commitment)
        ch2 = ZKPVerifier.generate_challenge(commitment)

        assert ch1 == ch2

    def test_generate_challenge_in_range(self):
        """Challenge index should be in valid range (0-9)."""
        for i in range(100):
            commitment = f"{i:064x}"
            challenge = ZKPVerifier.generate_challenge(commitment)
            assert 0 <= challenge < 10

    def test_verify_returns_bool(self):
        """Verify should return a boolean result."""
        bitstream = np.array([1, 0, 1, 1, 0, 1, 0, 0], dtype=np.uint8)
        commitment = ZKPVerifier.commit(bitstream)
        challenge = ZKPVerifier.generate_challenge(commitment)

        result = ZKPVerifier.verify(
            commitment=commitment,
            challenge_idx=challenge,
            revealed_bit=int(bitstream[challenge % len(bitstream)]),
            bitstream_slice=bitstream,
        )

        assert isinstance(result, bool)

    def test_full_zkp_protocol_flow(self):
        """Test complete ZKP protocol: commit -> challenge -> verify."""
        # Prover has a bitstream
        prover_bitstream = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=np.uint8)

        # Step 1: Prover commits
        commitment = ZKPVerifier.commit(prover_bitstream)

        # Step 2: Verifier generates challenge
        challenge_idx = ZKPVerifier.generate_challenge(commitment)

        # Step 3: Prover reveals bit at challenge index
        revealed_bit = int(prover_bitstream[challenge_idx])

        # Step 4: Verifier checks
        is_valid = ZKPVerifier.verify(
            commitment=commitment,
            challenge_idx=challenge_idx,
            revealed_bit=revealed_bit,
            bitstream_slice=prover_bitstream,
        )

        assert is_valid is True
