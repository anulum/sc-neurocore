# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCommitmentScheme from former test_federated_sc.py

"""Focused suite: TestCommitmentScheme from former test_federated_sc.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.federated.federated_sc import (
    CommitmentScheme,
)


class TestCommitmentScheme:
    """Hash commitments to client updates."""

    def test_commit_deterministic(self) -> None:
        """The same data commits to the same value."""
        data = np.array([1, 0, 1, 1], dtype=np.uint8)
        c1 = CommitmentScheme.commit(data)
        c2 = CommitmentScheme.commit(data)
        assert c1 == c2

    def test_commit_different_data(self) -> None:
        """Different data commit to different values."""
        a = np.array([1, 0, 1], dtype=np.uint8)
        b = np.array([0, 1, 0], dtype=np.uint8)
        assert CommitmentScheme.commit(a) != CommitmentScheme.commit(b)

    def test_verify(self) -> None:
        """A commitment verifies against its data."""
        data = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        c = CommitmentScheme.commit(data)
        assert CommitmentScheme.verify(data, c)

    def test_verify_with_nonce(self) -> None:
        """A nonced commitment verifies with its nonce."""
        rng = np.random.default_rng(42)
        data = np.array([1, 0, 1], dtype=np.uint8)
        nonce = CommitmentScheme.generate_nonce(rng)
        c = CommitmentScheme.commit(data, nonce)
        assert CommitmentScheme.verify(data, c, nonce)

    def test_nonce_binding(self) -> None:
        """Different nonces give different commitments."""
        rng = np.random.default_rng(42)
        data = np.array([1, 0, 1], dtype=np.uint8)
        n1 = CommitmentScheme.generate_nonce(rng)
        n2 = CommitmentScheme.generate_nonce(rng)
        assert CommitmentScheme.commit(data, n1) != CommitmentScheme.commit(data, n2)

    def test_sha256_length(self) -> None:
        """A commitment is a 64-character SHA-256 digest."""
        data = np.array([1, 0], dtype=np.uint8)
        c = CommitmentScheme.commit(data)
        assert len(c) == 64
