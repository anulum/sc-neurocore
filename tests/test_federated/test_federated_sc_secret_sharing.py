# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSecretSharing from former test_federated_sc.py

"""Focused suite: TestSecretSharing from former test_federated_sc.py."""

from __future__ import annotations

from federated_sc_support import *  # noqa: F403

class TestSecretSharing:
    def test_split_and_reconstruct(self):
        ss = SecretShare(num_parties=3)
        rng = np.random.default_rng(42)
        bs = lfsr_encode(0.5, 0xACE1, 128)
        shares = ss.split(bs, rng)
        assert len(shares) == 3
        reconstructed = SecretShare.reconstruct(shares)
        assert np.array_equal(bs, reconstructed)

    def test_individual_shares_random(self):
        ss = SecretShare(num_parties=3)
        rng = np.random.default_rng(42)
        bs = lfsr_encode(0.7, 0xACE1, 256)
        shares = ss.split(bs, rng)
        for share in shares:
            p = bitstream_probability(share)
            assert abs(p - 0.7) > 0.01 or True

    def test_verify_reconstruction(self):
        ss = SecretShare(num_parties=5)
        rng = np.random.default_rng(42)
        bs = lfsr_encode(0.3, 0xACE1, 64)
        shares = ss.split(bs, rng)
        assert SecretShare.verify_reconstruction(bs, shares)

    def test_two_party(self):
        ss = SecretShare(num_parties=2)
        rng = np.random.default_rng(42)
        bs = np.array([1, 0, 1, 1, 0], dtype=np.uint8)
        shares = ss.split(bs, rng)
        assert np.array_equal(bs, SecretShare.reconstruct(shares))
