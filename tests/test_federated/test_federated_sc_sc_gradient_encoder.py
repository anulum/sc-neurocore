# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCGradientEncoder from former test_federated_sc.py

"""Focused suite: TestSCGradientEncoder from former test_federated_sc.py."""

from __future__ import annotations

from federated_sc_support import *  # noqa: F403


class TestSCGradientEncoder:
    def test_encode_decode_roundtrip(self):
        enc = SCGradientEncoder(bitstream_length=1024, dp=DPMechanism(epsilon=10.0))
        rng = np.random.default_rng(42)
        gradients = np.array([0.1, 0.5, 0.9])
        seeds = np.array([0xACE1, 0xBEEF, 0xCAFE])
        bitstreams = enc.encode(gradients, seeds, rng)
        decoded = enc.decode(bitstreams, gradients.min(), gradients.max())
        assert len(decoded) == 3
        for i in range(3):
            assert abs(decoded[i] - gradients[i]) < 0.15

    def test_encode_length(self):
        enc = SCGradientEncoder(bitstream_length=512)
        rng = np.random.default_rng(42)
        gradients = np.array([0.3, 0.7])
        seeds = np.array([0xACE1, 0xBEEF])
        bitstreams = enc.encode(gradients, seeds, rng)
        assert len(bitstreams) == 2
        assert len(bitstreams[0]) == 512

    def test_encode_zero_seed_is_reset(self):
        # A supplied seed of 0 masks to a zero register, which the encoder must
        # bump to 1 before LFSR stepping.
        enc = SCGradientEncoder(bitstream_length=64)
        rng = np.random.default_rng(0)
        gradients = np.array([0.2, 0.8])
        bitstreams = enc.encode(gradients, np.array([0, 0]), rng)
        assert len(bitstreams) == 2
        assert all(len(bs) == 64 for bs in bitstreams)
