# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFederatedClient from former test_federated_sc.py

"""Focused suite: TestFederatedClient from former test_federated_sc.py."""

from __future__ import annotations

from federated_sc_support import *  # noqa: F403


class TestFederatedClient:
    def test_local_train(self):
        enc = SCGradientEncoder(bitstream_length=128, dp=DPMechanism(epsilon=2.0))
        client = FederatedClient(client_id=0, encoder=enc)
        data = np.random.default_rng(42).standard_normal((20, 5))
        labels = np.random.default_rng(42).standard_normal(20)
        grads = client.local_train(data, labels)
        assert len(grads) == 5

    def test_encode_gradients(self):
        enc = SCGradientEncoder(bitstream_length=128, dp=DPMechanism(epsilon=2.0))
        client = FederatedClient(client_id=1, encoder=enc)
        grads = np.array([0.1, -0.2, 0.3])
        bitstreams, commitment, g_min, g_max = client.encode_gradients(grads)
        assert len(bitstreams) == 3
        assert len(commitment) == 64
        assert g_min <= g_max

    def test_deterministic_by_client_id(self):
        enc = SCGradientEncoder(bitstream_length=128, dp=DPMechanism(epsilon=5.0))
        c1 = FederatedClient(client_id=0, encoder=enc)
        c2 = FederatedClient(client_id=1, encoder=enc)
        grads = np.array([0.5])
        bs1, _, _, _ = c1.encode_gradients(grads)
        bs2, _, _, _ = c2.encode_gradients(grads)
        assert not np.array_equal(bs1[0], bs2[0])
