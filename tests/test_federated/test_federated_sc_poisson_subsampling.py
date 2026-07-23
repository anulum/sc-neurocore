# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPoissonSubsampling from former test_federated_sc.py

"""Focused suite: TestPoissonSubsampling from former test_federated_sc.py."""

from __future__ import annotations

from federated_sc_support import *  # noqa: F403

class TestPoissonSubsampling:
    def test_always_returns_at_least_one(self):
        enc = SCGradientEncoder(bitstream_length=64)
        clients = [FederatedClient(client_id=i, encoder=enc) for i in range(5)]
        rng = np.random.default_rng(42)
        selected = poisson_subsample(clients, sampling_rate=0.01, rng=rng)
        assert len(selected) >= 1

    def test_full_rate_selects_all(self):
        enc = SCGradientEncoder(bitstream_length=64)
        clients = [FederatedClient(client_id=i, encoder=enc) for i in range(5)]
        rng = np.random.default_rng(42)
        selected = poisson_subsample(clients, sampling_rate=1.0, rng=rng)
        assert len(selected) == 5

    def test_half_rate_reasonable(self):
        enc = SCGradientEncoder(bitstream_length=64)
        clients = [FederatedClient(client_id=i, encoder=enc) for i in range(100)]
        rng = np.random.default_rng(42)
        selected = poisson_subsample(clients, sampling_rate=0.5, rng=rng)
        assert 20 < len(selected) < 80
