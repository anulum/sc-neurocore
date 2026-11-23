# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPoissonSubsampling from former test_federated_sc.py

"""Focused suite: TestPoissonSubsampling from former test_federated_sc.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.federated.federated_sc import (
    FederatedClient,
    poisson_subsample,
    SCGradientEncoder,
)


class TestPoissonSubsampling:
    """Poisson subsampling of clients."""

    def test_always_returns_at_least_one(self) -> None:
        """At least one client is always selected."""
        enc = SCGradientEncoder(bitstream_length=64)
        clients = [FederatedClient(client_id=i, encoder=enc) for i in range(5)]
        rng = np.random.default_rng(42)
        selected = poisson_subsample(clients, sampling_rate=0.01, rng=rng)
        assert len(selected) >= 1

    def test_full_rate_selects_all(self) -> None:
        """A rate of 1 selects every client."""
        enc = SCGradientEncoder(bitstream_length=64)
        clients = [FederatedClient(client_id=i, encoder=enc) for i in range(5)]
        rng = np.random.default_rng(42)
        selected = poisson_subsample(clients, sampling_rate=1.0, rng=rng)
        assert len(selected) == 5

    def test_half_rate_reasonable(self) -> None:
        """A rate of 0.5 selects between 20 and 80 of 100."""
        enc = SCGradientEncoder(bitstream_length=64)
        clients = [FederatedClient(client_id=i, encoder=enc) for i in range(100)]
        rng = np.random.default_rng(42)
        selected = poisson_subsample(clients, sampling_rate=0.5, rng=rng)
        assert 20 < len(selected) < 80
