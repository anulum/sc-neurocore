# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFederatedRound from former test_federated_sc.py

"""Focused suite: TestFederatedRound from former test_federated_sc.py."""

from __future__ import annotations

from federated_sc_support import *  # noqa: F403

class TestFederatedRound:
    def _make_round(
        self,
        num_clients=3,
        epsilon=10.0,
        target_eps=1000.0,
        clip_norm=0.0,
        sampling_rate=1.0,
        audit_log=None,
    ):
        enc = SCGradientEncoder(bitstream_length=128, dp=DPMechanism(epsilon=epsilon))
        clients = [FederatedClient(client_id=i, encoder=enc) for i in range(num_clients)]
        agg = FederatedAggregator(num_clients=num_clients, bitstream_length=128)
        acc = PrivacyAccountant(target_epsilon=target_eps)
        return FederatedRound(
            clients=clients,
            aggregator=agg,
            accountant=acc,
            clip_norm=clip_norm,
            sampling_rate=sampling_rate,
            audit_log=audit_log,
        )

    def test_single_round(self):
        rng = np.random.default_rng(42)
        fr = self._make_round()
        data = [rng.standard_normal((10, 3)) for _ in range(3)]
        labels = [rng.standard_normal(10) for _ in range(3)]
        result = fr.run(data, labels)
        assert result is not None
        assert len(result) == 3

    def test_multiple_rounds(self):
        rng = np.random.default_rng(42)
        fr = self._make_round()
        for _ in range(5):
            data = [rng.standard_normal((10, 3)) for _ in range(3)]
            labels = [rng.standard_normal(10) for _ in range(3)]
            fr.run(data, labels)
        assert fr.round_number == 5

    def test_budget_exhaustion_stops(self):
        fr = self._make_round(target_eps=0.001)
        rng = np.random.default_rng(42)
        data = [rng.standard_normal((10, 3)) for _ in range(3)]
        labels = [rng.standard_normal(10) for _ in range(3)]
        fr.run(data, labels)
        result = fr.run(data, labels)
        assert result is None

    def test_status(self):
        fr = self._make_round()
        status = fr.status()
        assert "round" in status
        assert "epsilon_consumed" in status
        assert "epsilon_remaining" in status
        assert "budget_exhausted" in status
        assert "converged" in status
        assert "trend" in status

    def test_convergence_tracking(self):
        rng = np.random.default_rng(42)
        fr = self._make_round()
        data = [rng.standard_normal((10, 3)) for _ in range(3)]
        labels = [rng.standard_normal(10) for _ in range(3)]
        fr.run(data, labels)
        assert len(fr.convergence.grad_norms) == 1

    def test_gradient_clipping_active(self):
        rng = np.random.default_rng(42)
        fr = self._make_round(clip_norm=0.01)
        data = [rng.standard_normal((10, 3)) for _ in range(3)]
        labels = [rng.standard_normal(10) for _ in range(3)]
        result = fr.run(data, labels)
        assert result is not None

    def test_subsampling_round(self):
        rng = np.random.default_rng(42)
        fr = self._make_round(num_clients=10, sampling_rate=0.5)
        data = [rng.standard_normal((10, 3)) for _ in range(10)]
        labels = [rng.standard_normal(10) for _ in range(10)]
        result = fr.run(data, labels)
        assert result is not None

    def test_weighted_round(self):
        rng = np.random.default_rng(42)
        fr = self._make_round(num_clients=3)
        data = [rng.standard_normal((10, 3)) for _ in range(3)]
        labels = [rng.standard_normal(10) for _ in range(3)]
        result = fr.run(data, labels, client_weights=[0.5, 0.3, 0.2])
        assert result is not None
        assert len(result) == 3

    def test_audit_log_integration(self):
        log = AuditLog()
        rng = np.random.default_rng(42)
        fr = self._make_round(audit_log=log)
        data = [rng.standard_normal((10, 3)) for _ in range(3)]
        labels = [rng.standard_normal(10) for _ in range(3)]
        fr.run(data, labels)
        fr.run(data, labels)
        assert log.total_rounds == 2
        entries = log.to_list()
        assert entries[0]["round"] == 1
        assert entries[1]["round"] == 2
