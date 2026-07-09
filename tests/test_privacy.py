# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.privacy (differential privacy)

from __future__ import annotations

import numpy as np

from sc_neurocore.privacy import SpikeLevelDP, PrivacyAccountant, MembershipAudit


class TestPrivacyAccountant:
    def test_init(self):
        pa = PrivacyAccountant(target_epsilon=2.0)
        assert pa.spent_epsilon == 0.0
        assert pa.remaining_epsilon == 2.0
        assert not pa.budget_exhausted

    def test_record_steps(self):
        pa = PrivacyAccountant(target_epsilon=1.0)
        pa.record_step(0.3)
        pa.record_step(0.3)
        assert pa.spent_epsilon == 0.6
        assert pa.remaining_epsilon == 0.4
        assert not pa.budget_exhausted

    def test_budget_exhausted(self):
        pa = PrivacyAccountant(target_epsilon=0.5)
        pa.record_step(0.6)
        assert pa.budget_exhausted

    def test_summary(self):
        pa = PrivacyAccountant(target_epsilon=1.0)
        pa.record_step(0.1)
        s = pa.summary()
        assert "epsilon" in s


class TestSpikeLevelDP:
    def test_randomized_response(self):
        dp = SpikeLevelDP(epsilon=1.0, mechanism="randomized_response", seed=42)
        spikes = np.ones((20, 10), dtype=np.int8)
        private = dp.privatize(spikes)
        assert private.shape == spikes.shape
        # Some bits should be flipped
        assert not np.array_equal(private, spikes)

    def test_subsampling(self):
        dp = SpikeLevelDP(epsilon=1.0, mechanism="subsampling", seed=42)
        spikes = np.ones((20, 10), dtype=np.int8)
        private = dp.privatize(spikes)
        assert private.sum() <= spikes.sum()

    def test_high_epsilon_low_noise(self):
        dp = SpikeLevelDP(epsilon=10.0, mechanism="randomized_response", seed=42)
        spikes = np.ones((20, 10), dtype=np.int8)
        private = dp.privatize(spikes)
        # High epsilon = low noise, most bits preserved
        agreement = (private == spikes).mean()
        assert agreement > 0.9

    def test_low_epsilon_high_noise(self):
        dp = SpikeLevelDP(epsilon=0.1, mechanism="randomized_response", seed=42)
        spikes = np.ones((20, 10), dtype=np.int8)
        private = dp.privatize(spikes)
        # Low epsilon = high noise, nearly random
        agreement = (private == spikes).mean()
        assert agreement < 0.8

    def test_per_step_epsilon(self):
        dp = SpikeLevelDP(epsilon=2.0)
        assert dp.per_step_epsilon == 2.0

    def test_unknown_mechanism(self):
        import pytest

        with pytest.raises(ValueError):
            SpikeLevelDP(mechanism="unknown")

    def test_1d_input(self):
        dp = SpikeLevelDP(epsilon=1.0, seed=42)
        spikes = np.array([1, 0, 1, 1, 0], dtype=np.int8)
        private = dp.privatize(spikes)
        assert private.shape == (5,)


class TestMembershipAudit:
    def test_basic(self):
        def model(s):
            return s.sum(axis=0).astype(np.float64)

        members = [np.ones((10, 4), dtype=np.int8) for _ in range(5)]
        non_members = [np.zeros((10, 4), dtype=np.int8) for _ in range(5)]

        audit = MembershipAudit(run_fn=model)
        result = audit.audit(members, non_members)

        assert "accuracy" in result
        assert "vulnerable" in result
        assert 0 <= result["accuracy"] <= 1

    def test_indistinguishable(self):
        def constant_model(s):
            return np.ones(4)

        members = [np.random.randint(0, 2, (10, 4), dtype=np.int8) for _ in range(5)]
        non_members = [np.random.randint(0, 2, (10, 4), dtype=np.int8) for _ in range(5)]

        audit = MembershipAudit(run_fn=constant_model)
        result = audit.audit(members, non_members)
        # Constant model → 50% accuracy (no leakage)
        assert result["accuracy"] == 0.5
