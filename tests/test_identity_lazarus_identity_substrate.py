# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIdentitySubstrate from former test_identity_lazarus.py

"""Focused suite: TestIdentitySubstrate from former test_identity_lazarus.py."""

from __future__ import annotations

from tests.identity_lazarus_support import *  # noqa: F403


class TestIdentitySubstrate:
    def test_creation(self):
        sub = IdentitySubstrate(n_cortical=50, n_inhibitory=20, n_memory=10, seed=42)
        assert sub.n_cortical == 50
        assert sub.n_inhibitory == 20
        assert sub.n_memory == 10

    def test_step_returns_array(self):
        sub = IdentitySubstrate(n_cortical=20, n_inhibitory=10, n_memory=5, seed=42)
        spikes = sub.step(dt=0.001)
        assert isinstance(spikes, np.ndarray)
        assert len(spikes) == 20

    def test_run_shape(self):
        sub = IdentitySubstrate(n_cortical=20, n_inhibitory=10, n_memory=5, seed=42)
        result = sub.run(duration=0.05, dt=0.001)
        assert result.shape[0] == 50  # 50 ms / 1 ms
        assert result.shape[1] == 20

    def test_extract_state_keys(self):
        sub = IdentitySubstrate(n_cortical=30, n_inhibitory=10, n_memory=5, seed=42)
        sub.run(duration=0.02, dt=0.001)
        state = sub.extract_state()
        assert isinstance(state, dict)
        assert "firing_rates" in state or "total_steps" in state

    def test_health_check(self):
        sub = IdentitySubstrate(n_cortical=30, n_inhibitory=10, n_memory=5, seed=42)
        sub.run(duration=0.05, dt=0.001)
        health = sub.health_check()
        assert isinstance(health, dict)
        assert "is_healthy" in health or "mean_rate" in health
