# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Dopamine STDP synapse contracts

"""Module-specific behavioural contracts for ``DopamineStdpSynapse``."""

from __future__ import annotations

import pytest


class TestDopamineStdpSynapse:
    @pytest.fixture()
    def synapse(self):
        from sc_neurocore.synapses import DopamineStdpSynapse

        return DopamineStdpSynapse(weight=0.5)

    def test_defaults(self, synapse):
        assert synapse.tau_e == 1000.0
        assert synapse.tau_da == 200.0
        assert synapse.a_plus == 1.0
        assert synapse.a_minus == -1.0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"w_min": 1.0, "w_max": 0.0},
            {"weight": -0.01},
            {"weight": 1.01},
            {"tau_e": 0.0},
            {"tau_da": 0.0},
            {"tau_pre": 0.0},
            {"tau_post": 0.0},
            {"a_plus": -0.01},
            {"a_minus": 0.01},
            {"lr": -0.01},
            {"dt": 0.0},
            {"eligibility": float("nan")},
            {"dopamine": float("inf")},
            {"trace_pre": float("nan")},
            {"trace_post": float("inf")},
            {"trace_pre": -1.0},
        ],
    )
    def test_rejects_non_physical_dopamine_stdp_parameters(self, kwargs):
        """Dopamine-gated STDP constants, traces, and bounds must be physical."""
        from sc_neurocore.synapses import DopamineStdpSynapse

        with pytest.raises(ValueError):
            DopamineStdpSynapse(**kwargs)

    @pytest.mark.parametrize(
        ("pre_spike", "post_spike", "reward"),
        [(1, False, 0.0), (False, 0, 0.0), (False, False, float("nan"))],
    )
    def test_rejects_invalid_dopamine_stdp_step_inputs(self, pre_spike, post_spike, reward):
        """Spike flags must be boolean and reward must be finite."""
        from sc_neurocore.synapses import DopamineStdpSynapse

        with pytest.raises((TypeError, ValueError)):
            DopamineStdpSynapse().step(pre_spike, post_spike, reward)

    def test_step_returns_float(self, synapse):
        w = synapse.step(True, False, 0.0)
        assert isinstance(w, float)

    def test_no_reward_no_weight_change(self, synapse):
        """Without dopamine, eligibility doesn't convert to weight change."""
        w0 = synapse.weight
        for i in range(50):
            synapse.step(i % 10 == 0, i % 10 == 2, reward=0.0)
        # Small or zero weight change without DA.
        assert abs(synapse.weight - w0) < 0.01

    def test_reward_drives_weight_change(self, synapse):
        """With reward (dopamine), weight should change."""
        w0 = synapse.weight
        for i in range(200):
            synapse.step(
                i % 10 == 0,
                i % 10 == 2,
                reward=0.5 if i % 5 == 0 else 0.0,
            )
        assert synapse.weight != w0, "Reward must drive weight change"

    def test_eligibility_trace_builds(self, synapse):
        """Pre/post spikes build eligibility trace."""
        synapse.step(True, False, 0.0)
        synapse.step(False, True, 0.0)
        assert synapse.eligibility != 0.0

    def test_eligibility_decays(self, synapse):
        synapse.step(True, False, 0.0)
        synapse.step(False, True, 0.0)
        e_after_spikes = abs(synapse.eligibility)
        for _ in range(5000):
            synapse.step(False, False, 0.0)
        assert abs(synapse.eligibility) < e_after_spikes * 0.01

    def test_dopamine_integrates_reward(self, synapse):
        synapse.step(False, False, reward=1.0)
        assert synapse.dopamine > 0.0

    def test_dopamine_decays(self, synapse):
        synapse.step(False, False, reward=10.0)
        da_high = synapse.dopamine
        for _ in range(2000):
            synapse.step(False, False, reward=0.0)
        assert synapse.dopamine < da_high * 0.01

    def test_weight_clamped(self, synapse):
        for _ in range(1000):
            synapse.step(True, True, reward=10.0)
        assert synapse.w_min <= synapse.weight <= synapse.w_max

    def test_reset(self, synapse):
        synapse.step(True, True, reward=5.0)
        synapse.reset()
        assert synapse.eligibility == 0.0
        assert synapse.dopamine == 0.0
        assert synapse.trace_pre == 0.0
        assert synapse.trace_post == 0.0

    def test_distal_reward_problem(self):
        """Core Izhikevich (2007) result: delayed reward still modifies weight."""
        from sc_neurocore.synapses import DopamineStdpSynapse

        syn = DopamineStdpSynapse(weight=0.5, lr=0.01)
        # Phase 1: STDP pairing (builds eligibility, no reward).
        for i in range(50):
            syn.step(i % 5 == 0, i % 5 == 1, reward=0.0)
        assert syn.eligibility != 0.0
        w_before_reward = syn.weight
        # Phase 2: Delayed reward (no more spikes).
        for _ in range(100):
            syn.step(False, False, reward=1.0)
        # Weight should change from delayed reward acting on eligibility.
        assert syn.weight != w_before_reward, "Delayed reward must drive learning"
