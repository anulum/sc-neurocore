# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRewardModulatedSTDPSynapse from former test_synapses_stdp.py

"""Focused suite: TestRewardModulatedSTDPSynapse from former test_synapses_stdp.py."""

from __future__ import annotations

from tests.synapses_stdp_support import *  # noqa: F403


class TestRewardModulatedSTDPSynapse:
    def _make(self, w=0.5, lr=0.01, seed=42):
        return RewardModulatedSTDPSynapse(
            w_min=0.0,
            w_max=1.0,
            length=256,
            w=w,
            learning_rate=lr,
            window_size=5,
            seed=seed,
            trace_decay=0.9,
        )

    def test_construction(self):
        syn = self._make()
        assert syn.eligibility_trace == 0.0
        assert syn.trace_decay == 0.9

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("eligibility_trace", float("nan")),
            ("trace_decay", -0.1),
            ("trace_decay", 1.1),
            ("trace_decay", float("nan")),
            ("anti_hebbian_scale", -0.1),
            ("anti_hebbian_scale", float("inf")),
        ],
    )
    def test_invalid_reward_stdp_parameters_fail_closed(self, field, value):
        kwargs = {
            "w_min": 0.0,
            "w_max": 1.0,
            "length": 256,
            "w": 0.5,
            "learning_rate": 0.01,
            "window_size": 5,
            "seed": 42,
            "trace_decay": 0.9,
        }
        kwargs[field] = value
        with pytest.raises(ValueError, match=field):
            RewardModulatedSTDPSynapse(**kwargs)

    @pytest.mark.parametrize(
        ("pre_bit", "post_bit"),
        [
            (2, 0),
            (-1, 1),
            (1, 2),
            (True, 0),
            (1, False),
        ],
    )
    def test_invalid_reward_stdp_step_bits_fail_closed(self, pre_bit, post_bit):
        syn = self._make()
        with pytest.raises(ValueError, match="bit"):
            syn.process_step(pre_bit=pre_bit, post_bit=post_bit)

    @pytest.mark.parametrize("reward", [float("nan"), float("inf"), -float("inf")])
    def test_invalid_reward_signal_fails_closed(self, reward):
        syn = self._make()
        syn.process_step(pre_bit=1, post_bit=1)
        with pytest.raises(ValueError, match="reward"):
            syn.apply_reward(reward=reward)

    def test_process_step_returns_binary(self):
        syn = self._make()
        for _ in range(50):
            out = syn.process_step(pre_bit=1, post_bit=1)
            assert out in (0, 1)

    def test_eligibility_builds_on_coincidence(self):
        """Pre=1 + Post=1 should increase eligibility trace."""
        syn = self._make()
        for _ in range(20):
            syn.process_step(pre_bit=1, post_bit=1)
        assert syn.eligibility_trace > 0

    def test_eligibility_decreases_on_mismatch(self):
        """Pre=1 + Post=0 should decrease (or keep negative) eligibility trace."""
        syn = self._make()
        for _ in range(50):
            syn.process_step(pre_bit=1, post_bit=0)
        assert syn.eligibility_trace < 0

    def test_trace_decays(self):
        """After a burst, trace should decay toward 0 with no input."""
        syn = self._make()
        for _ in range(10):
            syn.process_step(pre_bit=1, post_bit=1)
        trace_after_burst = syn.eligibility_trace
        # Run steps with no coincidence (pre=0, post=0)
        for _ in range(50):
            syn.process_step(pre_bit=0, post_bit=0)
        assert abs(syn.eligibility_trace) < abs(trace_after_burst)

    def test_positive_reward_increases_weight(self):
        """Positive eligibility + positive reward should increase weight."""
        syn = self._make(w=0.5, lr=0.05)
        # Build positive eligibility
        for _ in range(30):
            syn.process_step(pre_bit=1, post_bit=1)
        assert syn.eligibility_trace > 0
        w_before = syn.w
        syn.apply_reward(reward=1.0)
        assert syn.w > w_before

    def test_negative_reward_decreases_weight(self):
        """Positive eligibility + negative reward should decrease weight."""
        syn = self._make(w=0.5, lr=0.05)
        for _ in range(30):
            syn.process_step(pre_bit=1, post_bit=1)
        assert syn.eligibility_trace > 0
        w_before = syn.w
        syn.apply_reward(reward=-1.0)
        assert syn.w < w_before

    def test_zero_reward_no_change(self):
        """Zero reward should not change weight."""
        syn = self._make(w=0.5, lr=0.05)
        for _ in range(30):
            syn.process_step(pre_bit=1, post_bit=1)
        w_before = syn.w
        syn.apply_reward(reward=0.0)
        assert syn.w == pytest.approx(w_before)

    def test_reward_respects_weight_bounds(self):
        """Weight should stay in [w_min, w_max] after reward."""
        syn = self._make(w=0.99, lr=0.5)
        for _ in range(50):
            syn.process_step(pre_bit=1, post_bit=1)
        syn.apply_reward(reward=10.0)
        assert syn.w <= syn.w_max

        syn2 = self._make(w=0.01, lr=0.5)
        for _ in range(50):
            syn2.process_step(pre_bit=1, post_bit=1)
        syn2.apply_reward(reward=-10.0)
        assert syn2.w >= syn2.w_min

    def test_no_weight_change_without_reward(self):
        """process_step should NOT change weight (only eligibility)."""
        syn = self._make(w=0.5)
        for _ in range(100):
            syn.process_step(pre_bit=1, post_bit=1)
        # Weight should be unchanged (R-STDP only updates on apply_reward)
        assert syn.w == pytest.approx(0.5)
