"""
Tests for StochasticSTDPSynapse and RewardModulatedSTDPSynapse.
Covers the untested code paths in stochastic_stdp.py and r_stdp.py.
"""

import pytest
import numpy as np

from sc_neurocore.synapses.stochastic_stdp import StochasticSTDPSynapse
from sc_neurocore.synapses.r_stdp import RewardModulatedSTDPSynapse


# ---------------------------------------------------------------------------
# StochasticSTDPSynapse
# ---------------------------------------------------------------------------

class TestStochasticSTDPSynapse:
    def _make(self, w=0.5, lr=0.01, seed=42):
        return StochasticSTDPSynapse(
            w_min=0.0, w_max=1.0, length=256, w=w,
            learning_rate=lr, window_size=5, seed=seed,
        )

    def test_construction(self):
        syn = self._make()
        assert syn.w == pytest.approx(0.5)
        assert syn._pre_trace.shape == (5,)
        assert np.all(syn._pre_trace == 0)

    def test_process_step_returns_binary(self):
        syn = self._make()
        for _ in range(100):
            out = syn.process_step(pre_bit=1, post_bit=0)
            assert out in (0, 1)

    def test_output_is_and_of_pre_and_weight(self):
        """Output is pre_bit AND weight_bit; when pre=0, output must be 0."""
        syn = self._make()
        for _ in range(50):
            assert syn.process_step(pre_bit=0, post_bit=1) == 0

    def test_ltp_increases_weight(self):
        """Sustained pre=1, post=1 should increase weight over many steps."""
        syn = self._make(w=0.3, lr=0.05, seed=0)
        initial_w = syn.w
        for _ in range(500):
            syn.process_step(pre_bit=1, post_bit=1)
        assert syn.w > initial_w

    def test_ltd_decreases_weight(self):
        """Sustained pre=1, post=0 should decrease weight over many steps."""
        syn = self._make(w=0.7, lr=0.05, seed=0)
        initial_w = syn.w
        for _ in range(500):
            syn.process_step(pre_bit=1, post_bit=0)
        assert syn.w < initial_w

    def test_weight_stays_in_bounds(self):
        """Weight should never exceed [w_min, w_max] regardless of input."""
        syn = self._make(w=0.99, lr=0.1, seed=0)
        for _ in range(1000):
            syn.process_step(pre_bit=1, post_bit=1)
        assert syn.w <= syn.w_max

        syn2 = self._make(w=0.01, lr=0.1, seed=0)
        for _ in range(1000):
            syn2.process_step(pre_bit=1, post_bit=0)
        assert syn2.w >= syn2.w_min

    def test_potentiate_directly(self):
        syn = self._make(w=0.5)
        syn._potentiate()
        assert syn.w > 0.5

    def test_depress_directly(self):
        syn = self._make(w=0.5)
        syn._depress()
        assert syn.w < 0.5

    def test_pre_trace_shifts(self):
        """Pre-trace buffer should shift bits in correctly."""
        syn = self._make()
        syn.process_step(pre_bit=1, post_bit=0)
        assert syn._pre_trace[0] == 1
        syn.process_step(pre_bit=0, post_bit=0)
        assert syn._pre_trace[0] == 0
        assert syn._pre_trace[1] == 1  # shifted


# ---------------------------------------------------------------------------
# RewardModulatedSTDPSynapse
# ---------------------------------------------------------------------------

class TestRewardModulatedSTDPSynapse:
    def _make(self, w=0.5, lr=0.01, seed=42):
        return RewardModulatedSTDPSynapse(
            w_min=0.0, w_max=1.0, length=256, w=w,
            learning_rate=lr, window_size=5, seed=seed,
            trace_decay=0.9,
        )

    def test_construction(self):
        syn = self._make()
        assert syn.eligibility_trace == 0.0
        assert syn.trace_decay == 0.9

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
