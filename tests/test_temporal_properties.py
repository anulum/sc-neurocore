# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Tests for sc_neurocore.verification.temporal_properties

from __future__ import annotations

import numpy as np

from sc_neurocore.verification.temporal_properties import (
    fires_within,
    mutual_exclusion,
    rate_bound,
    refractory_guarantee,
    causal_order,
    bounded_activity,
    PropertyResult,
)


def _make_spikes(T=50, N=5):
    return np.zeros((T, N), dtype=np.int8)


class TestFiresWithin:
    def test_verified(self):
        s = _make_spikes()
        s[12, 0] = 1  # responds 2 steps after stimulus at t=10
        r = fires_within(s, neuron_id=0, stimulus_times=[10], max_latency=5)
        assert r.result == PropertyResult.VERIFIED

    def test_violated(self):
        s = _make_spikes()
        # No response
        r = fires_within(s, neuron_id=0, stimulus_times=[10], max_latency=5)
        assert r.result == PropertyResult.VIOLATED
        assert r.counterexample is not None
        assert r.counterexample.timestep == 10

    def test_multiple_stimuli(self):
        s = _make_spikes()
        s[12, 0] = 1
        s[22, 0] = 1
        r = fires_within(s, neuron_id=0, stimulus_times=[10, 20], max_latency=5)
        assert r.result == PropertyResult.VERIFIED

    def test_summary(self):
        s = _make_spikes()
        r = fires_within(s, neuron_id=0, stimulus_times=[10], max_latency=5)
        assert "FAIL" in r.summary()


class TestMutualExclusion:
    def test_verified(self):
        s = _make_spikes()
        s[5, 0] = 1
        s[10, 1] = 1
        r = mutual_exclusion(s, neuron_set=[0, 1])
        assert r.result == PropertyResult.VERIFIED

    def test_violated(self):
        s = _make_spikes()
        s[5, 0] = 1
        s[5, 1] = 1
        r = mutual_exclusion(s, neuron_set=[0, 1])
        assert r.result == PropertyResult.VIOLATED
        assert r.counterexample.timestep == 5
        assert set(r.counterexample.neuron_ids) == {0, 1}

    def test_three_neurons(self):
        s = _make_spikes()
        s[5, 0] = 1
        s[5, 2] = 1
        r = mutual_exclusion(s, neuron_set=[0, 1, 2])
        assert r.result == PropertyResult.VIOLATED


class TestRateBound:
    def test_verified(self):
        s = _make_spikes()
        s[10, 0] = 1
        s[30, 0] = 1
        r = rate_bound(s, neuron_id=0, max_rate=0.5, window_size=10)
        assert r.result == PropertyResult.VERIFIED

    def test_violated(self):
        s = _make_spikes()
        s[10:18, 0] = 1  # 8 spikes in 10-step window = rate 0.8
        r = rate_bound(s, neuron_id=0, max_rate=0.5, window_size=10)
        assert r.result == PropertyResult.VIOLATED


class TestRefractoryGuarantee:
    def test_verified(self):
        s = _make_spikes()
        s[10, 0] = 1
        s[20, 0] = 1
        r = refractory_guarantee(s, neuron_id=0, min_gap=5)
        assert r.result == PropertyResult.VERIFIED

    def test_violated(self):
        s = _make_spikes()
        s[10, 0] = 1
        s[12, 0] = 1  # gap = 2 < min_gap = 5
        r = refractory_guarantee(s, neuron_id=0, min_gap=5)
        assert r.result == PropertyResult.VIOLATED
        assert r.counterexample.timestep == 10

    def test_no_spikes(self):
        s = _make_spikes()
        r = refractory_guarantee(s, neuron_id=0, min_gap=5)
        assert r.result == PropertyResult.VERIFIED


class TestCausalOrder:
    def test_verified(self):
        s = _make_spikes()
        s[8, 0] = 1  # A fires at t=8
        s[10, 1] = 1  # B fires at t=10 (within 5 steps of A)
        r = causal_order(s, neuron_a=0, neuron_b=1, max_delay=5)
        assert r.result == PropertyResult.VERIFIED

    def test_violated(self):
        s = _make_spikes()
        s[10, 1] = 1  # B fires, A never fires
        r = causal_order(s, neuron_a=0, neuron_b=1, max_delay=5)
        assert r.result == PropertyResult.VIOLATED

    def test_no_b_spikes(self):
        s = _make_spikes()
        s[5, 0] = 1  # A fires but B never does → vacuously true
        r = causal_order(s, neuron_a=0, neuron_b=1, max_delay=5)
        assert r.result == PropertyResult.VERIFIED


class TestBoundedActivity:
    def test_verified(self):
        s = _make_spikes()
        s[10, 0] = 1
        s[20, 1] = 1
        r = bounded_activity(s, neuron_set=[0, 1, 2], window_size=10, max_total_spikes=3)
        assert r.result == PropertyResult.VERIFIED

    def test_violated(self):
        s = _make_spikes()
        s[10:15, 0] = 1
        s[10:15, 1] = 1
        r = bounded_activity(s, neuron_set=[0, 1], window_size=10, max_total_spikes=5)
        assert r.result == PropertyResult.VIOLATED

    def test_summary_pass(self):
        s = _make_spikes()
        r = bounded_activity(s, neuron_set=[0, 1], window_size=5, max_total_spikes=10)
        assert "PASS" in r.summary()
