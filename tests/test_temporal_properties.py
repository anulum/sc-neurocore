# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Tests for sc_neurocore.verification.temporal_properties

"""Real-surface tests for temporal spike-train property verification."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from sc_neurocore.verification.temporal_properties import (
    PropertyResult,
    bounded_activity,
    causal_order,
    fires_within,
    mutual_exclusion,
    rate_bound,
    refractory_guarantee,
)


def _make_spikes(T: int = 50, N: int = 5) -> npt.NDArray[np.int8]:
    """Create an empty binary spike raster with shape ``(T, N)``."""
    return np.zeros((T, N), dtype=np.int8)


class TestFiresWithin:
    """Response-latency checks after explicit stimulus times."""

    def test_verified(self) -> None:
        """A response inside the latency window verifies the property."""
        s = _make_spikes()
        s[12, 0] = 1  # responds 2 steps after stimulus at t=10
        r = fires_within(s, neuron_id=0, stimulus_times=[10], max_latency=5)
        assert r.result == PropertyResult.VERIFIED

    def test_violated(self) -> None:
        """A missing response returns a counterexample at the stimulus time."""
        s = _make_spikes()
        # No response
        r = fires_within(s, neuron_id=0, stimulus_times=[10], max_latency=5)
        assert r.result == PropertyResult.VIOLATED
        assert r.counterexample is not None
        assert r.counterexample.timestep == 10

    def test_multiple_stimuli(self) -> None:
        """Multiple stimuli all require responses inside their latency windows."""
        s = _make_spikes()
        s[12, 0] = 1
        s[22, 0] = 1
        r = fires_within(s, neuron_id=0, stimulus_times=[10, 20], max_latency=5)
        assert r.result == PropertyResult.VERIFIED

    def test_summary(self) -> None:
        """Violation summaries include the fail status marker."""
        s = _make_spikes()
        r = fires_within(s, neuron_id=0, stimulus_times=[10], max_latency=5)
        assert "FAIL" in r.summary()


class TestMutualExclusion:
    """Mutual-exclusion checks over neuron subsets."""

    def test_verified(self) -> None:
        """Separated spikes in the checked set satisfy mutual exclusion."""
        s = _make_spikes()
        s[5, 0] = 1
        s[10, 1] = 1
        r = mutual_exclusion(s, neuron_set=[0, 1])
        assert r.result == PropertyResult.VERIFIED

    def test_violated(self) -> None:
        """Co-firing neurons produce a counterexample with both neuron IDs."""
        s = _make_spikes()
        s[5, 0] = 1
        s[5, 1] = 1
        r = mutual_exclusion(s, neuron_set=[0, 1])
        assert r.result == PropertyResult.VIOLATED
        assert r.counterexample is not None
        assert r.counterexample.timestep == 5
        assert set(r.counterexample.neuron_ids) == {0, 1}

    def test_three_neurons(self) -> None:
        """The checked subset may include more than two neurons."""
        s = _make_spikes()
        s[5, 0] = 1
        s[5, 2] = 1
        r = mutual_exclusion(s, neuron_set=[0, 1, 2])
        assert r.result == PropertyResult.VIOLATED


class TestRateBound:
    """Sliding-window firing-rate safety bound checks."""

    def test_verified(self) -> None:
        """Sparse spikes remain below the configured rate bound."""
        s = _make_spikes()
        s[10, 0] = 1
        s[30, 0] = 1
        r = rate_bound(s, neuron_id=0, max_rate=0.5, window_size=10)
        assert r.result == PropertyResult.VERIFIED

    def test_violated(self) -> None:
        """A dense burst inside one window violates the rate bound."""
        s = _make_spikes()
        s[10:18, 0] = 1  # 8 spikes in 10-step window = rate 0.8
        r = rate_bound(s, neuron_id=0, max_rate=0.5, window_size=10)
        assert r.result == PropertyResult.VIOLATED


class TestRefractoryGuarantee:
    """Minimum inter-spike interval checks for one neuron."""

    def test_verified(self) -> None:
        """Spikes separated by at least ``min_gap`` verify the property."""
        s = _make_spikes()
        s[10, 0] = 1
        s[20, 0] = 1
        r = refractory_guarantee(s, neuron_id=0, min_gap=5)
        assert r.result == PropertyResult.VERIFIED

    def test_violated(self) -> None:
        """A too-short inter-spike interval returns the first spike time."""
        s = _make_spikes()
        s[10, 0] = 1
        s[12, 0] = 1  # gap = 2 < min_gap = 5
        r = refractory_guarantee(s, neuron_id=0, min_gap=5)
        assert r.result == PropertyResult.VIOLATED
        assert r.counterexample is not None
        assert r.counterexample.timestep == 10

    def test_no_spikes(self) -> None:
        """Silent neurons vacuously satisfy the refractory guarantee."""
        s = _make_spikes()
        r = refractory_guarantee(s, neuron_id=0, min_gap=5)
        assert r.result == PropertyResult.VERIFIED


class TestCausalOrder:
    """Causal-order checks between source and target neuron spikes."""

    def test_verified(self) -> None:
        """A source spike before each target spike verifies causal order."""
        s = _make_spikes()
        s[8, 0] = 1  # A fires at t=8
        s[10, 1] = 1  # B fires at t=10 (within 5 steps of A)
        r = causal_order(s, neuron_a=0, neuron_b=1, max_delay=5)
        assert r.result == PropertyResult.VERIFIED

    def test_violated(self) -> None:
        """A target spike without a recent source spike violates causal order."""
        s = _make_spikes()
        s[10, 1] = 1  # B fires, A never fires
        r = causal_order(s, neuron_a=0, neuron_b=1, max_delay=5)
        assert r.result == PropertyResult.VIOLATED

    def test_no_b_spikes(self) -> None:
        """No target spikes make the implication vacuously true."""
        s = _make_spikes()
        s[5, 0] = 1  # A fires but B never does → vacuously true
        r = causal_order(s, neuron_a=0, neuron_b=1, max_delay=5)
        assert r.result == PropertyResult.VERIFIED


class TestBoundedActivity:
    """Bounded total activity checks over neuron subsets."""

    def test_verified(self) -> None:
        """Activity within the total-spike bound verifies the property."""
        s = _make_spikes()
        s[10, 0] = 1
        s[20, 1] = 1
        r = bounded_activity(s, neuron_set=[0, 1, 2], window_size=10, max_total_spikes=3)
        assert r.result == PropertyResult.VERIFIED

    def test_violated(self) -> None:
        """A high-activity window violates the total-spike bound."""
        s = _make_spikes()
        s[10:15, 0] = 1
        s[10:15, 1] = 1
        r = bounded_activity(s, neuron_set=[0, 1], window_size=10, max_total_spikes=5)
        assert r.result == PropertyResult.VIOLATED

    def test_summary_pass(self) -> None:
        """Verified bounded-activity results include the pass status marker."""
        s = _make_spikes()
        r = bounded_activity(s, neuron_set=[0, 1], window_size=5, max_total_spikes=10)
        assert "PASS" in r.summary()
