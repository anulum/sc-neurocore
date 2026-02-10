"""
Tests for the 6 untested core utils modules:
  - adaptive.py (AdaptiveInference)
  - connectomes.py (ConnectomeGenerator)
  - decorrelators.py (ShufflingDecorrelator, LFSRRegenDecorrelator)
  - fault_injection.py (FaultInjector)
  - fsm_activations.py (TanhFSM, ReLKFSM)
  - model_bridge.py (normalize_weights, SCBridge)
"""

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# adaptive.py
# ---------------------------------------------------------------------------
from sc_neurocore.utils.adaptive import AdaptiveInference


class TestAdaptiveInference:
    def test_converges_early(self):
        """Stable step_func should trigger early exit before max_length."""
        ai = AdaptiveInference(
            check_interval=16, tolerance=0.02, min_length=64, max_length=1024
        )
        call_count = 0

        def step():
            nonlocal call_count
            call_count += 1
            return 0.5  # perfectly stable

        result = ai.run_adaptive(step)
        assert result == pytest.approx(0.5)
        # Must exit well before max_length (3 checks after min_length)
        assert call_count < 1024

    def test_runs_to_max_length_when_noisy(self):
        """Noisy step_func should exhaust max_length."""
        rng = np.random.default_rng(42)
        ai = AdaptiveInference(
            check_interval=16, tolerance=0.001, min_length=64, max_length=256
        )
        call_count = 0

        def step():
            nonlocal call_count
            call_count += 1
            return rng.uniform(0.0, 1.0)

        ai.run_adaptive(step)
        assert call_count == 256

    def test_min_length_respected(self):
        """Even a constant function must run at least min_length steps."""
        ai = AdaptiveInference(
            check_interval=16, tolerance=0.1, min_length=128, max_length=512
        )
        call_count = 0

        def step():
            nonlocal call_count
            call_count += 1
            return 0.42

        ai.run_adaptive(step)
        assert call_count >= 128

    def test_returns_last_value(self):
        """Return value should be the last estimate from step_func."""
        counter = [0]

        def step():
            counter[0] += 1
            return float(counter[0])

        ai = AdaptiveInference(
            check_interval=32, tolerance=100.0, min_length=64, max_length=128
        )
        result = ai.run_adaptive(step)
        # With huge tolerance and checks starting at min_length=64,
        # should converge early once 3 checks accumulate
        assert result > 0


# ---------------------------------------------------------------------------
# connectomes.py
# ---------------------------------------------------------------------------
from sc_neurocore.utils.connectomes import ConnectomeGenerator


class TestConnectomeGenerator:
    def test_watts_strogatz_shape(self):
        adj = ConnectomeGenerator.generate_watts_strogatz(20, 4, 0.0)
        assert adj.shape == (20, 20)

    def test_watts_strogatz_no_self_loops(self):
        adj = ConnectomeGenerator.generate_watts_strogatz(20, 4, 0.0)
        assert np.all(np.diag(adj) == 0)

    def test_watts_strogatz_regular_ring(self):
        """With p_rewire=0, each node connects to k/2 forward neighbors."""
        np.random.seed(0)
        adj = ConnectomeGenerator.generate_watts_strogatz(10, 4, 0.0)
        # Each row should have at least 2 outgoing edges (k/2 = 2)
        row_sums = adj.sum(axis=1)
        assert np.all(row_sums >= 2)

    def test_watts_strogatz_full_rewire(self):
        """With p_rewire=1.0, graph is random but still connected."""
        np.random.seed(42)
        adj = ConnectomeGenerator.generate_watts_strogatz(10, 4, 1.0)
        # Should still have edges
        assert adj.sum() > 0
        # No self-loops
        assert np.all(np.diag(adj) == 0)

    def test_watts_strogatz_k_ge_n(self):
        """When k >= n, should return all-to-all minus diagonal."""
        adj = ConnectomeGenerator.generate_watts_strogatz(5, 5, 0.0)
        expected = np.ones((5, 5)) - np.eye(5)
        np.testing.assert_array_equal(adj, expected)

    def test_scale_free_shape(self):
        np.random.seed(0)
        adj = ConnectomeGenerator.generate_scale_free(20)
        assert adj.shape == (20, 20)

    def test_scale_free_edge_count(self):
        """Initial bidirectional edge (2 entries) + 13 directed edges = 15."""
        np.random.seed(0)
        adj = ConnectomeGenerator.generate_scale_free(15)
        # Initial: adj[0,1]=1 and adj[1,0]=1 (2 entries).
        # Nodes 2..14 each add 1 directed edge = 13 more.
        assert adj.sum() == 15

    def test_scale_free_no_self_loops(self):
        np.random.seed(0)
        adj = ConnectomeGenerator.generate_scale_free(15)
        assert np.all(np.diag(adj) == 0)


# ---------------------------------------------------------------------------
# decorrelators.py
# ---------------------------------------------------------------------------
from sc_neurocore.utils.decorrelators import (
    ShufflingDecorrelator,
    LFSRRegenDecorrelator,
)


class TestShufflingDecorrelator:
    def test_preserves_bit_count(self):
        """Shuffling must keep exact same number of ones."""
        bs = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0], dtype=np.uint8)
        dec = ShufflingDecorrelator(window_size=8, seed=42)
        result = dec.process(bs)
        assert result.sum() == bs.sum()
        assert len(result) == len(bs)

    def test_preserves_probability(self):
        """On a long bitstream, probability should be unchanged."""
        rng = np.random.default_rng(7)
        bs = (rng.random(1024) < 0.3).astype(np.uint8)
        dec = ShufflingDecorrelator(window_size=16, seed=99)
        result = dec.process(bs)
        assert result.mean() == pytest.approx(bs.mean(), abs=1e-12)

    def test_output_length_preserved_non_divisible(self):
        """Length not divisible by window_size should still return correct length."""
        bs = np.ones(100, dtype=np.uint8)
        dec = ShufflingDecorrelator(window_size=16, seed=1)
        result = dec.process(bs)
        assert len(result) == 100

    def test_changes_bit_positions(self):
        """Shuffled output should differ from input in at least some positions."""
        rng = np.random.default_rng(0)
        bs = (rng.random(256) < 0.5).astype(np.uint8)
        dec = ShufflingDecorrelator(window_size=16, seed=42)
        result = dec.process(bs)
        # With p=0.5 and window=16, very unlikely to be identical
        assert not np.array_equal(result, bs)


class TestLFSRRegenDecorrelator:
    def test_preserves_approximate_probability(self):
        """Regenerated stream should have approximately the same probability."""
        rng = np.random.default_rng(0)
        bs = (rng.random(2048) < 0.4).astype(np.uint8)
        dec = LFSRRegenDecorrelator(seed=42)
        result = dec.process(bs)
        assert result.mean() == pytest.approx(bs.mean(), abs=0.05)
        assert len(result) == len(bs)

    def test_output_is_binary(self):
        bs = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
        dec = LFSRRegenDecorrelator(seed=0)
        result = dec.process(bs)
        assert set(np.unique(result)).issubset({0, 1})

    def test_produces_different_sequence(self):
        """Regenerated bitstream should not be identical to input."""
        rng = np.random.default_rng(0)
        bs = (rng.random(512) < 0.5).astype(np.uint8)
        dec = LFSRRegenDecorrelator(seed=77)
        result = dec.process(bs)
        # With different RNG seed, almost certainly different
        assert not np.array_equal(result, bs)


# ---------------------------------------------------------------------------
# fault_injection.py
# ---------------------------------------------------------------------------
from sc_neurocore.utils.fault_injection import FaultInjector


class TestFaultInjector:
    def test_bit_flip_zero_rate(self):
        """Error rate 0 should return unchanged bitstream."""
        bs = np.array([0, 1, 0, 1, 1, 0], dtype=np.uint8)
        result = FaultInjector.inject_bit_flips(bs, 0.0)
        np.testing.assert_array_equal(result, bs)

    def test_bit_flip_full_rate(self):
        """Error rate 1.0 should flip every bit."""
        bs = np.array([0, 1, 0, 1, 1, 0], dtype=np.uint8)
        np.random.seed(0)
        result = FaultInjector.inject_bit_flips(bs, 1.0)
        expected = 1 - bs
        np.testing.assert_array_equal(result, expected)

    def test_bit_flip_output_binary(self):
        """Output must be strictly 0/1."""
        np.random.seed(0)
        bs = np.zeros(1000, dtype=np.uint8)
        result = FaultInjector.inject_bit_flips(bs, 0.5)
        assert set(np.unique(result)).issubset({0, 1})

    def test_bit_flip_approximate_rate(self):
        """Fraction of flipped bits should roughly match error_rate."""
        np.random.seed(42)
        bs = np.zeros(10000, dtype=np.uint8)
        result = FaultInjector.inject_bit_flips(bs, 0.1)
        flip_rate = result.mean()  # started all zeros, flips become ones
        assert flip_rate == pytest.approx(0.1, abs=0.02)

    def test_stuck_at_zero(self):
        """Stuck-at-0 forces selected bits to 0."""
        np.random.seed(0)
        bs = np.ones(1000, dtype=np.uint8)
        result = FaultInjector.inject_stuck_at(bs, 0.3, value=0)
        # Some bits should now be 0
        assert result.sum() < 1000
        # Rate should be approximately 30% zeros
        zero_frac = 1.0 - result.mean()
        assert zero_frac == pytest.approx(0.3, abs=0.05)

    def test_stuck_at_one(self):
        """Stuck-at-1 forces selected bits to 1."""
        np.random.seed(0)
        bs = np.zeros(1000, dtype=np.uint8)
        result = FaultInjector.inject_stuck_at(bs, 0.3, value=1)
        one_frac = result.mean()
        assert one_frac == pytest.approx(0.3, abs=0.05)

    def test_stuck_at_preserves_shape(self):
        bs = np.zeros((4, 5), dtype=np.uint8)
        result = FaultInjector.inject_stuck_at(bs, 0.5, value=1)
        assert result.shape == (4, 5)


# ---------------------------------------------------------------------------
# fsm_activations.py
# ---------------------------------------------------------------------------
from sc_neurocore.utils.fsm_activations import TanhFSM, ReLKFSM


class TestTanhFSM:
    def test_initial_state(self):
        fsm = TanhFSM(states=16)
        assert fsm.state == 8  # N/2

    def test_all_ones_saturates_high(self):
        """All-ones input should drive state to max and output 1."""
        fsm = TanhFSM(states=8)
        bs = np.ones(100, dtype=np.uint8)
        out = fsm.process(bs)
        # After enough 1s, state saturates at 7, output always 1
        assert out[-1] == 1
        # Last 50 outputs should all be 1
        assert np.all(out[-50:] == 1)

    def test_all_zeros_saturates_low(self):
        """All-zeros input should drive state to 0 and output 0."""
        fsm = TanhFSM(states=8)
        bs = np.zeros(100, dtype=np.uint8)
        out = fsm.process(bs)
        assert out[-1] == 0
        assert np.all(out[-50:] == 0)

    def test_balanced_input(self):
        """p=0.5 input should give ~0.5 output probability."""
        rng = np.random.default_rng(42)
        fsm = TanhFSM(states=16)
        bs = (rng.random(4096) < 0.5).astype(np.uint8)
        out = fsm.process(bs)
        # Output probability should be near 0.5
        assert out.mean() == pytest.approx(0.5, abs=0.1)

    def test_high_input_bias(self):
        """High-probability input should produce high-probability output."""
        rng = np.random.default_rng(0)
        fsm = TanhFSM(states=16)
        bs = (rng.random(4096) < 0.9).astype(np.uint8)
        out = fsm.process(bs)
        assert out.mean() > 0.7

    def test_process_returns_correct_length(self):
        fsm = TanhFSM(states=8)
        bs = np.ones(123, dtype=np.uint8)
        out = fsm.process(bs)
        assert len(out) == 123

    def test_output_is_binary(self):
        rng = np.random.default_rng(0)
        fsm = TanhFSM(states=8)
        bs = (rng.random(200) < 0.6).astype(np.uint8)
        out = fsm.process(bs)
        assert set(np.unique(out)).issubset({0, 1})


class TestReLKFSM:
    def test_initial_state_zero(self):
        fsm = ReLKFSM(states=16)
        assert fsm.state == 0

    def test_all_zeros_stays_off(self):
        """All-zeros input keeps state at 0, output always 0."""
        fsm = ReLKFSM(states=8)
        bs = np.zeros(100, dtype=np.uint8)
        out = fsm.process(bs)
        assert np.all(out == 0)

    def test_all_ones_turns_on(self):
        """All-ones input drives state above 0, output becomes 1."""
        fsm = ReLKFSM(states=8)
        bs = np.ones(100, dtype=np.uint8)
        out = fsm.process(bs)
        # First step: state goes to 1, output is 1
        assert out[0] == 1
        assert np.all(out == 1)

    def test_relu_like_behavior(self):
        """Low input (p<0.5) should produce lower output than high input (p>0.5)."""
        rng = np.random.default_rng(42)
        fsm_low = ReLKFSM(states=16)
        fsm_high = ReLKFSM(states=16)
        bs_low = (rng.random(4096) < 0.2).astype(np.uint8)
        bs_high = (rng.random(4096) < 0.8).astype(np.uint8)
        out_low = fsm_low.process(bs_low)
        out_high = fsm_high.process(bs_high)
        assert out_high.mean() > out_low.mean()

    def test_step_transitions(self):
        """Verify individual step state transitions."""
        fsm = ReLKFSM(states=4)
        # state=0, input 0 -> state stays 0, output 0
        assert fsm.step(0) == 0
        assert fsm.state == 0
        # state=0, input 1 -> state 1, output 1
        assert fsm.step(1) == 1
        assert fsm.state == 1
        # state=1, input 0 -> state 0, output 0
        assert fsm.step(0) == 0
        assert fsm.state == 0


# ---------------------------------------------------------------------------
# model_bridge.py
# ---------------------------------------------------------------------------
from sc_neurocore.utils.model_bridge import normalize_weights, SCBridge


class TestNormalizeWeights:
    def test_basic_normalization(self):
        w = np.array([0.0, 5.0, 10.0])
        result = normalize_weights(w)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_negative_weights(self):
        w = np.array([-10.0, 0.0, 10.0])
        result = normalize_weights(w)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])

    def test_uniform_weights(self):
        """All equal weights should map to 0.5."""
        w = np.array([3.0, 3.0, 3.0])
        result = normalize_weights(w)
        np.testing.assert_allclose(result, [0.5, 0.5, 0.5])

    def test_single_element(self):
        w = np.array([7.0])
        result = normalize_weights(w)
        assert result[0] == 0.5

    def test_2d_array(self):
        w = np.array([[0.0, 1.0], [2.0, 3.0]])
        result = normalize_weights(w)
        assert result.min() == 0.0
        assert result.max() == 1.0
        assert result.shape == (2, 2)


class TestSCBridge:
    def _make_mock_layer(self, shape):
        """Create a simple mock layer with weights attribute."""

        class MockLayer:
            def __init__(self, shape):
                self.weights = np.zeros(shape)

        return MockLayer(shape)

    def test_load_matching_shapes(self, capsys):
        layer = self._make_mock_layer((3, 4))
        state_dict = {"fc1.weight": np.random.randn(3, 4)}
        layer_mapping = {"fc1": layer}
        SCBridge.load_from_state_dict(state_dict, layer_mapping)
        # Weights should be normalized to [0, 1]
        assert layer.weights.min() >= 0.0
        assert layer.weights.max() <= 1.0

    def test_load_shape_mismatch(self, caplog):
        layer = self._make_mock_layer((3, 4))
        state_dict = {"fc1.weight": np.random.randn(5, 6)}
        layer_mapping = {"fc1": layer}
        with caplog.at_level("WARNING", logger="sc_neurocore.utils.model_bridge"):
            SCBridge.load_from_state_dict(state_dict, layer_mapping)
        assert "Shape mismatch" in caplog.text
        # Weights should NOT have changed
        np.testing.assert_array_equal(layer.weights, np.zeros((3, 4)))

    def test_load_missing_key(self, caplog):
        layer = self._make_mock_layer((3, 4))
        state_dict = {"other.weight": np.random.randn(3, 4)}
        layer_mapping = {"fc1": layer}
        with caplog.at_level("DEBUG", logger="sc_neurocore.utils.model_bridge"):
            SCBridge.load_from_state_dict(state_dict, layer_mapping)
        assert "No weights found" in caplog.text

    def test_export_to_numpy(self):
        class MockLayerWithGetWeights:
            def get_weights(self):
                return np.ones((2, 3))

        class MockLayerWithWeights:
            weights = np.zeros((4, 5))

        layers = {
            "layer_a": MockLayerWithGetWeights(),
            "layer_b": MockLayerWithWeights(),
        }
        state = SCBridge.export_to_numpy(layers)
        assert "layer_a.weight" in state
        assert "layer_b.weight" in state
        np.testing.assert_array_equal(state["layer_a.weight"], np.ones((2, 3)))
        np.testing.assert_array_equal(state["layer_b.weight"], np.zeros((4, 5)))

    def test_export_empty(self):
        state = SCBridge.export_to_numpy({})
        assert state == {}

    def test_load_triggers_refresh(self, capsys):
        """If layer has _refresh_packed_weights, it should be called."""
        refreshed = [False]

        class MockLayerRefresh:
            def __init__(self):
                self.weights = np.zeros((2, 3))

            def _refresh_packed_weights(self):
                refreshed[0] = True

        layer = MockLayerRefresh()
        state_dict = {"fc1.weight": np.random.randn(2, 3)}
        SCBridge.load_from_state_dict(state_dict, {"fc1": layer})
        assert refreshed[0] is True
