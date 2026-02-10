"""Phase 10 acceptance tests: SIMD pack, LIF optimization, rayon guard."""

from __future__ import annotations

import numpy as np
import pytest

import sc_neurocore_engine as v3


class TestSIMDPack:
    """Test SIMD-accelerated pack_bitstream_numpy correctness."""

    def test_pack_numpy_matches_list_pack(self):
        """SIMD pack must produce identical output to list pack."""
        rng = np.random.RandomState(42)
        bits = rng.randint(0, 2, 10_000).astype(np.uint8)
        packed_list = v3.pack_bitstream(bits.tolist())
        packed_numpy = np.asarray(v3.pack_bitstream_numpy(bits))
        np.testing.assert_array_equal(packed_list, packed_numpy)

    @pytest.mark.parametrize("length", [1, 63, 64, 65, 127, 128, 256, 1024, 4096])
    def test_pack_numpy_various_lengths(self, length):
        """SIMD pack handles all lengths including non-aligned."""
        rng = np.random.RandomState(42)
        bits = rng.randint(0, 2, length).astype(np.uint8)
        packed_list = v3.pack_bitstream(bits.tolist())
        packed_numpy = np.asarray(v3.pack_bitstream_numpy(bits))
        np.testing.assert_array_equal(packed_list, packed_numpy)

    def test_pack_numpy_deterministic(self):
        """Same input -> same output."""
        bits = np.array([1, 0, 1, 1, 0, 0, 1, 0] * 128, dtype=np.uint8)
        a = np.asarray(v3.pack_bitstream_numpy(bits))
        b = np.asarray(v3.pack_bitstream_numpy(bits))
        np.testing.assert_array_equal(a, b)

    def test_pack_unpack_roundtrip(self):
        """Pack->unpack roundtrip preserves bits."""
        rng = np.random.RandomState(42)
        bits = rng.randint(0, 2, 2048).astype(np.uint8)
        packed = v3.pack_bitstream_numpy(bits)
        unpacked = v3.unpack_bitstream_numpy(packed, len(bits))
        np.testing.assert_array_equal(bits, np.asarray(unpacked))


class TestBranchlessLIF:
    """Test that branchless LIF step produces identical results."""

    def test_100_steps_constant_input(self):
        """Standard equivalence: same as equivalence suite."""
        lif = v3.FixedPointLif()
        results = []
        for _ in range(100):
            s, v = lif.step(20, 256, 128, 0)
            results.append((s, v))
        assert len(results) == 100
        for s, v in results:
            assert s in (0, 1)
            assert isinstance(v, (int, np.integer))

    def test_batch_matches_step_by_step(self):
        """batch_lif_run must match step-by-step execution."""
        lif = v3.FixedPointLif()
        step_spikes, step_voltages = [], []
        for _ in range(1000):
            s, v = lif.step(20, 256, 128, 0)
            step_spikes.append(s)
            step_voltages.append(v)

        batch_spikes, batch_voltages = v3.batch_lif_run(1000, 20, 256, 128)
        np.testing.assert_array_equal(step_spikes, np.asarray(batch_spikes))
        np.testing.assert_array_equal(step_voltages, np.asarray(batch_voltages))

    def test_refractory_period(self):
        """Refractory behavior preserved under branchless mask."""
        spikes, _ = v3.batch_lif_run(200, 20, 256, 200, refractory_period=5)
        spikes_arr = np.asarray(spikes)
        spike_indices = np.where(spikes_arr == 1)[0]
        for idx in spike_indices:
            for ref_step in range(1, 6):
                if idx + ref_step < len(spikes_arr):
                    assert (
                        spikes_arr[idx + ref_step] == 0
                    ), f"Spike during refractory at step {idx + ref_step}"


class TestMultiNeuronBatch:
    """Test parallel multi-neuron LIF batch."""

    def test_shape_and_dtype(self):
        """Output shape is (n_neurons, n_steps)."""
        currents = np.full(10, 128, dtype=np.int16)
        spikes, voltages = v3.batch_lif_run_multi(10, 100, 20, 256, currents)
        assert np.asarray(spikes).shape == (10, 100)
        assert np.asarray(voltages).shape == (10, 100)

    def test_matches_sequential(self):
        """Parallel multi-neuron must match N sequential single-neuron runs."""
        n_neurons = 8
        n_steps = 500
        i_values = [64, 96, 128, 160, 192, 224, 100, 140]
        currents = np.array(i_values, dtype=np.int16)

        sequential_spikes = []
        for i_t in i_values:
            s, _ = v3.batch_lif_run(n_steps, 20, 256, i_t)
            sequential_spikes.append(np.asarray(s))

        par_spikes, _ = v3.batch_lif_run_multi(n_neurons, n_steps, 20, 256, currents)
        par_arr = np.asarray(par_spikes)

        for ni in range(n_neurons):
            np.testing.assert_array_equal(
                par_arr[ni], sequential_spikes[ni], err_msg=f"Neuron {ni} mismatch"
            )

    def test_deterministic(self):
        """Same inputs -> same outputs."""
        currents = np.full(4, 128, dtype=np.int16)
        s1, v1 = v3.batch_lif_run_multi(4, 100, 20, 256, currents)
        s2, v2 = v3.batch_lif_run_multi(4, 100, 20, 256, currents)
        np.testing.assert_array_equal(np.asarray(s1), np.asarray(s2))
        np.testing.assert_array_equal(np.asarray(v1), np.asarray(v2))


class TestRayonThreshold:
    """Test that rayon threshold does not change forward_fast outputs."""

    def test_forward_fast_determinism(self):
        """forward_fast with small inputs (below threshold) stays deterministic."""
        layer = v3.DenseLayer(16, 8, 1024)
        inputs = [0.5] * 16
        a = layer.forward_fast(inputs, seed=42)
        b = layer.forward_fast(inputs, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_forward_fast_consistent_across_sizes(self):
        """forward_fast produces valid outputs for various input sizes."""
        for n_in in [4, 16, 64, 128, 256]:
            layer = v3.DenseLayer(n_in, 8, 1024)
            inputs = [0.5] * n_in
            result = layer.forward_fast(inputs, seed=42)
            assert len(result) == 8
            for val in result:
                assert 0.0 <= val <= float(n_in), f"Out of range: {val}"


class TestPhase10Version:
    def test_version(self):
        assert v3.__version__ == "3.6.0"
