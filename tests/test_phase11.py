"""Tests for Phase 11: SIMD pipeline acceleration + zero-allocation LIF paths."""

from __future__ import annotations

import numpy as np

import sc_neurocore_engine as v3


class TestSIMDFusedAndPopcount:
    """Verify SIMD fused AND+popcount preserves dense behavior."""

    def test_dense_forward_unchanged(self):
        layer = v3.DenseLayer(8, 4, 1024, seed=42)
        inputs = [0.1, 0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8]
        out1 = layer.forward(inputs, seed=123)
        out2 = layer.forward(inputs, seed=123)
        np.testing.assert_array_equal(out1, out2)
        assert all(0.0 <= x <= 8.0 for x in out1)

    def test_dense_prepacked_unchanged(self):
        layer = v3.DenseLayer(8, 4, 1024, seed=42)
        probs = np.array([0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=1024, seed=99)
        out_legacy = layer.forward_prepacked(packed)
        out_numpy = layer.forward_prepacked_numpy(packed)
        np.testing.assert_allclose(out_numpy, out_legacy)

    def test_determinism(self):
        layer = v3.DenseLayer(16, 8, 1024, seed=42)
        inputs = [0.5] * 16
        out1 = layer.forward_fast(inputs, seed=77)
        out2 = layer.forward_fast(inputs, seed=77)
        np.testing.assert_array_equal(out1, out2)


class TestSIMDBernoulliEncode:
    """Verify SIMD Bernoulli encoder statistical correctness and determinism."""

    def test_batch_encode_statistics(self):
        probs = np.array([0.25, 0.75], dtype=np.float64)
        packed = v3.batch_encode_numpy(probs, length=10_000, seed=42)
        pc0 = sum(int(w).bit_count() for w in packed[0])
        pc1 = sum(int(w).bit_count() for w in packed[1])
        assert abs(pc0 / 10_000 - 0.25) < 0.03
        assert abs(pc1 / 10_000 - 0.75) < 0.03

    def test_batch_encode_determinism(self):
        probs = np.array([0.15, 0.35, 0.55, 0.75], dtype=np.float64)
        a = v3.batch_encode_numpy(probs, length=1024, seed=1234)
        b = v3.batch_encode_numpy(probs, length=1024, seed=1234)
        np.testing.assert_array_equal(a, b)

    def test_dense_fast_correctness(self):
        layer = v3.DenseLayer(16, 8, 1024, seed=42)
        low = np.mean(layer.forward_fast([0.1] * 16, seed=22))
        high = np.mean(layer.forward_fast([0.9] * 16, seed=22))
        assert high > low


class TestFlatWeightStorage:
    """Verify flat packed weight storage keeps API behavior unchanged."""

    def test_weight_roundtrip(self):
        layer = v3.DenseLayer(4, 3, 256, seed=42)
        weights = np.array(
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.4, 0.3, 0.2, 0.1],
                [0.5, 0.6, 0.7, 0.8],
            ],
            dtype=np.float64,
        )
        layer.set_weights(weights.tolist())
        got = np.array(layer.get_weights(), dtype=np.float64)
        np.testing.assert_allclose(got, weights)

    def test_forward_equivalence_vs_prepacked(self):
        layer = v3.DenseLayer(8, 4, 512, seed=42)
        probs = np.array([0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7], dtype=np.float64)
        seed = 31415
        packed = v3.batch_encode_numpy(probs, length=512, seed=seed)
        out_fast = np.asarray(layer.forward_fast(probs.tolist(), seed=seed), dtype=np.float64)
        out_prepacked = np.asarray(layer.forward_prepacked_numpy(packed), dtype=np.float64)
        np.testing.assert_allclose(out_fast, out_prepacked)


class TestZeroAllocLIF:
    """Verify pre-allocated LIF batch outputs stay correct."""

    def test_batch_lif_unchanged(self):
        lif = v3.FixedPointLif()
        step_spikes, step_voltages = [], []
        for _ in range(1000):
            s, v = lif.step(20, 256, 128, 0)
            step_spikes.append(s)
            step_voltages.append(v)

        batch_spikes, batch_voltages = v3.batch_lif_run(1000, 20, 256, 128)
        np.testing.assert_array_equal(step_spikes, np.asarray(batch_spikes))
        np.testing.assert_array_equal(step_voltages, np.asarray(batch_voltages))

    def test_batch_lif_multi_unchanged(self):
        n_steps = 500
        currents = np.array([64, 96, 128, 160, 192, 224, 100, 140], dtype=np.int16)
        sequential_spikes = []
        for i_t in currents:
            spikes, _ = v3.batch_lif_run(n_steps, 20, 256, int(i_t))
            sequential_spikes.append(np.asarray(spikes))

        spikes_multi, _ = v3.batch_lif_run_multi(len(currents), n_steps, 20, 256, currents)
        spikes_multi = np.asarray(spikes_multi)
        for idx in range(len(currents)):
            np.testing.assert_array_equal(spikes_multi[idx], sequential_spikes[idx])

    def test_batch_lif_multi_shape(self):
        currents = np.full(10, 128, dtype=np.int16)
        spikes, voltages = v3.batch_lif_run_multi(10, 100, 20, 256, currents)
        spikes_arr = np.asarray(spikes)
        voltages_arr = np.asarray(voltages)
        assert spikes_arr.shape == (10, 100)
        assert voltages_arr.shape == (10, 100)
        assert spikes_arr.dtype == np.int32
        assert voltages_arr.dtype == np.int16

    def test_batch_lif_varying_unchanged(self):
        currents = np.array([120, 128, 136, 150, 160, 100, 80, 140], dtype=np.int16)
        noises = np.array([0, 1, -1, 2, -2, 0, 1, -1], dtype=np.int16)

        lif = v3.FixedPointLif()
        ref_spikes, ref_voltages = [], []
        for i_t, n_t in zip(currents, noises):
            s, v = lif.step(20, 256, int(i_t), int(n_t))
            ref_spikes.append(s)
            ref_voltages.append(v)

        spikes, voltages = v3.batch_lif_run_varying(
            leak_k=20,
            gain_k=256,
            currents=currents,
            noises=noises,
        )
        np.testing.assert_array_equal(np.asarray(spikes), np.array(ref_spikes, dtype=np.int32))
        np.testing.assert_array_equal(np.asarray(voltages), np.array(ref_voltages, dtype=np.int16))


class TestPhase11Version:
    def test_version(self):
        assert v3.__version__ == "3.6.0"
