"""Tests for batch LIF and encode operations."""

from __future__ import annotations

import numpy as np
import pytest

import sc_neurocore_engine as v3
from sc_neurocore_engine import FixedPointLif


class TestBatchLifRun:
    """batch_lif_run: constant-input batch."""

    def test_basic_shape(self):
        spikes, voltages = v3.batch_lif_run(100, leak_k=20, gain_k=256, i_t=128)
        assert spikes.shape == (100,)
        assert voltages.shape == (100,)
        assert spikes.dtype == np.int32
        assert voltages.dtype == np.int16

    def test_matches_per_step(self):
        """Batch variant must produce identical results to per-step calls."""
        n_steps = 200
        leak, gain, i_t, noise = 20, 256, 128, 0

        # Per-step
        lif = FixedPointLif()
        per_step_spikes = []
        per_step_voltages = []
        for _ in range(n_steps):
            s, v = lif.step(leak, gain, i_t, noise)
            per_step_spikes.append(s)
            per_step_voltages.append(v)

        # Batch
        batch_spikes, batch_voltages = v3.batch_lif_run(
            n_steps, leak_k=leak, gain_k=gain, i_t=i_t
        )

        np.testing.assert_array_equal(batch_spikes, per_step_spikes)
        np.testing.assert_array_equal(batch_voltages, per_step_voltages)

    def test_spiking_input(self):
        """Strong current should produce non-degenerate membrane dynamics."""
        spikes, voltages = v3.batch_lif_run(100, leak_k=20, gain_k=256, i_t=200)
        assert spikes.shape == (100,)
        assert len(np.unique(voltages)) > 1, "Membrane voltage should evolve over time"

    def test_zero_steps(self):
        spikes, voltages = v3.batch_lif_run(0, leak_k=20, gain_k=256, i_t=128)
        assert spikes.shape == (0,)
        assert voltages.shape == (0,)

    def test_custom_params(self):
        spikes, voltages = v3.batch_lif_run(
            50,
            leak_k=10,
            gain_k=512,
            i_t=100,
            data_width=16,
            fraction=8,
            v_rest=0,
            v_reset=0,
            v_threshold=256,
            refractory_period=3,
        )
        assert spikes.shape == (50,)


class TestBatchLifRunVarying:
    """batch_lif_run_varying: per-step current array."""

    def test_basic(self):
        currents = np.full(100, 128, dtype=np.int16)
        spikes, voltages = v3.batch_lif_run_varying(
            leak_k=20, gain_k=256, currents=currents
        )
        assert spikes.shape == (100,)
        assert voltages.shape == (100,)

    def test_with_noise(self):
        currents = np.full(50, 200, dtype=np.int16)
        noises = np.zeros(50, dtype=np.int16)
        spikes, voltages = v3.batch_lif_run_varying(
            leak_k=20, gain_k=256, currents=currents, noises=noises
        )
        assert spikes.shape == (50,)

    def test_matches_constant_batch(self):
        """Varying with constant array == constant batch."""
        n = 100
        currents = np.full(n, 128, dtype=np.int16)
        s1, v1 = v3.batch_lif_run(n, leak_k=20, gain_k=256, i_t=128)
        s2, v2 = v3.batch_lif_run_varying(leak_k=20, gain_k=256, currents=currents)
        np.testing.assert_array_equal(s1, s2)
        np.testing.assert_array_equal(v1, v2)


class TestBatchEncode:
    """batch_encode: Bernoulli encoding for arrays of probabilities."""

    def test_basic_shape(self):
        probs = np.array([0.3, 0.5, 0.8])
        packed = v3.batch_encode(probs, length=1024, seed=0xACE1)
        assert len(packed) == 3
        words_per = (1024 + 63) // 64
        assert all(len(row) == words_per for row in packed)

    def test_probability_accuracy(self):
        """Encoded rates should be close to input probabilities."""
        probs = np.array([0.25, 0.5, 0.75])
        packed = v3.batch_encode(probs, length=10000, seed=42)
        for i, p in enumerate(probs):
            bits_set = sum(bin(w).count("1") for w in packed[i])
            rate = bits_set / 10000
            assert abs(rate - p) < 0.05, f"prob {p}: rate {rate}"

    def test_seed_determinism(self):
        probs = np.array([0.5, 0.5])
        p1 = v3.batch_encode(probs, length=1024, seed=42)
        p2 = v3.batch_encode(probs, length=1024, seed=42)
        assert p1 == p2

    def test_empty(self):
        probs = np.array([], dtype=np.float64)
        packed = v3.batch_encode(probs, length=1024, seed=42)
        assert len(packed) == 0
