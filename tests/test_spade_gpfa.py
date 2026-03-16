# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for SPADE and GPFA modules."""

from __future__ import annotations

import numpy as np

from sc_neurocore.analysis.spike_stats import spade_detect, gpfa, gpfa_transform
from sc_neurocore.analysis.spike_stats.spade import (
    _find_frequent_itemsets,
    _extend_to_spatiotemporal,
)
from sc_neurocore.analysis.spike_stats.gpfa import _gp_kernel


def _sync_trains(n_neurons=5, n_steps=2000, sync_every=100, seed=42):
    """Generate spike trains with a planted synchronous pattern."""
    rng = np.random.default_rng(seed)
    trains = []
    for i in range(n_neurons):
        t = np.zeros(n_steps, dtype=np.uint8)
        spikes = rng.choice(n_steps, size=20, replace=False)
        t[spikes] = 1
        trains.append(t)
    # Plant synchronous events for neurons 0,1,2 at regular intervals
    for offset in range(0, n_steps, sync_every):
        if offset < n_steps:
            for nid in [0, 1, 2]:
                trains[nid][offset] = 1
    return trains


def _poisson_trains(n_neurons=6, rate_hz=30.0, duration_s=1.0, dt=0.001, seed=42):
    rng = np.random.default_rng(seed)
    n_steps = int(duration_s / dt)
    trains = []
    for _ in range(n_neurons):
        t = (rng.random(n_steps) < rate_hz * dt).astype(np.uint8)
        trains.append(t)
    return trains


# ── SPADE tests ──


class TestFindFrequentItemsets:
    def test_single_coactive_pair(self):
        mat = np.array(
            [
                [1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1],
                [0, 1, 0, 1, 0],
            ],
            dtype=np.int8,
        )
        result = _find_frequent_itemsets(mat, min_support=3, max_size=3)
        pair_sets = [s for s, c in result if len(s) == 2]
        assert frozenset([0, 1]) in pair_sets

    def test_min_support_filters(self):
        mat = np.array(
            [
                [1, 0, 1, 0, 0],
                [1, 0, 0, 0, 0],
            ],
            dtype=np.int8,
        )
        result = _find_frequent_itemsets(mat, min_support=3, max_size=2)
        pair_sets = [s for s, c in result if len(s) == 2]
        assert len(pair_sets) == 0


class TestSpadeDetect:
    def test_planted_pattern_detected(self):
        trains = _sync_trains(n_neurons=5, n_steps=2000, sync_every=100)
        results = spade_detect(
            trains,
            bin_ms=5.0,
            dt=0.001,
            min_support=3,
            n_surrogates=50,
            alpha=0.05,
            seed=42,
        )
        detected_sets = [frozenset(r["neurons"]) for r in results]
        assert any(frozenset([0, 1, 2]).issubset(s) for s in detected_sets)

    def test_empty_trains(self):
        assert spade_detect([], bin_ms=5.0) == []

    def test_single_train(self):
        trains = [np.zeros(100, dtype=np.uint8)]
        assert spade_detect(trains) == []

    def test_results_have_required_keys(self):
        trains = _sync_trains()
        results = spade_detect(trains, n_surrogates=20, alpha=0.5)
        for r in results:
            assert "neurons" in r
            assert "lags" in r
            assert "count" in r
            assert "p_value" in r
            assert 0.0 <= r["p_value"] <= 1.0

    def test_no_false_positives_on_independent(self):
        rng = np.random.default_rng(99)
        trains = []
        for i in range(4):
            t = np.zeros(500, dtype=np.uint8)
            t[rng.choice(500, size=5, replace=False)] = 1
            trains.append(t)
        results = spade_detect(trains, min_support=3, n_surrogates=50, alpha=0.01)
        assert len(results) == 0


# ── GPFA tests ──


class TestGPKernel:
    def test_shape_and_symmetry(self):
        K = _gp_kernel(50, tau=10.0)
        assert K.shape == (50, 50)
        np.testing.assert_allclose(K, K.T)

    def test_diagonal_equals_sigma_squared(self):
        K = _gp_kernel(30, tau=5.0, sigma=2.0)
        np.testing.assert_allclose(np.diag(K), 4.0)

    def test_positive_definite(self):
        K = _gp_kernel(20, tau=8.0)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-10)


class TestGPFA:
    def test_basic_output_shape(self):
        trains = _poisson_trains(n_neurons=6, duration_s=0.5)
        result = gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=5)
        assert result["trajectories"].shape[0] == 2
        assert result["C"].shape == (6, 2)
        assert result["d"].shape == (6,)
        assert len(result["log_likelihoods"]) > 0

    def test_log_likelihood_increases(self):
        trains = _poisson_trains(n_neurons=8, duration_s=0.5, seed=7)
        result = gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=15)
        lls = result["log_likelihoods"]
        if len(lls) >= 3:
            assert lls[-1] >= lls[0] - 1.0  # allow small numerical wobble

    def test_empty_trains(self):
        result = gpfa([], n_latents=2)
        assert result["trajectories"].size == 0

    def test_transform_matches_shape(self):
        trains = _poisson_trains(n_neurons=4, duration_s=0.5)
        result = gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=5)
        new_trains = _poisson_trains(n_neurons=4, duration_s=0.5, seed=99)
        proj = gpfa_transform(new_trains, result, bin_ms=20.0)
        assert proj.shape[0] == 2

    def test_single_latent(self):
        trains = _poisson_trains(n_neurons=3, duration_s=0.3)
        result = gpfa(trains, n_latents=1, bin_ms=20.0, max_iter=5)
        assert result["trajectories"].shape[0] == 1
        assert result["C"].shape[1] == 1
