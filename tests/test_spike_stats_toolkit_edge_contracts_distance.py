# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (distance) from former test_spike_stats_toolkit_edge_contracts.py

from __future__ import annotations

from tests.spike_stats_toolkit_edge_contracts_support import *  # noqa: F403

def test_vp_empty_b():
    r = victor_purpura_distance(np.array([0.1, 0.2]), np.array([]))
    assert r == 2.0


def test_vp_empty_a():
    r = victor_purpura_distance(np.array([]), np.array([0.1, 0.2]))
    assert r == 2.0


def test_isi_dist_silent():
    r = isi_distance(np.zeros(10, dtype=np.int8), np.zeros(10, dtype=np.int8))
    assert r == 0.0 or np.isnan(r)


def test_spike_sync_empty():
    assert spike_sync(np.array([]), np.array([0.1, 0.2])) == 0.0


def test_hunter_milton_empty():
    assert hunter_milton_similarity(np.array([]), np.array([0.1])) == 0.0


def test_gvp_empty():
    r = generalized_victor_purpura(np.array([]), np.array([0.1, 0.2]))
    assert r >= 0


def test_gvp_empty_b():
    r = generalized_victor_purpura(np.array([0.1, 0.2]), np.array([]))
    assert r >= 0


def test_schreiber_silent():
    r = schreiber_similarity(np.zeros(100, dtype=np.int8), np.zeros(100, dtype=np.int8))
    assert r == 0.0


def test_isi_profile_short():
    r = isi_profile(np.array([1, 0], dtype=np.int8), np.array([0, 1], dtype=np.int8), n_bins=100)
    assert r.shape[0] > 0


def test_distance_python_fallback_algorithms(monkeypatch):
    monkeypatch.setattr(distance_module, "_HAS_RUST", False)
    monkeypatch.setattr(distance_module, "_ssc", None)

    train_a = np.array([1, 0, 1, 0, 0, 1], dtype=np.float64)
    train_b = np.array([1, 0, 0, 1, 0, 1], dtype=np.float64)
    assert van_rossum_distance(train_a, train_a, tau_ms=5.0) == 0.0
    assert van_rossum_distance(train_a, train_b, tau_ms=5.0) > 0.0
    assert np.isnan(van_rossum_distance(train_a, train_b, tau_ms=0.0))

    assert victor_purpura_distance(np.array([]), np.array([0.2]), cost_per_s=100.0) == 1.0
    assert np.isclose(
        victor_purpura_distance(np.array([0.1]), np.array([0.102]), cost_per_s=100.0),
        0.2,
    )
    assert victor_purpura_distance(np.array([0.1, 0.3]), np.array([0.1, 0.32]), 10.0) < 1.0

    dense_a = np.array([1, 0, 1, 0, 0, 1, 0, 0], dtype=np.int8)
    dense_b = np.array([1, 0, 0, 1, 0, 0, 1, 0], dtype=np.int8)
    isi_result = isi_distance(dense_a, dense_b, dt=0.001)
    assert np.isfinite(isi_result)
    assert isi_result >= 0.0

    outside = np.array([-0.1, 1.1])
    assert spike_distance(outside, outside, 0.0, 1.0) == 0.0
    assert spike_distance(np.array([0.2]), outside, 0.0, 1.0) == 1.0
    assert spike_distance(np.array([0.2, 0.6]), np.array([0.2, 0.6]), 0.0, 1.0) == 0.0

    times = np.array([0.1, 0.3, 0.8])
    assert _local_isi(np.array([0.5]), 0) == 1.0
    assert np.isclose(_local_isi(times, 0), 0.2)
    assert np.isclose(_local_isi(times, 1), 0.2)
    assert np.isclose(_local_isi(times, 2), 0.5)

    assert spike_sync(np.array([]), np.array([0.1, 0.3])) == 0.0
    assert spike_sync(np.array([0.1, 0.3]), np.array([0.1, 0.3])) == 1.0
    shifted_sync = spike_sync(np.array([0.1, 0.3]), np.array([0.2, 0.4]))
    assert 0.0 <= shifted_sync < 1.0

    sync_profile = spike_sync_profile(
        np.array([0.1, 0.7]), np.array([0.1, 0.8]), n_bins=2, t_start=0.0, t_end=1.0
    )
    spike_dist_profile = spike_profile(
        np.array([0.1, 0.7]), np.array([0.2, 0.8]), n_bins=2, t_start=0.0, t_end=1.0
    )
    assert sync_profile.shape == (2,)
    assert spike_dist_profile.shape == (2,)
    assert np.all(np.isfinite(sync_profile))
    assert np.all(np.isfinite(spike_dist_profile))

    adaptive_zero = adaptive_spike_distance(np.array([0.2, 0.6]), np.array([0.2, 0.7]), cost=0.0)
    adaptive_one = adaptive_spike_distance(np.array([0.2, 0.6]), np.array([0.2, 0.7]), cost=1.0)
    assert adaptive_zero == spike_distance(np.array([0.2, 0.6]), np.array([0.2, 0.7]))
    assert 0.0 <= adaptive_one <= 1.0

    assert hunter_milton_similarity(np.array([0.1, 0.2]), np.array([0.101, 0.5])) == 0.5
    assert hunter_milton_similarity(np.array([0.1]), np.array([])) == 0.0
    correlated = schreiber_similarity(dense_a, dense_b, sigma_ms=1.0)
    assert -1.0 <= correlated <= 1.0
    assert correlated != 0.0
    assert earth_movers_distance(np.array([]), np.array([0.5]), n_bins=10) > 0.0
    assert earth_movers_distance(np.array([0.1, 0.2]), np.array([0.7, 0.8]), n_bins=10) > 0.0
    matrix = multi_neuron_victor_purpura(
        [np.array([0.1, 0.2]), np.array([0.1, 0.25]), np.array([])],
        cost_per_s=10.0,
    )
    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)

    gvp_default = generalized_victor_purpura(np.array([0.1, 0.2]), np.array([0.1, 0.22]))
    gvp_custom = generalized_victor_purpura(
        np.array([0.1, 0.2]),
        np.array([0.1, 0.22]),
        cost_func=lambda delta: 5.0 * abs(delta),
    )
    assert 0.0 <= gvp_custom < gvp_default

    trains = [np.array([0.1, 0.2]), np.array([0.1, 0.25]), np.array([0.7])]
    for metric in ["spike_distance", "spike_sync", "victor_purpura", "unknown"]:
        distances = spike_distance_matrix(trains, metric=metric, t_start=0.0, t_end=1.0)
        assert distances.shape == (3, 3)
        assert np.allclose(distances, distances.T)
        assert np.allclose(np.diag(distances), 0.0)


def test_distance_rust_acceleration_delegation(monkeypatch):
    class RustCore:
        def py_van_rossum_distance(self, a, b, dt, tau_ms):
            assert a.flags.c_contiguous
            assert b.flags.c_contiguous
            return 1.25

        def py_spike_distance(self, a, b, t_start, t_end):
            assert a.flags.c_contiguous
            assert b.flags.c_contiguous
            return 0.75

        def py_hunter_milton(self, a, b, dt_max):
            assert a.flags.c_contiguous
            assert b.flags.c_contiguous
            return 0.5

        def py_multi_neuron_vp(self, arrs, cost_per_s):
            assert all(arr.flags.c_contiguous for arr in arrs)
            return [0.0, 2.0, 2.0, 0.0]

    monkeypatch.setattr(distance_module, "_HAS_RUST", True)
    monkeypatch.setattr(distance_module, "_ssc", RustCore())

    assert van_rossum_distance(np.array([1, 0]), np.array([0, 1])) == 1.25
    assert spike_distance(np.array([0.1]), np.array([0.2])) == 0.75
    assert hunter_milton_similarity(np.array([0.1]), np.array([0.1])) == 0.5
    matrix = multi_neuron_victor_purpura([np.array([0.1]), np.array([0.2])])
    assert np.array_equal(matrix, np.array([[0.0, 2.0], [2.0, 0.0]]))


def test_gpfa_transform_empty():
    params = {
        "C": np.zeros((0, 2)),
        "d": np.array([]),
        "R": np.array([]),
        "tau": np.array([10.0, 10.0]),
    }
    r = gpfa_transform([], params)
    assert r.size == 0


