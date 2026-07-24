# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (sync_and_coincidence) from former test_spike_stats_toolkit_edge_contracts_variability.py

from __future__ import annotations

from tests.spike_stats_toolkit_edge_contracts_support import *  # noqa: F403


def test_kernel_bandwidth_zero():
    assert np.isnan(optimal_kernel_bandwidth(np.ones(5, dtype=np.int8) * 3))


def test_recovery_slope_peak_at_end():
    r = waveform_recovery_slope(np.array([0.0, 0.5, 1.0]))
    assert np.isnan(r)


def test_coincidence_with_spikes():
    # correlation.py:272 — norm > expected path
    rng = np.random.default_rng(42)
    a = rng.integers(0, 2, size=1000, dtype=np.int8)
    b = np.roll(a, 2)
    r = coincidence_index(a, b, delta_ms=5.0)
    assert np.isfinite(r)


def test_bayesian_decode_single_entry():
    # decoding.py:84 — len(classes) == 1
    r = bayesian_decode(np.array([5.0]), np.array([[5.0]]))
    assert r == 0


def test_spike_sync_with_data():
    # distance.py:160 — total_possible > 0
    ta = np.array([0.1, 0.2, 0.3, 0.5])
    tb = np.array([0.11, 0.21, 0.31, 0.51])
    r = _spike_sync(ta, tb)
    assert r > 0


def test_ssi_with_classes():
    # information.py:145 — n_s > 0 path
    counts = np.array([5, 10, 3, 8, 2])
    labels = np.array([0, 1, 0, 1, 0])
    r = stimulus_specific_information(counts, labels)
    assert np.isfinite(r)


def test_spike_sync_close_spikes():
    # distance.py:160 — total_coincidences > 0
    ta = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    tb = np.array([0.101, 0.201, 0.301, 0.401, 0.501])
    r = _spike_sync(ta, tb)
    assert r > 0


def test_ssi_mixed_classes():
    # information.py:145 — n_s > 0 AND mean_s > 0
    counts = np.array([1, 5, 2, 8, 3, 7, 4, 6])
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    r = stimulus_specific_information(counts, labels)
    assert r >= 0


def test_sttc_with_real_spikes():
    # correlation.py:123 — ta and tb both non-empty
    rng = np.random.default_rng(42)
    a = rng.integers(0, 2, size=500, dtype=np.int8)
    b = rng.integers(0, 2, size=500, dtype=np.int8)
    r = spike_time_tiling_coefficient(a, b, delta_ms=5.0)
    assert np.isfinite(r)
