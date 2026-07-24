# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (distance_spatial_decode) from former test_spike_stats_toolkit_edge_contracts_variability.py

from __future__ import annotations

from tests.spike_stats_toolkit_edge_contracts_support import *  # noqa: F403


def test_isi_distance_zero_timestep_collapses_intervals():
    # distance.py:97 — a zero timestep maps every spike to t=0, so both ISI
    # sequences are all-zero and the matched ratio is exactly 0.0.
    train_a = np.array([1, 0, 1, 0, 1], dtype=np.int8)
    train_b = np.array([1, 1, 0, 1, 0], dtype=np.int8)
    r = isi_distance(train_a, train_b, dt=0.0)
    assert r == 0.0


def test_spike_directionality_one_sided_neighbours():
    # patterns.py:47 — both trains are non-empty but every reference spike sees
    # partner spikes on only one side, so no lead is ever scored (total == 0).
    r = spike_directionality(np.array([0.5]), np.array([0.6, 0.7]))
    assert r == 0.0


def test_cubic_higher_order_lag_exceeds_signal():
    # patterns.py:82 — lags beyond the signal length leave valid_n <= 0, so the
    # corresponding cumulant entries are skipped and stay zero.
    r = cubic_higher_order(np.array([0, 1, 0], dtype=np.int8), max_lag=20)
    assert r.shape == (20, 20)
    assert r[10, 10] == 0.0


def test_spatial_information_zero_timestep_no_occupancy():
    # stimulus.py:73 — a zero timestep yields zero occupancy everywhere, so the
    # information measure is undefined and returns 0.0.
    train = np.ones(12, dtype=np.int8)
    positions = np.linspace(0.0, 1.0, 12)
    assert spatial_information(train, positions, dt=0.0) == 0.0


def test_spatial_information_silent_train_zero_mean_rate():
    # stimulus.py:80 — a silent train has zero mean firing rate, so the
    # bits-per-spike normalisation is undefined and returns 0.0.
    train = np.zeros(20, dtype=np.int8)
    positions = np.linspace(0.0, 1.0, 20)
    assert spatial_information(train, positions) == 0.0


def test_lda_decode_valid():
    # decoding.py:84 — len(classes) >= 2
    from sc_neurocore.analysis.spike_stats.decoding import linear_discriminant_decode

    train_data = np.array([[1, 2], [2, 3], [5, 6], [6, 7]], dtype=float)
    labels = np.array([0, 0, 1, 1])
    r = linear_discriminant_decode(train_data, labels, np.array([3.0, 4.0]))
    assert r in (0, 1)


def test_lda_decode_single_class():
    # decoding.py:89 — len(classes) == 1 → return the sole class without a solve
    from sc_neurocore.analysis.spike_stats.decoding import linear_discriminant_decode

    train_data = np.array([[1.0, 2.0], [2.0, 3.0]])
    labels = np.array([7, 7])
    assert linear_discriminant_decode(train_data, labels, np.array([1.0, 1.0])) == 7


def test_lda_decode_empty_labels():
    # decoding.py:89 — len(classes) == 0 → 0
    from sc_neurocore.analysis.spike_stats.decoding import linear_discriminant_decode

    empty = np.empty((0, 2))
    r = linear_discriminant_decode(empty, np.array([]), np.array([1.0, 1.0]))
    assert r == 0


def test_population_vector_decode_no_bins():
    # decoding.py:34 — min_len // window == 0 → empty result
    from sc_neurocore.analysis.spike_stats.decoding import population_vector_decode

    short = [np.ones(10)]
    r = population_vector_decode(short, np.array([0.0]), window=50)
    assert r.size == 0
