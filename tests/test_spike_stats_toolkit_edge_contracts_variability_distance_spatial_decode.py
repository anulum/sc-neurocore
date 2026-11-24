# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (distance_spatial_decode) from former test_spike_stats_toolkit_edge_contracts_variability.py

"""Edge contracts of the ISI distance, spike directionality, the cubic statistic, spatial information and the LDA and population-vector decoders."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.distance import (
    isi_distance,
)
from sc_neurocore.analysis.spike_stats.patterns import (
    cubic_higher_order,
    spike_directionality,
)
from sc_neurocore.analysis.spike_stats.stimulus import (
    spatial_information,
)


def test_isi_distance_zero_timestep_collapses_intervals() -> None:
    """With a zero time step the ISI distance collapses to zero."""
    # distance.py:97 — a zero timestep maps every spike to t=0, so both ISI
    # sequences are all-zero and the matched ratio is exactly 0.0.
    train_a = np.array([1, 0, 1, 0, 1], dtype=np.int8)
    train_b = np.array([1, 1, 0, 1, 0], dtype=np.int8)
    r = isi_distance(train_a, train_b, dt=0.0)
    assert r == 0.0


def test_spike_directionality_one_sided_neighbours() -> None:
    """Spike directionality with neighbours on one side only is zero."""
    # patterns.py:47 — both trains are non-empty but every reference spike sees
    # partner spikes on only one side, so no lead is ever scored (total == 0).
    r = spike_directionality(np.array([0.5]), np.array([0.6, 0.7]))
    assert r == 0.0


def test_cubic_higher_order_lag_exceeds_signal() -> None:
    """A maximum lag beyond the signal gives a 20x20 statistic that is zero at the centre."""
    # patterns.py:82 — lags beyond the signal length leave valid_n <= 0, so the
    # corresponding cumulant entries are skipped and stay zero.
    r = cubic_higher_order(np.array([0, 1, 0], dtype=np.int8), max_lag=20)
    assert r.shape == (20, 20)
    assert r[10, 10] == 0.0


def test_spatial_information_zero_timestep_no_occupancy() -> None:
    """With a zero time step spatial information has no occupancy and is zero."""
    # stimulus.py:73 — a zero timestep yields zero occupancy everywhere, so the
    # information measure is undefined and returns 0.0.
    train = np.ones(12, dtype=np.int8)
    positions = np.linspace(0.0, 1.0, 12)
    assert spatial_information(train, positions, dt=0.0) == 0.0


def test_spatial_information_silent_train_zero_mean_rate() -> None:
    """Spatial information of a silent train is zero."""
    # stimulus.py:80 — a silent train has zero mean firing rate, so the
    # bits-per-spike normalisation is undefined and returns 0.0.
    train = np.zeros(20, dtype=np.int8)
    positions = np.linspace(0.0, 1.0, 20)
    assert spatial_information(train, positions) == 0.0


def test_lda_decode_valid() -> None:
    """LDA decoding of a point between two classes returns one of them."""
    # decoding.py:84 — len(classes) >= 2
    from sc_neurocore.analysis.spike_stats.decoding import linear_discriminant_decode

    train_data = np.array([[1, 2], [2, 3], [5, 6], [6, 7]], dtype=float)
    labels = np.array([0, 0, 1, 1])
    r = linear_discriminant_decode(train_data, labels, np.array([3.0, 4.0]))
    assert r in (0, 1)


def test_lda_decode_single_class() -> None:
    """LDA decoding with a single training class returns that class."""
    # decoding.py:89 — len(classes) == 1 → return the sole class without a solve
    from sc_neurocore.analysis.spike_stats.decoding import linear_discriminant_decode

    train_data = np.array([[1.0, 2.0], [2.0, 3.0]])
    labels = np.array([7, 7])
    assert linear_discriminant_decode(train_data, labels, np.array([1.0, 1.0])) == 7


def test_lda_decode_empty_labels() -> None:
    """LDA decoding without training data returns 0."""
    # decoding.py:89 — len(classes) == 0 → 0
    from sc_neurocore.analysis.spike_stats.decoding import linear_discriminant_decode

    empty = np.empty((0, 2))
    r = linear_discriminant_decode(empty, np.array([]), np.array([1.0, 1.0]))
    assert r == 0


def test_population_vector_decode_no_bins() -> None:
    """Population-vector decoding of trains shorter than the window is empty."""
    # decoding.py:34 — min_len // window == 0 → empty result
    from sc_neurocore.analysis.spike_stats.decoding import population_vector_decode

    short = [np.ones(10)]
    r = population_vector_decode(short, np.array([0.0]), window=50)
    assert r.size == 0
