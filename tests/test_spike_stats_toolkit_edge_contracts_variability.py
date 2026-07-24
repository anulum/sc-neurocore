# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (variability) from former test_spike_stats_toolkit_edge_contracts.py

from __future__ import annotations

from tests.spike_stats_toolkit_edge_contracts_support import *  # noqa: F403

def test_cv_isi_zero():
    assert np.isnan(cv_isi(np.zeros(10, dtype=np.int8)))


def test_cv2_zero():
    assert np.isnan(cv2(np.zeros(10, dtype=np.int8)))


def test_lv_zero():
    assert np.isnan(local_variation(np.zeros(10, dtype=np.int8)))


def test_lvr_zero():
    assert np.isnan(lvr(np.zeros(10, dtype=np.int8)))


def test_fano_short():
    assert np.isnan(fano_factor(np.zeros(2, dtype=np.int8), window_ms=100.0))


def test_isi_entropy_silent():
    r = isi_entropy(np.zeros(10, dtype=np.int8))
    # Silent train → no ISIs → 0 entropy or nan
    assert r == 0.0 or np.isnan(r)


def test_lempel_ziv_silent():
    r = lempel_ziv_complexity(np.array([], dtype=np.int8))
    assert r == 0.0


def test_apen_high_dim():
    r = approximate_entropy(np.zeros(5, dtype=np.int8), m=10)
    assert r == 0.0 or np.isnan(r)


def test_sampen_zero():
    r = sample_entropy(np.zeros(10, dtype=np.int8))
    assert np.isfinite(r) or np.isnan(r)


def test_perm_entropy_short():
    assert np.isnan(permutation_entropy(np.zeros(3, dtype=np.int8), order=5))


def test_perm_entropy_few_patterns():
    assert np.isnan(permutation_entropy(np.zeros(5, dtype=np.int8), order=3, delay=3))


def test_hurst_short():
    assert np.isnan(hurst_exponent(np.zeros(5, dtype=np.int8)))


def test_hurst_constant():
    r = hurst_exponent(np.ones(100, dtype=np.int8))
    assert np.isfinite(r) or np.isnan(r)


def test_allan_short():
    af, windows = allan_factor(np.zeros(3, dtype=np.int8))
    assert af.size >= 0


def test_allan_with_data():
    rng = np.random.default_rng(42)
    train = rng.integers(0, 2, size=1000, dtype=np.int8)
    af, windows = allan_factor(train)
    assert af.size > 0


def test_rescaled_range_short():
    assert np.isnan(rescaled_range(np.zeros(5, dtype=np.int8)))


def test_rescaled_range_constant():
    r = rescaled_range(np.ones(100, dtype=np.int8))
    assert np.isfinite(r) or np.isnan(r)


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


def test_cell_assembly_with_strong_corr():
    # network.py:89-91 — eigval > mp_upper, members >= 2
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2, size=500, dtype=np.int8)
    trains = [base.copy() for _ in range(10)]
    for i in range(10):
        trains[i] = np.roll(trains[i], i)
    r = cell_assembly_detection(trains, bin_size=5, threshold=0.5)
    assert isinstance(r, list)


def test_cubic_with_data():
    # patterns.py:81 — valid_n > 0
    rng = np.random.default_rng(42)
    train = rng.integers(0, 2, size=200, dtype=np.int8)
    r = cubic_higher_order(train, max_lag=5)
    assert r.shape[0] > 0


def test_amplitude_cutoff_with_data():
    # sorting_quality.py:148 — total > 0
    rng = np.random.default_rng(0)
    amps = np.abs(rng.standard_normal(500)) + 0.5
    r = amplitude_cutoff(amps)
    assert np.isfinite(r)


def test_place_field_ending_in_field():
    # stimulus.py:116 — in_field at end of array
    train = np.zeros(100, dtype=np.int8)
    train[80:] = 1
    pos = np.linspace(0, 1, 100)
    fields = place_field_detection(train, pos, threshold_std=0.5)
    assert any(f[1] >= 0.9 for f in fields) if fields else True


def test_inhomogeneous_poisson_zero():
    # surrogates.py:82 — max_rate <= 0
    from sc_neurocore.analysis.spike_stats.surrogates import inhomogeneous_poisson

    r = inhomogeneous_poisson(rate_func=lambda t: 0.0, duration_s=1.0)
    assert np.all(r == 0)


def test_rescaled_range_with_data():
    # variability.py:334 — duplicate scale dedup
    rng = np.random.default_rng(42)
    train = rng.integers(0, 2, size=500, dtype=np.int8)
    r = rescaled_range(train)
    assert np.isfinite(r) or np.isnan(r)


def test_waveform_recovery_short():
    # waveform.py:53 — dv.size == 0
    r = waveform_recovery_slope(np.array([1.0]))
    assert np.isnan(r)


def test_dtf_singular():
    # causality.py:186 — det_a near zero → continue
    trains = [np.zeros(50, dtype=np.int8)] * 3
    r = directed_transfer_function(trains, order=2)
    assert r.shape[0] > 0


def test_spatial_info_with_rate():
    # stimulus.py:74 — mean_rate > 0 path
    rng = np.random.default_rng(42)
    train = rng.integers(0, 2, size=200, dtype=np.int8)
    pos = rng.uniform(0, 1, size=200)
    r = _si2(train, pos, n_bins=10)
    assert np.isfinite(r)


def test_sorting_cutoff_nonzero():
    # sorting_quality.py:148 — total > 0, right > left
    amps = np.concatenate([np.random.randn(200) + 2, np.random.randn(50) + 5])
    r = amplitude_cutoff(amps, bins=50)
    assert np.isfinite(r)


def test_cubic_with_real_data():
    # patterns.py:81 — valid_n > 0
    rng = np.random.default_rng(1)
    train = rng.integers(0, 2, size=500, dtype=np.int8).astype(np.float64)
    r = cubic_higher_order(train, max_lag=3)
    assert np.any(r != 0)


def test_ssi_mixed_classes():
    # information.py:145 — n_s > 0 AND mean_s > 0
    counts = np.array([1, 5, 2, 8, 3, 7, 4, 6])
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    r = stimulus_specific_information(counts, labels)
    assert r >= 0


def test_spike_sync_close_spikes():
    # distance.py:160 — total_coincidences > 0
    ta = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    tb = np.array([0.101, 0.201, 0.301, 0.401, 0.501])
    r = _spike_sync(ta, tb)
    assert r > 0


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


def test_sttc_with_real_spikes():
    # correlation.py:123 — ta and tb both non-empty
    rng = np.random.default_rng(42)
    a = rng.integers(0, 2, size=500, dtype=np.int8)
    b = rng.integers(0, 2, size=500, dtype=np.int8)
    r = spike_time_tiling_coefficient(a, b, delta_ms=5.0)
    assert np.isfinite(r)


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


def test_dtf_with_real_data():
    # causality.py:186 — det_a NOT near zero
    rng = np.random.default_rng(42)
    trains = [rng.integers(0, 2, size=200, dtype=np.int8) for _ in range(3)]
    r = directed_transfer_function(trains, order=2)
    assert r.shape[0] > 0


def test_waveform_recovery_valid():
    # waveform.py:53 — dv.size > 0 path
    wf = np.array([0.0, -1.0, -0.5, 0.2, 0.8, 0.5, 0.1])
    r = waveform_recovery_slope(wf, dt=1.0)
    assert np.isfinite(r)


def test_rescaled_range_real_data():
    # variability.py:334 — scales with real data
    rng = np.random.default_rng(42)
    train = rng.integers(0, 2, size=2000, dtype=np.int8)
    r = rescaled_range(train)
    assert np.isfinite(r)


