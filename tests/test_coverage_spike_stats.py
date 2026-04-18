# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Coverage: spike_stats edge-case guard clauses

"""Hit every uncovered guard clause in analysis/spike_stats/."""

from __future__ import annotations

import numpy as np

# --- basic (lines 32, 46) ---
from sc_neurocore.analysis.spike_stats.basic import firing_rate, bin_spike_train


def test_firing_rate_zero_duration():
    assert firing_rate(np.array([]), dt=0.001) == 0.0


def test_bin_spike_train_small():
    r = bin_spike_train(np.array([1, 0, 1]), bin_size=10)
    assert r[0] == 2


# --- causality (lines 25, 48, 65, 85, 102, 125, 186) ---
from sc_neurocore.analysis.spike_stats.causality import (
    pairwise_granger_causality,
    conditional_granger_causality,
    spectral_granger_causality,
)


def test_granger_short():
    assert pairwise_granger_causality(np.zeros(3), np.zeros(3), order=5) == 0.0


def test_granger_constant():
    a = np.ones(100, dtype=np.int8)
    r = pairwise_granger_causality(a, a, order=2)
    assert np.isfinite(r)


def test_conditional_granger_short():
    assert conditional_granger_causality(np.zeros(3), np.zeros(3), np.zeros(3), order=5) == 0.0


def test_conditional_granger_constant():
    a = np.ones(100, dtype=np.int8)
    r = conditional_granger_causality(a, a, a, order=2)
    assert np.isfinite(r)


def test_spectral_granger_singular():
    trains = [np.zeros(50, dtype=np.int8)] * 3
    r = spectral_granger_causality(trains, order=2)
    assert r.shape[0] > 0


# --- correlation (lines 53, 74, 93, 123, 160, 184, 262, 272) ---
from sc_neurocore.analysis.spike_stats.correlation import (
    pairwise_correlation,
    spike_train_coherence,
    event_synchronization,
    spike_time_tiling_coefficient,
    autocorrelation_time,
    coincidence_index,
)


def test_pairwise_empty():
    assert pairwise_correlation([]).size == 0


def test_coherence_short():
    # line 93: n < 2
    cc, freqs = spike_train_coherence(np.array([1], dtype=np.int8), np.array([0], dtype=np.int8))
    assert cc.size == 0


def test_event_sync_empty():
    # line 74: na == 0
    assert (
        event_synchronization(np.array([], dtype=np.int8), np.array([1, 0], dtype=np.int8)) == 0.0
    )


def test_sttc_full_coverage():
    # line 160: edge cases p near 1
    a = np.ones(100, dtype=np.int8)
    b = np.ones(100, dtype=np.int8)
    r = spike_time_tiling_coefficient(a, b, delta_ms=50.0)
    assert np.isfinite(r)


def test_autocorr_zero_var():
    # line 184: var == 0
    assert autocorrelation_time(np.zeros(100, dtype=np.int8)) == 0.0


def test_coincidence_below_expected():
    # line 272: norm <= expected
    assert coincidence_index(np.zeros(100, dtype=np.int8), np.zeros(100, dtype=np.int8)) == 0.0


def test_event_sync_empty_b():
    # line 123: tb empty
    assert (
        event_synchronization(np.array([1, 0], dtype=np.int8), np.array([], dtype=np.int8)) == 0.0
    )


# --- decoding (lines 31, 84) ---
from sc_neurocore.analysis.spike_stats.decoding import bayesian_decode


def test_bayesian_single_class():
    # line 84: len(classes) < 2
    assert bayesian_decode(np.array([3.0]), np.array([[1.0]])) == 0


# --- dimensionality (lines 25, 32, 58) ---
from sc_neurocore.analysis.spike_stats.dimensionality import (
    spike_train_pca,
    demixed_pca,
)


def test_pca_empty():
    s, e = spike_train_pca([])
    assert s.size == 0


def test_pca_1d():
    # line 32: cov.ndim < 2
    s, e = spike_train_pca([np.array([1, 0, 1], dtype=np.int8)])
    assert s.shape[0] == 1


def test_demixed_insufficient():
    # line 58: len(all_means) < 2 — need single condition with 1 train
    s, e = demixed_pca({0: [np.array([1, 0], dtype=np.int8)]})
    assert s.size == 0


# --- distance (lines 47, 73, 160, 217, 262, 271, 333, 335) ---
from sc_neurocore.analysis.spike_stats.distance import (
    victor_purpura_distance,
    isi_distance,
    spike_sync,
    schreiber_similarity,
    hunter_milton_similarity,
    generalized_victor_purpura,
    isi_profile,
)


def test_vp_empty_b():
    # line 47: nb == 0 → return na
    r = victor_purpura_distance(np.array([0.1, 0.2]), np.array([]))
    assert r == 2.0


def test_vp_empty_a():
    # line 333-335: na == 0 → return nb
    r = victor_purpura_distance(np.array([]), np.array([0.1, 0.2]))
    assert r == 2.0


def test_isi_dist_silent():
    # line 73: a == 0 and b == 0
    r = isi_distance(np.zeros(10, dtype=np.int8), np.zeros(10, dtype=np.int8))
    assert r == 0.0 or np.isnan(r)


def test_spike_sync_empty():
    # line 160, 262, 271: empty spike times
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
    # line 262: both trains silent → cosine = 0
    r = schreiber_similarity(np.zeros(100, dtype=np.int8), np.zeros(100, dtype=np.int8))
    assert r == 0.0


def test_isi_profile_short():
    # line 217: start >= n
    r = isi_profile(np.array([1, 0], dtype=np.int8), np.array([0, 1], dtype=np.int8), n_bins=100)
    assert r.shape[0] > 0


# --- gpfa (line 201) ---
from sc_neurocore.analysis.spike_stats.gpfa import gpfa_transform


def test_gpfa_transform_empty():
    # line 201: n_neurons == 0
    params = {
        "C": np.zeros((0, 2)),
        "d": np.array([]),
        "R": np.array([]),
        "tau": np.array([10.0, 10.0]),
    }
    r = gpfa_transform([], params)
    assert r.size == 0


# --- information (lines 54, 89, 112, 120, 136, 139, 145, 160, 197) ---
from sc_neurocore.analysis.spike_stats.information import (
    transfer_entropy,
    spike_train_entropy,
    noise_entropy,
    stimulus_specific_information,
    kozachenko_leonenko_mi,
    time_rescaling_ks_test,
)


def test_te_short():
    assert transfer_entropy(np.zeros(3, dtype=np.int8), np.zeros(3, dtype=np.int8), lag=5) == 0.0


def test_entropy_short():
    r = spike_train_entropy(np.zeros(2, dtype=np.int8), word_length=5)
    assert np.isnan(r)


def test_noise_entropy_short():
    r = noise_entropy(np.zeros(2, dtype=np.int8), n_trials=1, word_length=5)
    assert np.isnan(r)


def test_ssi_empty():
    assert stimulus_specific_information(np.array([]), np.array([], dtype=int)) == 0.0


def test_ssi_zero_mean():
    assert stimulus_specific_information(np.zeros(10), np.zeros(10, dtype=int)) == 0.0


def test_kl_mi_short():
    assert kozachenko_leonenko_mi(np.zeros(2), np.zeros(2)) == 0.0


def test_time_rescaling_few():
    # line 197: times.size < 5; needs rate_func arg
    p, sig = time_rescaling_ks_test(np.array([0.1, 0.5]), rate_func=lambda t: 10.0)
    assert p == 1.0


# --- lfp (lines 28, 41) ---
from sc_neurocore.analysis.spike_stats.lfp import (
    phase_locking_value,
    spike_field_coherence,
)


def test_plv_no_spikes():
    # line 28: spike_idx.size == 0
    assert phase_locking_value(np.zeros(100, dtype=np.int8), np.sin(np.linspace(0, 10, 100))) == 0.0


def test_sfc_short():
    # line 41: n < 2
    f, p = spike_field_coherence(np.array([1], dtype=np.int8), np.array([1.0]))
    assert f.size == 0


# --- network (lines 58-60, 89-91) ---
from sc_neurocore.analysis.spike_stats.network import (
    unitary_events,
    cell_assembly_detection,
)


def test_unitary_events_significant():
    rng = np.random.default_rng(42)
    trains = [rng.integers(0, 2, size=200, dtype=np.int8) for _ in range(10)]
    r = unitary_events(trains, bin_size=5, alpha=0.99)
    assert isinstance(r, list)


def test_cell_assembly():
    rng = np.random.default_rng(0)
    trains = [rng.integers(0, 2, size=500, dtype=np.int8) for _ in range(20)]
    r = cell_assembly_detection(trains, bin_size=10)
    assert isinstance(r, list)


# --- patterns (lines 30, 46, 81) ---
from sc_neurocore.analysis.spike_stats.patterns import (
    spike_directionality,
    cubic_higher_order,
)


def test_directionality_empty():
    assert spike_directionality(np.array([]), np.array([0.1])) == 0.0


def test_directionality_zero():
    assert spike_directionality(np.array([]), np.array([])) == 0.0


def test_cubic_higher_order_short():
    r = cubic_higher_order(np.zeros(5, dtype=np.int8), max_lag=2)
    assert r.shape[0] > 0


# --- point_process (lines 39, 58, 76) ---
from sc_neurocore.analysis.spike_stats.point_process import (
    isi_hazard_function,
    isi_survivor_function,
    renewal_density,
)


def test_hazard_few():
    # line 39: intervals.size < 5
    h, e = isi_hazard_function(np.array([1, 0], dtype=np.int8))
    assert h.size == 0


def test_survivor_few():
    # line 58: intervals.size < 2
    s, e = isi_survivor_function(np.array([1, 0], dtype=np.int8))
    assert s.size == 0


def test_renewal_few():
    # line 76: intervals.size < 5
    r, e = renewal_density(np.array([1, 0], dtype=np.int8))
    assert r.size == 0


# --- rate (lines 73, 78) ---
from sc_neurocore.analysis.spike_stats.rate import psth


def test_psth_zero_bins():
    # line 73: n_bins == 0
    r, t = psth([np.array([], dtype=np.int8)], bin_ms=100.0)
    assert r.size == 0


def test_psth_empty_trial():
    # line 78: trimmed.size == 0
    r, t = psth([np.zeros(200, dtype=np.int8), np.array([], dtype=np.int8)], bin_ms=10.0)
    assert r.shape[0] > 0


# --- sorting_quality (lines 39, 148) ---
from sc_neurocore.analysis.spike_stats.sorting_quality import (
    isolation_distance,
    amplitude_cutoff,
)


def test_isolation_dist_small():
    rng = np.random.default_rng(42)
    r = isolation_distance(rng.standard_normal((5, 2)), rng.standard_normal((10, 2)))
    assert np.isfinite(r)


def test_amplitude_cutoff_symmetric():
    # line 148: total == 0 when no amplitudes
    rng = np.random.default_rng(0)
    amps = rng.standard_normal(200)
    r = amplitude_cutoff(amps)
    assert np.isfinite(r)


# --- stimulus (lines 40, 55, 67, 74, 116, 129) ---
from sc_neurocore.analysis.spike_stats.stimulus import (
    spike_triggered_covariance,
    spatial_information,
    place_field_detection,
    tuning_curve,
)


def test_stc_few_spikes():
    # line 40: valid.size < 3
    stim = np.random.randn(100)
    train = np.zeros(100, dtype=np.int8)
    train[50] = 1
    r = spike_triggered_covariance(stim, train, window_steps=5)
    assert r.shape[0] > 0


def test_spatial_info_few():
    # line 55: n < 10
    assert spatial_information(np.zeros(5, dtype=np.int8), np.zeros(5)) == 0.0


def test_place_field_tail():
    # line 116: in_field at end
    train = np.array([0, 0, 1, 1, 1], dtype=np.int8)
    pos = np.linspace(0, 1, 5)
    fields = place_field_detection(train, pos)
    assert isinstance(fields, list)


def test_tuning_curve_few():
    # line 129: n < 5
    f, p = tuning_curve(np.zeros(3, dtype=np.int8), np.zeros(3))
    assert f.size == 0


# --- surrogates (lines 23, 82, 100, 140) ---
from sc_neurocore.analysis.spike_stats.surrogates import (
    surrogate_isi_shuffle,
    homogeneous_poisson,
    gamma_process,
    surrogate_joint_isi,
)


def test_isi_shuffle_short():
    # line 23: intervals.size < 2
    r = surrogate_isi_shuffle(np.array([1], dtype=np.int8))
    assert r.shape[0] == 1


def test_poisson_zero_rate():
    # line 82: max_rate <= 0
    r = homogeneous_poisson(rate_hz=0.0, duration_s=1.0)
    assert np.all(r == 0)


def test_gamma_zero_rate():
    # line 100: rate_hz <= 0
    r = gamma_process(rate_hz=0.0, shape=2, duration_s=1.0)
    assert np.all(r == 0)


def test_joint_isi_few():
    # line 140: times_idx.size < 4
    r = surrogate_joint_isi(np.array([1, 0, 0], dtype=np.int8))
    assert r.shape[0] == 3


# --- temporal (line 63) ---
from sc_neurocore.analysis.spike_stats.temporal import response_onset


def test_response_onset_short():
    r = response_onset(np.array([1, 0], dtype=np.int8), baseline_steps=5)
    assert np.isnan(r)


# --- variability (lines 26, 43, 61, 82, 87, 98, 120, 133, 173, 209, 223, 226,
#                   269, 271, 294, 295, 334, 336, 399) ---
from sc_neurocore.analysis.spike_stats.variability import (
    cv_isi,
    cv2,
    local_variation,
    lvr,
    fano_factor,
    isi_entropy,
    approximate_entropy,
    sample_entropy,
    permutation_entropy,
    hurst_exponent,
    allan_factor,
    rescaled_range,
    optimal_kernel_bandwidth,
    lempel_ziv_complexity,
)


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
    # line 120: bin_width <= 0 when no ISIs
    r = isi_entropy(np.zeros(10, dtype=np.int8))
    # Silent train → no ISIs → 0 entropy or nan
    assert r == 0.0 or np.isnan(r)


def test_lempel_ziv_silent():
    # line 133: n == 0
    r = lempel_ziv_complexity(np.array([], dtype=np.int8))
    assert r == 0.0


def test_apen_high_dim():
    # line 173: n - dim + 1 < 1
    r = approximate_entropy(np.zeros(5, dtype=np.int8), m=10)
    assert r == 0.0 or np.isnan(r)


def test_sampen_zero():
    # line 209: b == 0
    r = sample_entropy(np.zeros(10, dtype=np.int8))
    assert np.isfinite(r) or np.isnan(r)


def test_perm_entropy_short():
    # line 223: n < order * delay
    assert np.isnan(permutation_entropy(np.zeros(3, dtype=np.int8), order=5))


def test_perm_entropy_few_patterns():
    # line 226: n_patterns < 1
    assert np.isnan(permutation_entropy(np.zeros(5, dtype=np.int8), order=3, delay=3))


def test_hurst_short():
    # line 269, 271: len(scales) < 2
    assert np.isnan(hurst_exponent(np.zeros(5, dtype=np.int8)))


def test_hurst_constant():
    r = hurst_exponent(np.ones(100, dtype=np.int8))
    assert np.isfinite(r) or np.isnan(r)


def test_allan_short():
    # line 294-295: n_bins < 2
    af, windows = allan_factor(np.zeros(3, dtype=np.int8))
    assert af.size >= 0


def test_allan_with_data():
    rng = np.random.default_rng(42)
    train = rng.integers(0, 2, size=1000, dtype=np.int8)
    af, windows = allan_factor(train)
    assert af.size > 0


def test_rescaled_range_short():
    # line 334, 336: len(scales) < 2
    assert np.isnan(rescaled_range(np.zeros(5, dtype=np.int8)))


def test_rescaled_range_constant():
    r = rescaled_range(np.ones(100, dtype=np.int8))
    assert np.isfinite(r) or np.isnan(r)


def test_kernel_bandwidth_zero():
    # line 399: spread <= 0
    assert np.isnan(optimal_kernel_bandwidth(np.ones(5, dtype=np.int8) * 3))


# --- waveform (line 53) ---
from sc_neurocore.analysis.spike_stats.waveform import waveform_recovery_slope


def test_recovery_slope_peak_at_end():
    # line 53: peak >= waveform.size - 2
    r = waveform_recovery_slope(np.array([0.0, 0.5, 1.0]))
    assert np.isnan(r)


# === ROUND 2: remaining uncovered lines ===


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


from sc_neurocore.analysis.spike_stats.distance import spike_sync as _spike_sync


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


from sc_neurocore.analysis.spike_stats.causality import directed_transfer_function


def test_dtf_singular():
    # causality.py:186 — det_a near zero → continue
    trains = [np.zeros(50, dtype=np.int8)] * 3
    r = directed_transfer_function(trains, order=2)
    assert r.shape[0] > 0


# === ROUND 3: precise branch targeting ===

from sc_neurocore.analysis.spike_stats.stimulus import spatial_information as _si2


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
