# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for extended spike train analysis functions (88

"""Tests for extended spike train analysis functions (88 new functions)."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.analysis.spike_stats import (
    # Regularity
    lvr,
    complexity_pdf,
    optimal_bin_width,
    optimal_kernel_bandwidth,
    lempel_ziv_complexity,
    approximate_entropy,
    sample_entropy,
    permutation_entropy,
    hurst_exponent,
    allan_factor,
    rescaled_range,
    # Distance metrics
    spike_distance,
    spike_sync,
    spike_sync_profile,
    spike_profile,
    isi_profile,
    adaptive_spike_distance,
    schreiber_similarity,
    hunter_milton_similarity,
    earth_movers_distance,
    multi_neuron_victor_purpura,
    generalized_victor_purpura,
    spike_distance_matrix,
    # Synchrony
    spike_time_tiling_coefficient,
    covariance_matrix,
    autocorrelation_time,
    noise_correlation,
    signal_correlation,
    spike_count_covariance,
    joint_psth,
    coincidence_index,
    # Pattern detection
    unitary_events,
    cell_assembly_detection,
    synfire_chain_detection,
    # Information theory
    spike_train_entropy,
    noise_entropy,
    stimulus_specific_information,
    kozachenko_leonenko_mi,
    time_rescaling_ks_test,
    # Causality
    pairwise_granger_causality,
    conditional_granger_causality,
    spectral_granger_causality,
    partial_directed_coherence,
    directed_transfer_function,
    # Point process
    conditional_intensity,
    isi_hazard_function,
    isi_survivor_function,
    renewal_density,
    # Dimensionality reduction
    demixed_pca,
    factor_analysis,
    # Decoding
    bayesian_decode,
    maximum_likelihood_decode,
    linear_discriminant_decode,
    naive_bayes_decode,
    # Surrogates
    homogeneous_poisson,
    inhomogeneous_poisson,
    gamma_process,
    compound_poisson_process,
    surrogate_joint_isi,
    surrogate_bin_shuffling,
    surrogate_spike_train_shifting,
    spike_directionality,
    spike_train_order,
    # Spike sorting quality
    isolation_distance,
    l_ratio,
    silhouette_score,
    d_prime,
    isi_violation_rate,
    presence_ratio,
    amplitude_cutoff,
    snr,
    nn_hit_rate,
    drift_metric,
    # Spike-triggered / receptive field
    spike_triggered_covariance,
    spatial_information,
    place_field_detection,
    tuning_curve,
    # Non-stationarity
    change_point_detection,
    cubic_higher_order,
    # Waveform
    waveform_width,
    waveform_amplitude,
    waveform_repolarization_slope,
    waveform_recovery_slope,
    waveform_halfwidth,
    waveform_pt_ratio,
)

# ── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture()
def regular_train():
    """Binary train with regular ISIs (spike every 20 steps)."""
    t = np.zeros(2000)
    t[np.arange(20, 2000, 20)] = 1
    return t


@pytest.fixture()
def poisson_train():
    """Binary train from Poisson process."""
    rng = np.random.default_rng(42)
    return (rng.random(5000) < 0.02).astype(np.float64)


@pytest.fixture()
def two_trains():
    """Pair of binary trains for pairwise tests."""
    rng = np.random.default_rng(10)
    a = (rng.random(3000) < 0.02).astype(np.float64)
    b = (rng.random(3000) < 0.02).astype(np.float64)
    return a, b


@pytest.fixture()
def spike_times_pair():
    """Pair of spike time arrays."""
    rng = np.random.default_rng(7)
    ta = np.sort(rng.uniform(0, 1, 30))
    tb = np.sort(rng.uniform(0, 1, 25))
    return ta, tb


@pytest.fixture()
def population():
    """Population of 5 binary spike trains."""
    rng = np.random.default_rng(99)
    return [(rng.random(2000) < 0.01 + 0.005 * i).astype(np.float64) for i in range(5)]


@pytest.fixture()
def waveform_fixture():
    """Typical extracellular waveform shape: negative trough then positive peak."""
    t = np.linspace(0, 1, 60)
    return -np.sin(2 * np.pi * t) + 0.3 * np.sin(4 * np.pi * t)


# ── Regularity ───────────────────────────────────────────────────


class TestRegularity:
    def test_lvr_regular(self, regular_train):
        val = lvr(regular_train)
        assert val < 0.5

    def test_lvr_nan_empty(self):
        assert np.isnan(lvr(np.zeros(10)))

    def test_complexity_pdf(self, poisson_train):
        pdf = complexity_pdf(poisson_train)
        assert pdf.size > 0
        assert np.all(pdf >= 0)

    def test_optimal_bin_width(self, poisson_train):
        bw = optimal_bin_width(poisson_train)
        assert bw > 0

    def test_optimal_kernel_bandwidth(self, poisson_train):
        h = optimal_kernel_bandwidth(poisson_train)
        assert h > 0

    def test_lempel_ziv(self, poisson_train):
        c = lempel_ziv_complexity(poisson_train)
        assert c > 0

    def test_lempel_ziv_constant(self):
        assert lempel_ziv_complexity(np.zeros(100)) > 0

    def test_approximate_entropy(self, poisson_train):
        ae = approximate_entropy(poisson_train[:500])
        assert np.isfinite(ae)

    def test_sample_entropy(self, poisson_train):
        se = sample_entropy(poisson_train[:500])
        assert np.isfinite(se)

    def test_permutation_entropy_regular(self, regular_train):
        pe = permutation_entropy(regular_train, order=3, delay=1)
        assert 0.0 <= pe <= 1.0

    def test_hurst_exponent(self, poisson_train):
        h = hurst_exponent(poisson_train)
        assert 0.0 < h < 2.0

    def test_allan_factor(self, poisson_train):
        af, windows = allan_factor(poisson_train)
        assert af.size == windows.size
        assert af.size > 0

    def test_rescaled_range(self, poisson_train):
        h = rescaled_range(poisson_train)
        assert np.isfinite(h)


# ── Distance metrics ─────────────────────────────────────────────


class TestDistanceMetrics:
    def test_spike_distance_identical(self):
        t = np.array([0.1, 0.3, 0.5, 0.7])
        assert spike_distance(t, t) < 0.01

    def test_spike_distance_different(self, spike_times_pair):
        ta, tb = spike_times_pair
        d = spike_distance(ta, tb)
        assert d >= 0

    def test_spike_sync_identical(self):
        t = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        assert spike_sync(t, t) > 0.5

    def test_spike_sync_profile(self, spike_times_pair):
        ta, tb = spike_times_pair
        prof = spike_sync_profile(ta, tb)
        assert prof.shape == (50,)

    def test_spike_profile(self, spike_times_pair):
        ta, tb = spike_times_pair
        prof = spike_profile(ta, tb)
        assert prof.shape == (50,)

    def test_isi_profile(self, two_trains):
        a, b = two_trains
        prof = isi_profile(a, b)
        assert prof.shape == (50,)

    def test_adaptive_spike_distance(self, spike_times_pair):
        ta, tb = spike_times_pair
        d = adaptive_spike_distance(ta, tb)
        assert d >= 0

    def test_schreiber_similarity_identical(self, poisson_train):
        s = schreiber_similarity(poisson_train, poisson_train)
        assert s > 0.99

    def test_hunter_milton(self, spike_times_pair):
        ta, tb = spike_times_pair
        s = hunter_milton_similarity(ta, tb, dt_max=0.05)
        assert 0.0 <= s <= 1.0

    def test_earth_movers_distance(self, spike_times_pair):
        ta, tb = spike_times_pair
        d = earth_movers_distance(ta, tb)
        assert d >= 0

    def test_multi_neuron_victor_purpura(self, spike_times_pair):
        ta, tb = spike_times_pair
        mat = multi_neuron_victor_purpura([ta, tb])
        assert mat.shape == (2, 2)
        assert mat[0, 0] == 0.0
        assert mat[0, 1] == mat[1, 0]

    def test_generalized_victor_purpura(self, spike_times_pair):
        ta, tb = spike_times_pair
        d = generalized_victor_purpura(ta, tb)
        assert d >= 0

    def test_spike_distance_matrix(self, spike_times_pair):
        ta, tb = spike_times_pair
        mat = spike_distance_matrix([ta, tb])
        assert mat.shape == (2, 2)
        assert mat[0, 1] == mat[1, 0]


# ── Synchrony ────────────────────────────────────────────────────


class TestSynchrony:
    def test_sttc_identical(self, poisson_train):
        val = spike_time_tiling_coefficient(poisson_train, poisson_train)
        assert val > 0.5

    def test_sttc_independent(self, two_trains):
        a, b = two_trains
        val = spike_time_tiling_coefficient(a, b)
        assert -1.0 <= val <= 1.0

    def test_sttc_silent_train_is_zero(self) -> None:
        # With no spikes in one train the tiling coefficient is undefined and
        # collapses to zero rather than indexing an empty spike-time array.
        silent = np.zeros(200, dtype=np.float64)
        active = np.zeros(200, dtype=np.float64)
        active[::20] = 1.0
        assert spike_time_tiling_coefficient(silent, active) == 0.0

    def test_covariance_matrix(self, population):
        cov = covariance_matrix(population)
        assert cov.shape[0] == 5

    def test_autocorrelation_time(self, poisson_train):
        tau = autocorrelation_time(poisson_train)
        assert tau >= 0

    def test_noise_correlation(self, population):
        nc = noise_correlation(population)
        assert nc.shape == (5, 5)
        assert np.allclose(np.diag(nc), 1.0)

    def test_signal_correlation(self, population):
        sc = signal_correlation(population)
        assert sc.shape == (5, 5)

    def test_spike_count_covariance(self, population):
        cov = spike_count_covariance(population)
        assert cov.shape[0] == 5

    def test_joint_psth(self, two_trains):
        a, b = two_trains
        jp = joint_psth(a, b)
        assert jp.ndim == 2

    def test_coincidence_index(self, two_trains):
        a, b = two_trains
        ci = coincidence_index(a, b)
        assert np.isfinite(ci)


# ── Pattern detection ────────────────────────────────────────────


class TestPatternDetection:
    def test_unitary_events(self, population):
        events = unitary_events(population[:3])
        assert isinstance(events, list)

    def test_cell_assembly_detection(self, population):
        assemblies = cell_assembly_detection(population)
        assert isinstance(assemblies, list)

    def test_synfire_chain_detection(self, population):
        chains = synfire_chain_detection(population)
        assert isinstance(chains, list)


# ── Information theory ───────────────────────────────────────────


class TestInformationTheory:
    def test_spike_train_entropy(self, poisson_train):
        h = spike_train_entropy(poisson_train)
        assert h >= 0

    def test_noise_entropy(self, poisson_train):
        h = noise_entropy(poisson_train, n_trials=5)
        assert np.isfinite(h)

    def test_stimulus_specific_information(self):
        rng = np.random.default_rng(5)
        counts = rng.poisson(10, 100).astype(np.float64)
        labels = np.repeat([0, 1, 2, 3, 4], 20)
        counts[labels == 0] += 5
        ssi = stimulus_specific_information(counts, labels)
        assert ssi >= 0

    def test_kozachenko_leonenko_mi(self):
        rng = np.random.default_rng(3)
        x = rng.normal(0, 1, 200)
        y = x + rng.normal(0, 0.1, 200)
        mi = kozachenko_leonenko_mi(x, y)
        assert mi > 0

    def test_time_rescaling_ks_test(self):
        times = np.sort(np.random.default_rng(1).uniform(0, 1, 50))
        ks, passes = time_rescaling_ks_test(times, lambda t: 50.0)
        assert 0.0 <= ks <= 1.0
        assert isinstance(passes, bool)


# ── Causality ────────────────────────────────────────────────────


class TestCausality:
    def test_pairwise_granger(self, two_trains):
        a, b = two_trains
        gc = pairwise_granger_causality(a, b)
        assert np.isfinite(gc)

    def test_conditional_granger(self, two_trains):
        a, b = two_trains
        c = np.zeros_like(a)
        c[::50] = 1
        gc = conditional_granger_causality(a, b, c)
        assert np.isfinite(gc)

    def test_spectral_granger(self, population):
        gc = spectral_granger_causality(population[:3])
        assert gc.shape[0] == 3
        assert gc.shape[1] == 3

    def test_partial_directed_coherence(self, population):
        pdc = partial_directed_coherence(population[:3])
        assert pdc.shape[0] == 3
        assert np.all(pdc >= 0)

    def test_directed_transfer_function(self, population):
        dtf = directed_transfer_function(population[:3])
        assert dtf.shape[0] == 3
        assert np.all(dtf >= 0)


# ── Point process ────────────────────────────────────────────────


class TestPointProcess:
    def test_conditional_intensity(self, poisson_train):
        ci = conditional_intensity(poisson_train)
        assert ci.size == poisson_train.size
        assert np.all(ci >= 0)

    def test_isi_hazard_function(self, poisson_train):
        h, centers = isi_hazard_function(poisson_train)
        assert h.size == centers.size
        assert h.size > 0

    def test_isi_survivor_function(self, poisson_train):
        s, centers = isi_survivor_function(poisson_train)
        assert s.size > 0
        assert s[0] >= s[-1]

    def test_renewal_density(self, poisson_train):
        d, centers = renewal_density(poisson_train)
        assert d.size == centers.size


# ── Dimensionality reduction ─────────────────────────────────────


class TestDimensionality:
    def test_demixed_pca(self, population):
        conds = {
            0: population[:3],
            1: population[2:],
        }
        proj, explained = demixed_pca(conds, n_components=2)
        assert proj.ndim == 2
        assert explained.size == 2

    def test_factor_analysis(self, population):
        loadings, psi = factor_analysis(population, n_factors=2)
        assert loadings.shape == (5, 2)
        assert psi.size == 5


# ── Decoding ─────────────────────────────────────────────────────


class TestDecoding:
    def test_bayesian_decode(self):
        tuning = np.array([[10.0, 1.0], [1.0, 10.0], [5.0, 5.0]])
        counts = np.array([9, 2])
        result = bayesian_decode(counts, tuning)
        assert result == 0

    def test_maximum_likelihood_decode(self):
        tuning = np.array([[10.0, 1.0], [1.0, 10.0]])
        counts = np.array([1, 12])
        result = maximum_likelihood_decode(counts, tuning)
        assert result == 1

    def test_linear_discriminant_decode(self):
        rng = np.random.default_rng(8)
        train = np.vstack([rng.normal(0, 1, (20, 3)), rng.normal(3, 1, (20, 3))])
        labels = np.concatenate([np.zeros(20), np.ones(20)])
        test = np.array([3.0, 3.0, 3.0])
        pred = linear_discriminant_decode(train, labels, test)
        assert pred == 1

    def test_naive_bayes_decode(self):
        rng = np.random.default_rng(12)
        train = np.vstack([rng.normal(-2, 0.5, (30, 2)), rng.normal(2, 0.5, (30, 2))])
        labels = np.concatenate([np.zeros(30), np.ones(30)])
        test = np.array([2.0, 2.0])
        pred = naive_bayes_decode(train, labels, test)
        assert pred == 1


# ── Surrogates ───────────────────────────────────────────────────


class TestSurrogates:
    def test_homogeneous_poisson(self):
        t = homogeneous_poisson(20.0, 1.0)
        assert t.size == 1000
        rate = t.sum() / 1.0
        assert 5 < rate < 50

    def test_inhomogeneous_poisson(self):
        t = inhomogeneous_poisson(lambda x: 20.0 + 10.0 * np.sin(2 * np.pi * x), 1.0)
        assert t.size == 1000

    def test_gamma_process_shape1(self):
        t = gamma_process(20.0, 1.0, 1.0)
        rate = t.sum()
        assert 5 < rate < 50

    def test_gamma_process_regular(self):
        t = gamma_process(20.0, 10.0, 1.0)
        assert t.sum() > 0

    def test_compound_poisson(self):
        t = compound_poisson_process(10.0, 3.0, 1.0)
        assert t.sum() > 0

    def test_surrogate_joint_isi(self, poisson_train):
        surr = surrogate_joint_isi(poisson_train)
        assert surr.size == poisson_train.size
        assert surr.sum() > 0

    def test_surrogate_bin_shuffling(self, poisson_train):
        surr = surrogate_bin_shuffling(poisson_train)
        assert surr.sum() == poisson_train.sum()

    def test_surrogate_spike_train_shifting(self, poisson_train):
        surr = surrogate_spike_train_shifting(poisson_train)
        assert surr.sum() == poisson_train.sum()

    def test_spike_directionality(self):
        ta = np.array([0.1, 0.2, 0.3, 0.4])
        tb = np.array([0.15, 0.25, 0.35, 0.45])
        d = spike_directionality(ta, tb)
        assert -1.0 <= d <= 1.0

    def test_spike_train_order(self, spike_times_pair):
        ta, tb = spike_times_pair
        mat = spike_train_order([ta, tb])
        assert mat.shape == (2, 2)
        assert np.isclose(mat[0, 1], -mat[1, 0])


# ── Spike sorting quality ────────────────────────────────────────


class TestSpikeSortingQuality:
    @pytest.fixture()
    def clusters(self):
        rng = np.random.default_rng(0)
        c = rng.normal(0, 0.5, (50, 3))
        n = rng.normal(5, 1.0, (100, 3))
        return c, n

    def test_isolation_distance(self, clusters):
        c, n = clusters
        iso = isolation_distance(c, n)
        assert iso > 0

    def test_l_ratio(self, clusters):
        c, n = clusters
        lr = l_ratio(c, n)
        assert np.isfinite(lr)

    def test_silhouette_score(self):
        rng = np.random.default_rng(1)
        data = np.vstack([rng.normal(0, 0.3, (30, 2)), rng.normal(5, 0.3, (30, 2))])
        labels = np.concatenate([np.zeros(30), np.ones(30)])
        s = silhouette_score(data, labels)
        assert s > 0.5

    def test_d_prime(self, clusters):
        c, n = clusters
        dp = d_prime(c, n)
        assert dp > 0

    def test_isi_violation_rate(self, regular_train):
        vr = isi_violation_rate(regular_train)
        assert vr == 0.0

    def test_isi_violation_rate_with_violations(self):
        t = np.zeros(1000)
        t[[10, 11, 12, 50, 51, 200, 400]] = 1
        vr = isi_violation_rate(t, dt=0.001, refractory_ms=1.5)
        assert vr > 0

    def test_presence_ratio(self, poisson_train):
        pr = presence_ratio(poisson_train)
        assert 0.0 <= pr <= 1.0

    def test_amplitude_cutoff(self):
        rng = np.random.default_rng(4)
        amps = np.abs(rng.normal(100, 20, 500))
        ac = amplitude_cutoff(amps)
        assert 0.0 <= ac <= 1.0

    def test_snr(self):
        rng = np.random.default_rng(6)
        mean_wf = np.sin(np.linspace(0, 2 * np.pi, 40))
        waveforms = mean_wf[None, :] + rng.normal(0, 0.1, (100, 40))
        s = snr(waveforms)
        assert s > 1.0

    def test_nn_hit_rate(self, clusters):
        c, n = clusters
        hr = nn_hit_rate(c, n)
        assert 0.0 <= hr <= 1.0

    def test_drift_metric(self):
        rng = np.random.default_rng(2)
        wf = rng.normal(0, 1, (200, 30))
        wf[:100] *= 2.0
        ts = np.arange(200, dtype=np.float64)
        dm = drift_metric(wf, ts)
        assert dm > 0


# ── Spike-triggered / receptive field ────────────────────────────


class TestReceptiveField:
    def test_spike_triggered_covariance(self, poisson_train):
        stim = np.random.default_rng(3).normal(0, 1, poisson_train.size)
        stc = spike_triggered_covariance(stim, poisson_train, window_steps=20)
        assert stc.shape == (20, 20)

    def test_spatial_information(self, poisson_train):
        positions = np.linspace(0, 100, poisson_train.size)
        si = spatial_information(poisson_train, positions)
        assert si >= 0

    def test_place_field_detection(self):
        rng = np.random.default_rng(9)
        n = 5000
        positions = np.linspace(0, 100, n)
        train = (rng.random(n) < 0.005).astype(np.float64)
        train[(positions > 40) & (positions < 60)] += (
            rng.random(int(((positions > 40) & (positions < 60)).sum())) < 0.1
        ).astype(np.float64)
        train = np.clip(train, 0, 1)
        fields = place_field_detection(train, positions, threshold_std=1.5)
        assert isinstance(fields, list)

    def test_tuning_curve(self, poisson_train):
        stim = np.sin(np.linspace(0, 4 * np.pi, poisson_train.size))
        rates, centers = tuning_curve(poisson_train, stim)
        assert rates.size == centers.size
        assert rates.size > 0


# ── Non-stationarity ─────────────────────────────────────────────


class TestNonstationarity:
    def test_change_point_detection(self):
        t = np.zeros(2000)
        rng = np.random.default_rng(11)
        t[:1000] = (rng.random(1000) < 0.01).astype(np.float64)
        t[1000:] = (rng.random(1000) < 0.1).astype(np.float64)
        cps = change_point_detection(t, bin_size=50, threshold=3.0)
        assert isinstance(cps, list)
        assert len(cps) > 0

    def test_cubic_higher_order(self, poisson_train):
        c3 = cubic_higher_order(poisson_train, max_lag=10)
        assert c3.shape == (10, 10)


# ── Waveform analysis ────────────────────────────────────────────


class TestWaveform:
    def test_waveform_width(self, waveform_fixture):
        w = waveform_width(waveform_fixture, dt=1.0 / 60)
        assert w > 0

    def test_waveform_amplitude(self, waveform_fixture):
        a = waveform_amplitude(waveform_fixture)
        assert a > 0

    def test_waveform_repolarization_slope(self, waveform_fixture):
        s = waveform_repolarization_slope(waveform_fixture, dt=1.0 / 60)
        assert s > 0

    def test_waveform_recovery_slope(self, waveform_fixture):
        s = waveform_recovery_slope(waveform_fixture, dt=1.0 / 60)
        assert np.isfinite(s)

    def test_waveform_halfwidth(self, waveform_fixture):
        hw = waveform_halfwidth(waveform_fixture, dt=1.0 / 60)
        assert hw > 0

    def test_waveform_pt_ratio(self, waveform_fixture):
        r = waveform_pt_ratio(waveform_fixture)
        assert r > 0
