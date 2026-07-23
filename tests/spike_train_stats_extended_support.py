# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_spike_train_stats_extended.py

from __future__ import annotations

"""Tests for extended spike train analysis functions (88 new functions)."""
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

__all__ = ['np', 'pytest', 'lvr', 'complexity_pdf', 'optimal_bin_width', 'optimal_kernel_bandwidth', 'lempel_ziv_complexity', 'approximate_entropy', 'sample_entropy', 'permutation_entropy', 'hurst_exponent', 'allan_factor', 'rescaled_range', 'spike_distance', 'spike_sync', 'spike_sync_profile', 'spike_profile', 'isi_profile', 'adaptive_spike_distance', 'schreiber_similarity', 'hunter_milton_similarity', 'earth_movers_distance', 'multi_neuron_victor_purpura', 'generalized_victor_purpura', 'spike_distance_matrix', 'spike_time_tiling_coefficient', 'covariance_matrix', 'autocorrelation_time', 'noise_correlation', 'signal_correlation', 'spike_count_covariance', 'joint_psth', 'coincidence_index', 'unitary_events', 'cell_assembly_detection', 'synfire_chain_detection', 'spike_train_entropy', 'noise_entropy', 'stimulus_specific_information', 'kozachenko_leonenko_mi', 'time_rescaling_ks_test', 'pairwise_granger_causality', 'conditional_granger_causality', 'spectral_granger_causality', 'partial_directed_coherence', 'directed_transfer_function', 'conditional_intensity', 'isi_hazard_function', 'isi_survivor_function', 'renewal_density', 'demixed_pca', 'factor_analysis', 'bayesian_decode', 'maximum_likelihood_decode', 'linear_discriminant_decode', 'naive_bayes_decode', 'homogeneous_poisson', 'inhomogeneous_poisson', 'gamma_process', 'compound_poisson_process', 'surrogate_joint_isi', 'surrogate_bin_shuffling', 'surrogate_spike_train_shifting', 'spike_directionality', 'spike_train_order', 'isolation_distance', 'l_ratio', 'silhouette_score', 'd_prime', 'isi_violation_rate', 'presence_ratio', 'amplitude_cutoff', 'snr', 'nn_hit_rate', 'drift_metric', 'spike_triggered_covariance', 'spatial_information', 'place_field_detection', 'tuning_curve', 'change_point_detection', 'cubic_higher_order', 'waveform_width', 'waveform_amplitude', 'waveform_repolarization_slope', 'waveform_recovery_slope', 'waveform_halfwidth', 'waveform_pt_ratio', 'regular_train', 'poisson_train', 'two_trains', 'spike_times_pair', 'population', 'waveform_fixture', '__all__']
