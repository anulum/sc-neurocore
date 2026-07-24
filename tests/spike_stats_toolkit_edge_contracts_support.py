# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_spike_stats_toolkit_edge_contracts.py

from __future__ import annotations


"""Behavioural edge contracts for spike statistics helpers."""


import numpy as np


from sc_neurocore.analysis.spike_stats.basic import firing_rate, bin_spike_train


import sc_neurocore.analysis.spike_stats.distance as distance_module


import sc_neurocore.analysis.spike_stats.information as information_module


from sc_neurocore.analysis.spike_stats.causality import (
    pairwise_granger_causality,
    conditional_granger_causality,
    spectral_granger_causality,
)


from sc_neurocore.analysis.spike_stats.correlation import (
    pairwise_correlation,
    spike_train_coherence,
    event_synchronization,
    spike_time_tiling_coefficient,
    autocorrelation_time,
    coincidence_index,
)


from sc_neurocore.analysis.spike_stats.decoding import bayesian_decode


from sc_neurocore.analysis.spike_stats.dimensionality import (
    spike_train_pca,
    demixed_pca,
)


from sc_neurocore.analysis.spike_stats.distance import (
    van_rossum_distance,
    victor_purpura_distance,
    isi_distance,
    spike_distance,
    _local_isi,
    spike_sync,
    spike_sync_profile,
    spike_profile,
    adaptive_spike_distance,
    schreiber_similarity,
    hunter_milton_similarity,
    earth_movers_distance,
    multi_neuron_victor_purpura,
    generalized_victor_purpura,
    spike_distance_matrix,
    isi_profile,
)


from sc_neurocore.analysis.spike_stats.gpfa import gpfa_transform


from sc_neurocore.analysis.spike_stats.information import (
    mutual_information,
    transfer_entropy,
    spike_train_entropy,
    noise_entropy,
    stimulus_specific_information,
    kozachenko_leonenko_mi,
    time_rescaling_ks_test,
)


from sc_neurocore.analysis.spike_stats.lfp import (
    phase_locking_value,
    spike_field_coherence,
)


from sc_neurocore.analysis.spike_stats.network import (
    unitary_events,
    cell_assembly_detection,
)


from sc_neurocore.analysis.spike_stats.patterns import (
    spike_directionality,
    cubic_higher_order,
)


from sc_neurocore.analysis.spike_stats.point_process import (
    isi_hazard_function,
    isi_survivor_function,
    renewal_density,
)


from sc_neurocore.analysis.spike_stats.rate import psth


from sc_neurocore.analysis.spike_stats.sorting_quality import (
    isolation_distance,
    amplitude_cutoff,
)


from sc_neurocore.analysis.spike_stats.stimulus import (
    spike_triggered_covariance,
    spatial_information,
    place_field_detection,
    tuning_curve,
)


from sc_neurocore.analysis.spike_stats.surrogates import (
    surrogate_isi_shuffle,
    homogeneous_poisson,
    gamma_process,
    surrogate_joint_isi,
)


from sc_neurocore.analysis.spike_stats.temporal import response_onset


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


from sc_neurocore.analysis.spike_stats.waveform import waveform_recovery_slope


from sc_neurocore.analysis.spike_stats.distance import spike_sync as _spike_sync


from sc_neurocore.analysis.spike_stats.causality import directed_transfer_function


from sc_neurocore.analysis.spike_stats.stimulus import spatial_information as _si2


__all__ = [
    "np",
    "firing_rate",
    "bin_spike_train",
    "distance_module",
    "information_module",
    "pairwise_granger_causality",
    "conditional_granger_causality",
    "spectral_granger_causality",
    "pairwise_correlation",
    "spike_train_coherence",
    "event_synchronization",
    "spike_time_tiling_coefficient",
    "autocorrelation_time",
    "coincidence_index",
    "bayesian_decode",
    "spike_train_pca",
    "demixed_pca",
    "van_rossum_distance",
    "victor_purpura_distance",
    "isi_distance",
    "spike_distance",
    "_local_isi",
    "spike_sync",
    "spike_sync_profile",
    "spike_profile",
    "adaptive_spike_distance",
    "schreiber_similarity",
    "hunter_milton_similarity",
    "earth_movers_distance",
    "multi_neuron_victor_purpura",
    "generalized_victor_purpura",
    "spike_distance_matrix",
    "isi_profile",
    "gpfa_transform",
    "mutual_information",
    "transfer_entropy",
    "spike_train_entropy",
    "noise_entropy",
    "stimulus_specific_information",
    "kozachenko_leonenko_mi",
    "time_rescaling_ks_test",
    "phase_locking_value",
    "spike_field_coherence",
    "unitary_events",
    "cell_assembly_detection",
    "spike_directionality",
    "cubic_higher_order",
    "isi_hazard_function",
    "isi_survivor_function",
    "renewal_density",
    "psth",
    "isolation_distance",
    "amplitude_cutoff",
    "spike_triggered_covariance",
    "spatial_information",
    "place_field_detection",
    "tuning_curve",
    "surrogate_isi_shuffle",
    "homogeneous_poisson",
    "gamma_process",
    "surrogate_joint_isi",
    "response_onset",
    "cv_isi",
    "cv2",
    "local_variation",
    "lvr",
    "fano_factor",
    "isi_entropy",
    "approximate_entropy",
    "sample_entropy",
    "permutation_entropy",
    "hurst_exponent",
    "allan_factor",
    "rescaled_range",
    "optimal_kernel_bandwidth",
    "lempel_ziv_complexity",
    "waveform_recovery_slope",
    "_spike_sync",
    "directed_transfer_function",
    "_si2",
]
