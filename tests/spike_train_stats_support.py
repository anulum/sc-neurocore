# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_spike_train_stats.py

from __future__ import annotations

"""Tests for spike train analysis toolkit."""
import numpy as np
import pytest
from sc_neurocore.analysis.spike_stats import (
    spike_times,
    isi,
    firing_rate,
    cv_isi,
    fano_factor,
    spike_count,
    psth,
    cross_correlation,
    pairwise_correlation,
    power_spectrum,
    burst_detection,
    instantaneous_rate,
    van_rossum_distance,
    victor_purpura_distance,
    isi_distance,
    cv2,
    local_variation,
    isi_entropy,
    event_synchronization,
    spike_train_coherence,
    first_spike_latency,
    response_onset,
    spike_triggered_average,
    bin_spike_train,
    population_rate,
    surrogate_isi_shuffle,
    surrogate_dither,
    surrogate_trial_shuffle,
    mutual_information,
    transfer_entropy,
    phase_locking_value,
    spike_field_coherence,
    spike_phase_histogram,
    spike_train_pca,
    population_vector_decode,
    functional_connectivity,
    significance_bootstrap,
)
def _poisson_train(rate_hz: float, duration_s: float, dt: float = 0.001, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = int(duration_s / dt)
    return (rng.random(n) < rate_hz * dt).astype(np.uint8)

__all__ = ['np', 'pytest', 'spike_times', 'isi', 'firing_rate', 'cv_isi', 'fano_factor', 'spike_count', 'psth', 'cross_correlation', 'pairwise_correlation', 'power_spectrum', 'burst_detection', 'instantaneous_rate', 'van_rossum_distance', 'victor_purpura_distance', 'isi_distance', 'cv2', 'local_variation', 'isi_entropy', 'event_synchronization', 'spike_train_coherence', 'first_spike_latency', 'response_onset', 'spike_triggered_average', 'bin_spike_train', 'population_rate', 'surrogate_isi_shuffle', 'surrogate_dither', 'surrogate_trial_shuffle', 'mutual_information', 'transfer_entropy', 'phase_locking_value', 'spike_field_coherence', 'spike_phase_histogram', 'spike_train_pca', 'population_vector_decode', 'functional_connectivity', 'significance_bootstrap', '_poisson_train', '__all__']
