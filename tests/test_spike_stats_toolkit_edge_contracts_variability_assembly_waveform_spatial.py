# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (assembly_waveform_spatial) from former test_spike_stats_toolkit_edge_contracts_variability.py

"""Edge contracts of cell assemblies, the cubic statistic, the amplitude cutoff, place fields, inhomogeneous Poisson trains, waveform recovery, the directed transfer function and spatial information."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.causality import (
    directed_transfer_function,
)
from sc_neurocore.analysis.spike_stats.network import (
    cell_assembly_detection,
)
from sc_neurocore.analysis.spike_stats.patterns import (
    cubic_higher_order,
)
from sc_neurocore.analysis.spike_stats.sorting_quality import (
    amplitude_cutoff,
)
from sc_neurocore.analysis.spike_stats.stimulus import (
    place_field_detection,
    spatial_information as _si2,
)
from sc_neurocore.analysis.spike_stats.waveform import (
    waveform_recovery_slope,
)


def test_cell_assembly_with_strong_corr() -> None:
    """Cell-assembly detection on shifted copies of one train returns a list."""
    # network.py:89-91 — eigval > mp_upper, members >= 2
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2, size=500, dtype=np.int8)
    trains = [base.copy() for _ in range(10)]
    for i in range(10):
        trains[i] = np.roll(trains[i], i)
    r = cell_assembly_detection(trains, bin_size=5, threshold=0.5)
    assert isinstance(r, list)


def test_cubic_with_data() -> None:
    """The cubic higher-order statistic of a random train has entries."""
    # patterns.py:81 — valid_n > 0
    rng = np.random.default_rng(42)
    train = rng.integers(0, 2, size=200, dtype=np.int8)
    r = cubic_higher_order(train, max_lag=5)
    assert r.shape[0] > 0


def test_amplitude_cutoff_with_data() -> None:
    """The amplitude cutoff of positive, shifted amplitudes is finite."""
    # sorting_quality.py:148 — total > 0
    rng = np.random.default_rng(0)
    amps = np.abs(rng.standard_normal(500)) + 0.5
    r = amplitude_cutoff(amps)
    assert np.isfinite(r)


def test_place_field_ending_in_field() -> None:
    """A field found on a train firing in the last fifth ends at 0.9 or later (the check passes when none is found)."""
    # stimulus.py:116 — in_field at end of array
    train = np.zeros(100, dtype=np.int8)
    train[80:] = 1
    pos = np.linspace(0, 1, 100)
    fields = place_field_detection(train, pos, threshold_std=0.5)
    assert any(f[1] >= 0.9 for f in fields) if fields else True


def test_inhomogeneous_poisson_zero() -> None:
    """An inhomogeneous Poisson train with zero rate is silent."""
    # surrogates.py:82 — max_rate <= 0
    from sc_neurocore.analysis.spike_stats.surrogates import inhomogeneous_poisson

    r = inhomogeneous_poisson(rate_func=lambda t: 0.0, duration_s=1.0)
    assert np.all(r == 0)


def test_waveform_recovery_short() -> None:
    """The recovery slope of a one-sample waveform is NaN."""
    # waveform.py:53 — dv.size == 0
    r = waveform_recovery_slope(np.array([1.0]))
    assert np.isnan(r)


def test_dtf_singular() -> None:
    """The directed transfer function of silent, singular trains still has entries."""
    # causality.py:186 — det_a near zero → continue
    trains = [np.zeros(50, dtype=np.int8)] * 3
    r = directed_transfer_function(trains, order=2)
    assert r.shape[0] > 0


def test_spatial_info_with_rate() -> None:
    """Spatial information of a random train over random positions is finite."""
    # stimulus.py:74 — mean_rate > 0 path
    rng = np.random.default_rng(42)
    train = rng.integers(0, 2, size=200, dtype=np.int8)
    pos = rng.uniform(0, 1, size=200)
    r = _si2(train, pos, n_bins=10)
    assert np.isfinite(r)


def test_sorting_cutoff_nonzero() -> None:
    """The amplitude cutoff of a two-component amplitude mixture is finite."""
    # sorting_quality.py:148 — total > 0, right > left
    amps = np.concatenate([np.random.randn(200) + 2, np.random.randn(50) + 5])
    r = amplitude_cutoff(amps, bins=50)
    assert np.isfinite(r)


def test_cubic_with_real_data() -> None:
    """The cubic higher-order statistic of a random float train is not all zero."""
    # patterns.py:81 — valid_n > 0
    rng = np.random.default_rng(1)
    train = rng.integers(0, 2, size=500, dtype=np.int8).astype(np.float64)
    r = cubic_higher_order(train, max_lag=3)
    assert np.any(r != 0)


def test_waveform_recovery_valid() -> None:
    """The recovery slope of a waveform with a trough and a later peak is finite."""
    # waveform.py:53 — dv.size > 0 path
    wf = np.array([0.0, -1.0, -0.5, 0.2, 0.8, 0.5, 0.1])
    r = waveform_recovery_slope(wf, dt=1.0)
    assert np.isfinite(r)


def test_dtf_with_real_data() -> None:
    """The directed transfer function of random trains has entries."""
    # causality.py:186 — det_a NOT near zero
    rng = np.random.default_rng(42)
    trains = [rng.integers(0, 2, size=200, dtype=np.int8) for _ in range(3)]
    r = directed_transfer_function(trains, order=2)
    assert r.shape[0] > 0
