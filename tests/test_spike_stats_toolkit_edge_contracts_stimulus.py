# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (stimulus) from former test_spike_stats_toolkit_edge_contracts.py

"""Edge contracts of spike-triggered covariance, spatial information, place fields and tuning curves."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.stimulus import (
    place_field_detection,
    spatial_information,
    spike_triggered_covariance,
    tuning_curve,
)


def test_stc_few_spikes() -> None:
    """Spike-triggered covariance from a single spike still has entries."""
    stim = np.random.randn(100)
    train = np.zeros(100, dtype=np.int8)
    train[50] = 1
    r = spike_triggered_covariance(stim, train, window_steps=5)
    assert r.shape[0] > 0


def test_spatial_info_few() -> None:
    """Spatial information of a silent five-sample train is zero."""
    assert spatial_information(np.zeros(5, dtype=np.int8), np.zeros(5)) == 0.0


def test_place_field_tail() -> None:
    """Place-field detection on a train firing at the end of the track returns a list."""
    train = np.array([0, 0, 1, 1, 1], dtype=np.int8)
    pos = np.linspace(0, 1, 5)
    fields = place_field_detection(train, pos)
    assert isinstance(fields, list)


def test_tuning_curve_few() -> None:
    """The tuning curve of three silent samples is empty."""
    f, p = tuning_curve(np.zeros(3, dtype=np.int8), np.zeros(3))
    assert f.size == 0
