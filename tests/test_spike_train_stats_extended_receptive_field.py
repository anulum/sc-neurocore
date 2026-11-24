# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestReceptiveField from former test_spike_train_stats_extended.py

"""Focused suite: TestReceptiveField from former test_spike_train_stats_extended.py."""

from __future__ import annotations

from typing import Any

from sc_neurocore.analysis.spike_stats import place_field_detection
from sc_neurocore.analysis.spike_stats import spatial_information
from sc_neurocore.analysis.spike_stats import spike_triggered_covariance
from sc_neurocore.analysis.spike_stats import tuning_curve
import numpy as np
from tests.spike_train_stats_extended_support import poisson_train as poisson_train


class TestReceptiveField:
    """Spike-triggered covariance, spatial information, place fields and tuning curves."""

    def test_spike_triggered_covariance(self, poisson_train: np.ndarray[Any, Any]) -> None:
        """Spike-triggered covariance over a 20-step window is 20x20."""
        stim = np.random.default_rng(3).normal(0, 1, poisson_train.size)
        stc = spike_triggered_covariance(stim, poisson_train, window_steps=20)
        assert stc.shape == (20, 20)

    def test_spatial_information(self, poisson_train: np.ndarray[Any, Any]) -> None:
        """Spatial information of a Poisson train along a linear track is non-negative."""
        positions = np.linspace(0, 100, poisson_train.size)
        si = spatial_information(poisson_train, positions)
        assert si >= 0

    def test_place_field_detection(self) -> None:
        """Place-field detection on a train with a raised rate between 40 and 60 returns a list (the check is on the type only)."""
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

    def test_tuning_curve(self, poisson_train: np.ndarray[Any, Any]) -> None:
        """The tuning curve against a sinusoidal stimulus has one rate per bin centre and is not empty."""
        stim = np.sin(np.linspace(0, 4 * np.pi, poisson_train.size))
        rates, centers = tuning_curve(poisson_train, stim)
        assert rates.size == centers.size
        assert rates.size > 0
