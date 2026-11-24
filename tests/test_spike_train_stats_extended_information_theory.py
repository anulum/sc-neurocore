# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestInformationTheory from former test_spike_train_stats_extended.py

"""Focused suite: TestInformationTheory from former test_spike_train_stats_extended.py."""

from __future__ import annotations

from typing import Any

from sc_neurocore.analysis.spike_stats import kozachenko_leonenko_mi
from sc_neurocore.analysis.spike_stats import noise_entropy
from sc_neurocore.analysis.spike_stats import spike_train_entropy
from sc_neurocore.analysis.spike_stats import stimulus_specific_information
from sc_neurocore.analysis.spike_stats import time_rescaling_ks_test
import numpy as np
from tests.spike_train_stats_extended_support import poisson_train as poisson_train


class TestInformationTheory:
    """Entropy, noise entropy, stimulus-specific information, KL mutual information and the time-rescaling test."""

    def test_spike_train_entropy(self, poisson_train: np.ndarray[Any, Any]) -> None:
        """The entropy of a Poisson train is non-negative."""
        h = spike_train_entropy(poisson_train)
        assert h >= 0

    def test_noise_entropy(self, poisson_train: np.ndarray[Any, Any]) -> None:
        """The noise entropy of a Poisson train split into five trials is finite."""
        h = noise_entropy(poisson_train, n_trials=5)
        assert np.isfinite(h)

    def test_stimulus_specific_information(self) -> None:
        """Stimulus-specific information of counts raised for one of five stimuli is non-negative."""
        rng = np.random.default_rng(5)
        counts = rng.poisson(10, 100).astype(np.float64)
        labels = np.repeat([0, 1, 2, 3, 4], 20)
        counts[labels == 0] += 5
        ssi = stimulus_specific_information(counts, labels)
        assert ssi >= 0

    def test_kozachenko_leonenko_mi(self) -> None:
        """The Kozachenko-Leonenko mutual information of a variable and its noisy copy is positive."""
        rng = np.random.default_rng(3)
        x = rng.normal(0, 1, 200)
        y = x + rng.normal(0, 0.1, 200)
        mi = kozachenko_leonenko_mi(x, y)
        assert mi > 0

    def test_time_rescaling_ks_test(self) -> None:
        """The time-rescaling KS test of 50 uniform spikes gives a statistic inside [0, 1] and a boolean verdict."""
        times = np.sort(np.random.default_rng(1).uniform(0, 1, 50))
        ks, passes = time_rescaling_ks_test(times, lambda t: 50.0)
        assert 0.0 <= ks <= 1.0
        assert isinstance(passes, bool)
