# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCausality from former test_spike_train_stats_extended.py

"""Focused suite: TestCausality from former test_spike_train_stats_extended.py."""

from __future__ import annotations

from typing import Any

from sc_neurocore.analysis.spike_stats import conditional_granger_causality
from sc_neurocore.analysis.spike_stats import directed_transfer_function
from sc_neurocore.analysis.spike_stats import pairwise_granger_causality
from sc_neurocore.analysis.spike_stats import partial_directed_coherence
from sc_neurocore.analysis.spike_stats import spectral_granger_causality
import numpy as np
from tests.spike_train_stats_extended_support import population as population
from tests.spike_train_stats_extended_support import two_trains as two_trains


class TestCausality:
    """Granger causality, partial directed coherence and the directed transfer function."""

    def test_pairwise_granger(
        self, two_trains: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
    ) -> None:
        """Pairwise Granger causality of two random trains is finite."""
        a, b = two_trains
        gc = pairwise_granger_causality(a, b)
        assert np.isfinite(gc)

    def test_conditional_granger(
        self, two_trains: tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]
    ) -> None:
        """Conditional Granger causality given a regular third train is finite."""
        a, b = two_trains
        c = np.zeros_like(a)
        c[::50] = 1
        gc = conditional_granger_causality(a, b, c)
        assert np.isfinite(gc)

    def test_spectral_granger(self, population: list[np.ndarray[Any, Any]]) -> None:
        """Spectral Granger causality of three trains has three rows and three columns."""
        gc = spectral_granger_causality(population[:3])
        assert gc.shape[0] == 3
        assert gc.shape[1] == 3

    def test_partial_directed_coherence(self, population: list[np.ndarray[Any, Any]]) -> None:
        """Partial directed coherence of three trains has three rows and is non-negative."""
        pdc = partial_directed_coherence(population[:3])
        assert pdc.shape[0] == 3
        assert np.all(pdc >= 0)

    def test_directed_transfer_function(self, population: list[np.ndarray[Any, Any]]) -> None:
        """The directed transfer function of three trains has three rows and is non-negative."""
        dtf = directed_transfer_function(population[:3])
        assert dtf.shape[0] == 3
        assert np.all(dtf >= 0)
