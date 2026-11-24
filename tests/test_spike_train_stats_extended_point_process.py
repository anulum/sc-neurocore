# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPointProcess from former test_spike_train_stats_extended.py

"""Focused suite: TestPointProcess from former test_spike_train_stats_extended.py."""

from __future__ import annotations

from typing import Any

from sc_neurocore.analysis.spike_stats import conditional_intensity
from sc_neurocore.analysis.spike_stats import isi_hazard_function
from sc_neurocore.analysis.spike_stats import isi_survivor_function
from sc_neurocore.analysis.spike_stats import renewal_density
import numpy as np
from tests.spike_train_stats_extended_support import poisson_train as poisson_train


class TestPointProcess:
    """The conditional intensity and the ISI hazard, survivor and renewal-density functions."""

    def test_conditional_intensity(self, poisson_train: np.ndarray[Any, Any]) -> None:
        """The conditional intensity of a Poisson train has one non-negative value per step."""
        ci = conditional_intensity(poisson_train)
        assert ci.size == poisson_train.size
        assert np.all(ci >= 0)

    def test_isi_hazard_function(self, poisson_train: np.ndarray[Any, Any]) -> None:
        """The ISI hazard function has one value per bin centre and is not empty."""
        h, centers = isi_hazard_function(poisson_train)
        assert h.size == centers.size
        assert h.size > 0

    def test_isi_survivor_function(self, poisson_train: np.ndarray[Any, Any]) -> None:
        """The ISI survivor function is not empty and does not end above its start."""
        s, centers = isi_survivor_function(poisson_train)
        assert s.size > 0
        assert s[0] >= s[-1]

    def test_renewal_density(self, poisson_train: np.ndarray[Any, Any]) -> None:
        """The renewal density has one value per bin centre."""
        d, centers = renewal_density(poisson_train)
        assert d.size == centers.size
