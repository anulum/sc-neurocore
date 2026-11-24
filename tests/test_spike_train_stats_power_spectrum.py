# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPowerSpectrum from former test_spike_train_stats.py

"""Focused suite: TestPowerSpectrum from former test_spike_train_stats.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats import (
    power_spectrum,
)
from tests.spike_train_stats_support import (
    _poisson_train,
)


class TestPowerSpectrum:
    """The power spectrum of a spike train."""

    def test_has_values(self) -> None:
        """A Poisson train gives a non-negative spectrum with one frequency per value."""
        train = _poisson_train(100.0, 1.0)
        psd, freqs = power_spectrum(train)
        assert psd.size > 0
        assert freqs.size == psd.size
        assert np.all(psd >= 0)

    def test_empty(self) -> None:
        """A one-sample train gives an empty spectrum."""
        psd, freqs = power_spectrum(np.array([0], dtype=np.uint8))
        assert psd.size == 0
