# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeTrainCoherence from former test_spike_train_stats.py

"""Focused suite: TestSpikeTrainCoherence from former test_spike_train_stats.py."""

from __future__ import annotations

from sc_neurocore.analysis.spike_stats import (
    spike_train_coherence,
)
from tests.spike_train_stats_support import (
    _poisson_train,
)


class TestSpikeTrainCoherence:
    """The coherence of two spike trains."""

    def test_shape(self) -> None:
        """A train with itself gives a non-empty coherence with one frequency per value."""
        train = _poisson_train(100.0, 0.5)
        coh, freqs = spike_train_coherence(train, train)
        assert coh.size == freqs.size
        assert coh.size > 0
