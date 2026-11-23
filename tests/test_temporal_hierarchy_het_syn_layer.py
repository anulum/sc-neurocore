# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHetSynLayer from former test_temporal_hierarchy.py

"""Focused suite: TestHetSynLayer from former test_temporal_hierarchy.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.temporal_hierarchy import (
    HetSynLayer,
)


class TestHetSynLayer:
    """The heterogeneous-synapse layer."""

    def test_init(self) -> None:
        """Time constants and weights have one entry per synapse."""
        layer = HetSynLayer(n_inputs=8, n_neurons=4)
        assert layer.tau.shape == (4, 8)
        assert layer.W.shape == (4, 8)

    def test_step(self) -> None:
        """One step gives one spike per neuron."""
        layer = HetSynLayer(n_inputs=4, n_neurons=3, threshold=0.5)
        x = np.random.rand(4)
        spikes = layer.step(x)
        assert spikes.shape == (3,)

    def test_tau_distribution(self) -> None:
        """The time constants are positive and spread out."""
        layer = HetSynLayer(n_inputs=100, n_neurons=50, tau_mean=5.0, tau_std=1.0)
        stats = layer.tau_stats
        assert stats["mean"] > 0
        assert stats["min"] > 0
        assert stats["max"] > stats["min"]

    def test_reset(self) -> None:
        """A reset clears potentials and traces."""
        layer = HetSynLayer(n_inputs=4, n_neurons=3)
        layer.step(np.ones(4))
        layer.reset()
        assert np.allclose(layer._v, 0)
        assert np.allclose(layer._traces, 0)
