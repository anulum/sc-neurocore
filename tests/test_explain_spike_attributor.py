# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeAttributor from former test_explain.py

"""Focused suite: TestSpikeAttributor from former test_explain.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.explain import (
    SpikeAttributor,
)
from tests.explain_support import (
    _make_spikes,
)


class TestSpikeAttributor:
    """Attributing outputs to input spikes."""

    def test_basic(self) -> None:
        """The importance map covers every step and neuron within [0, 1]."""
        spikes = _make_spikes()
        weights = [np.random.randn(4, 8) * 0.3]
        attr = SpikeAttributor(decay=0.9)
        result = attr.attribute(spikes, weights, output_neuron=0)
        assert result.importance_map.shape == (20, 8)
        assert result.importance_map.max() <= 1.0

    def test_later_spikes_more_important(self) -> None:
        """Later spikes are more important than earlier ones."""
        spikes = np.zeros((10, 4), dtype=np.int8)
        spikes[2, 0] = 1
        spikes[8, 0] = 1
        weights = [np.ones((2, 4))]
        attr = SpikeAttributor(decay=0.9)
        result = attr.attribute(spikes, weights, output_neuron=0)
        # Later spike should have higher importance (less decay)
        assert result.importance_map[8, 0] > result.importance_map[2, 0]

    def test_multi_layer_weights(self) -> None:
        """Several weight layers give an input-sized map."""
        spikes = _make_spikes(N=4)
        weights = [np.random.randn(8, 4), np.random.randn(2, 8)]
        attr = SpikeAttributor()
        result = attr.attribute(spikes, weights, output_neuron=0)
        assert result.importance_map.shape == (20, 4)

    def test_zero_spikes(self) -> None:
        """Without spikes nothing is important."""
        spikes = np.zeros((10, 4), dtype=np.int8)
        weights = [np.random.randn(2, 4)]
        attr = SpikeAttributor()
        result = attr.attribute(spikes, weights, output_neuron=0)
        assert result.importance_map.max() == 0.0
