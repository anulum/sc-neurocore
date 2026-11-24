# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAdExNeuron from former test_new_neurons.py

"""Focused suite: TestAdExNeuron from former test_new_neurons.py."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


class TestAdExNeuron:
    """The adaptive exponential integrate-and-fire neuron."""

    def test_fires_with_input(self) -> None:
        """Sustained input makes the neuron fire."""
        from sc_neurocore.neurons.models import AdExNeuron

        n = AdExNeuron()
        spikes = sum(n.step(500.0) for _ in range(2000))
        assert spikes > 0

    def test_adaptation(self) -> None:
        """The adaptation variable w grows while the neuron fires."""
        from sc_neurocore.neurons.models import AdExNeuron

        n = AdExNeuron()
        for _ in range(1000):
            n.step(400.0)
        assert n.w > 0, "adaptation variable must grow"

    def test_reset(self) -> None:
        """A reset returns v to v_rest and w to zero."""
        from sc_neurocore.neurons.models import AdExNeuron

        n = AdExNeuron()
        for _ in range(100):
            n.step(500.0)
        n.reset()
        assert abs(n.v - n.v_rest) < 1e-10
        assert abs(n.w) < 1e-10
