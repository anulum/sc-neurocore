# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFractionalLIF from former test_biophysical_neurons.py

"""Focused suite: TestFractionalLIF from former test_biophysical_neurons.py."""

from __future__ import annotations


class TestFractionalLIF:
    """The fractional-order LIF neuron."""

    def test_fires(self) -> None:
        """The neuron fires under drive."""
        from sc_neurocore.neurons.models import FractionalLIFNeuron

        n = FractionalLIFNeuron(alpha=0.9, resistance=5.0)
        spikes = sum(n.step(3.0) for _ in range(200))
        assert spikes > 0

    def test_memory_effect(self) -> None:
        """The neuron keeps a voltage history."""
        from sc_neurocore.neurons.models import FractionalLIFNeuron

        n = FractionalLIFNeuron(alpha=0.5)
        for _ in range(50):
            n.step(0.5)
        assert len(n._history) > 1
