# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExpIFNeuron from former test_new_neurons.py

"""Focused suite: TestExpIFNeuron from former test_new_neurons.py."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


class TestExpIFNeuron:
    """The exponential integrate-and-fire neuron."""

    def test_fires(self) -> None:
        """Sustained input makes the neuron fire."""
        from sc_neurocore.neurons.models import ExpIFNeuron

        n = ExpIFNeuron()
        spikes = sum(n.step(500.0) for _ in range(2000))
        assert spikes > 0

    def test_no_fire_without_input(self) -> None:
        """Without input the neuron does not fire."""
        from sc_neurocore.neurons.models import ExpIFNeuron

        n = ExpIFNeuron()
        spikes = sum(n.step(0.0) for _ in range(500))
        assert spikes == 0
