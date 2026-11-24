# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQuadraticIF from former test_new_neurons.py

"""Focused suite: TestQuadraticIF from former test_new_neurons.py."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


class TestQuadraticIF:
    """The quadratic integrate-and-fire neuron."""

    def test_fires(self) -> None:
        """Sustained input makes the neuron fire."""
        from sc_neurocore.neurons.models import QuadraticIFNeuron

        n = QuadraticIFNeuron()
        spikes = sum(n.step(0.5) for _ in range(500))
        assert spikes > 0

    def test_no_fire_subthreshold(self) -> None:
        """Started and reset below threshold, the neuron does not fire."""
        from sc_neurocore.neurons.models import QuadraticIFNeuron

        n = QuadraticIFNeuron(v=-2.0, v_reset=-2.0)
        spikes = sum(n.step(-1.0) for _ in range(100))
        assert spikes == 0
