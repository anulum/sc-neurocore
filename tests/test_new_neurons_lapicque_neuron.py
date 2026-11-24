# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLapicqueNeuron from former test_new_neurons.py

"""Focused suite: TestLapicqueNeuron from former test_new_neurons.py."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


class TestLapicqueNeuron:
    """The Lapicque integrate-and-fire neuron."""

    def test_fires(self) -> None:
        """Sustained input makes the neuron fire."""
        from sc_neurocore.neurons.models import LapicqueNeuron

        n = LapicqueNeuron()
        spikes = sum(n.step(5.0) for _ in range(200))
        assert spikes > 0

    def test_reset(self) -> None:
        """A reset returns the voltage to zero."""
        from sc_neurocore.neurons.models import LapicqueNeuron

        n = LapicqueNeuron()
        for _ in range(50):
            n.step(5.0)
        n.reset()
        assert abs(n.v) < 1e-10
