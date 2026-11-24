# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestArcaneFI from former test_model_arcane_neuron.py

"""Focused suite: TestArcaneFI from former test_model_arcane_neuron.py."""

from __future__ import annotations

from sc_neurocore.neurons.models.arcane_neuron import (
    ArcaneNeuron,
)
from tests.model_arcane_neuron_support import (
    _run,
)


class TestArcaneFI:
    """The frequency-current curve of the Arcane neuron."""

    def test_zero_silent(self) -> None:
        """Zero current gives no spikes."""
        n = ArcaneNeuron()
        assert len(_run(n, current=0.0, steps=5000)) == 0

    def test_suprathreshold_fires(self) -> None:
        """Suprathreshold current fires at least 100 times."""
        n = ArcaneNeuron()
        assert len(_run(n, current=2.0, steps=5000)) >= 100

    def test_monotonic_fi(self) -> None:
        """The firing rate does not fall as the current rises."""
        rates = []
        for I in [2.0, 3.0, 5.0]:
            n = ArcaneNeuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))
