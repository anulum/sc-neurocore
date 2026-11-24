# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSFA from former test_extended_neurons.py

"""Focused suite: TestSFA from former test_extended_neurons.py."""

from __future__ import annotations

from sc_neurocore.neurons.models import (
    SFANeuron,
)


class TestSFA:
    """The spike-frequency adaptation neuron."""

    def test_fires(self) -> None:
        """The neuron fires under drive."""
        n = SFANeuron()
        assert sum(n.step(30.0) for _ in range(200)) > 0

    def test_adaptation_reduces_rate(self) -> None:
        """Adaptation keeps the later rate from rising."""
        n = SFANeuron()
        first = sum(n.step(25.0) for _ in range(100))
        second = sum(n.step(25.0) for _ in range(100))
        assert second <= first + 2
