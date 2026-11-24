# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHindmarshRose from former test_new_neurons.py

"""Focused suite: TestHindmarshRose from former test_new_neurons.py."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


class TestHindmarshRose:
    """The Hindmarsh-Rose bursting neuron."""

    def test_bursting(self) -> None:
        """Sustained input makes the neuron fire."""
        from sc_neurocore.neurons.models import HindmarshRoseNeuron

        n = HindmarshRoseNeuron()
        spikes = sum(n.step(3.0) for _ in range(2000))
        assert spikes > 0

    def test_z_evolves(self) -> None:
        """The slow adaptation variable z moves from its initial value."""
        from sc_neurocore.neurons.models import HindmarshRoseNeuron

        n = HindmarshRoseNeuron()
        for _ in range(500):
            n.step(3.0)
        assert n.z != 2.0
