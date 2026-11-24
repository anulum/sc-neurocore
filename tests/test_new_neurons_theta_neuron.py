# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestThetaNeuron from former test_new_neurons.py

"""Focused suite: TestThetaNeuron from former test_new_neurons.py."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


class TestThetaNeuron:
    """The theta (Ermentrout-Kopell) neuron."""

    def test_fires_above_threshold(self) -> None:
        """Input above threshold makes the neuron fire."""
        from sc_neurocore.neurons.models import ThetaNeuron

        n = ThetaNeuron()
        spikes = sum(n.step(1.0) for _ in range(1000))
        assert spikes > 0

    def test_no_fire_below(self) -> None:
        """Input below threshold gives no spikes."""
        from sc_neurocore.neurons.models import ThetaNeuron

        n = ThetaNeuron()
        spikes = sum(n.step(-1.0) for _ in range(100))
        assert spikes == 0
