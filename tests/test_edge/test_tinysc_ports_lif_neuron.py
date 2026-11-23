# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLifNeuron from former test_tinysc_ports.py

"""Focused suite: TestLifNeuron from former test_tinysc_ports.py."""

from __future__ import annotations

from sc_neurocore.edge.bitstream import (
    MASK32,
)
from sc_neurocore.edge.neuron import (
    LifNeuron,
)


class TestLifNeuron:
    """The integer LIF neuron of the tiny SC port."""

    def test_quiescent(self) -> None:
        """Without input the neuron stays at rest."""
        n = LifNeuron(threshold=100)
        assert not n.tick([0])
        assert n.membrane == 0

    def test_excitation(self) -> None:
        """Excitation makes the neuron spike."""
        n = LifNeuron(threshold=10, leak_shift=8)
        for _ in range(20):
            n.tick([MASK32])
        assert n.spike_count > 0

    def test_reset(self) -> None:
        """A reset clears the membrane and the spike count."""
        n = LifNeuron()
        n.membrane = 999
        n.spike_count = 5
        n.reset()
        assert n.membrane == 0
        assert n.spike_count == 0
