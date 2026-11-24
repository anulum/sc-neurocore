# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpiNNaker2 from former test_model_spinnaker2.py

"""Focused suite: TestSpiNNaker2 from former test_model_spinnaker2.py."""

from __future__ import annotations

from sc_neurocore.neurons.models.spinnaker2 import (
    SpiNNaker2Neuron,
)


class TestSpiNNaker2:
    """Firing and fixed-point arithmetic of the SpiNNaker2Neuron."""

    def test_fires(self) -> None:
        """Sustained drive makes the neuron fire."""
        n = SpiNNaker2Neuron()
        assert sum(n.step(200) for _ in range(100)) > 0

    def test_fixed_point(self) -> None:
        """The voltage stays an integer after a step, as in fixed point."""
        n = SpiNNaker2Neuron()
        n.step(100)
        assert isinstance(n.v, int)
