# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRallCableNetwork from former test_model_rall_cable.py

"""Focused suite: TestRallCableNetwork from former test_model_rall_cable.py."""

from __future__ import annotations

import pytest
from sc_neurocore.network.population import (
    Population,
)
from sc_neurocore.neurons.models.rall_cable import (
    RallCableNeuron,
)


class TestRallCableNetwork:
    """The RallCableNeuron in a network."""

    def test_population_incompatible(self) -> None:
        """A Population of RallCableNeuron is refused, documenting a limitation.

        The neuron has an array-valued v, and Population._sync_voltages expects a scalar v.
        """
        with pytest.raises((ValueError, TypeError)):
            Population(RallCableNeuron, n=5, label="rall")
