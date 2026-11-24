# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Fixture of the former tests/advanced_plasticity_support.py

"""Provide the two-population network fixture of the advanced plasticity suites.

Registered for the whole test session by ``pytest_plugins`` in ``tests/conftest.py``. The fixture name is
unique across ``tests/`` and the installed plugins, it is not autouse, and it keeps its function scope. The
network classes are imported inside the fixture, so loading this plugin imports nothing beyond pytest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from sc_neurocore.network.network import Network
    from sc_neurocore.network.population import Population
    from sc_neurocore.network.projection import Projection


@pytest.fixture()
def simple_net() -> tuple[Network, Population, Population, Projection]:
    """Two-population network with one projection."""
    from sc_neurocore.network.network import Network
    from sc_neurocore.network.population import Population
    from sc_neurocore.network.projection import Projection

    pop_a = Population("SCLapicqueLIFNeuron", 5, label="src")
    pop_b = Population("SCLapicqueLIFNeuron", 5, label="tgt")
    proj = Projection(pop_a, pop_b, weight=0.3, probability=1.0, seed=0)
    net = Network(pop_a, pop_b, proj)
    return net, pop_a, pop_b, proj
