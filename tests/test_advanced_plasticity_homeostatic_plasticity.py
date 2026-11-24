# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHomeostaticPlasticity from former test_advanced_plasticity.py

"""Focused suite: TestHomeostaticPlasticity from former test_advanced_plasticity.py."""

from __future__ import annotations

from sc_neurocore.network.projection import Projection
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.learning.advanced import (
    HomeostaticPlasticity,
)


class TestHomeostaticPlasticity:
    """Homeostatic scaling of the projection weights."""

    def test_update_runs(
        self, simple_net: tuple[Network, Population, Population, Projection]
    ) -> None:
        """An update records a firing-rate estimate."""
        _, pop_a, _, _ = simple_net
        hp = HomeostaticPlasticity(target_rate=10.0, tau=100.0)
        hp.update(pop_a)
        assert hp._rate_estimate is not None
