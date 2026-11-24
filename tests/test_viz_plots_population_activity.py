# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPopulationActivity from former test_viz_plots.py

"""Focused suite: TestPopulationActivity from former test_viz_plots.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sc_neurocore.network.monitor import SpikeMonitor, StateMonitor
    from sc_neurocore.network.network import Network
    from sc_neurocore.network.projection import Projection

from tests.viz_plots_support import (
    plots,
)
from tests.viz_plots_support import small_network as small_network


class TestPopulationActivity:
    """The population-activity plot."""

    def test_returns_axes(
        self, small_network: tuple[Network, SpikeMonitor, StateMonitor, Projection]
    ) -> None:
        """The population activity of the spike monitor returns a value (the check is that it is not None)."""
        _, sm, _, _ = small_network
        ax = plots.population_activity(sm, bin_ms=5)
        assert ax is not None
