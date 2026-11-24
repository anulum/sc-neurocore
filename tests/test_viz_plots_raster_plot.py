# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRasterPlot from former test_viz_plots.py

"""Focused suite: TestRasterPlot from former test_viz_plots.py."""

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


class TestRasterPlot:
    """The spike raster plot of a monitor."""

    def test_returns_axes(
        self, small_network: tuple[Network, SpikeMonitor, StateMonitor, Projection]
    ) -> None:
        """The raster plot of the spike monitor returns an object with a figure."""
        _, sm, _, _ = small_network
        ax = plots.raster_plot(sm)
        assert ax is not None
        assert hasattr(ax, "figure")
