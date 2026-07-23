# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFiringRatePlot from former test_viz_plots.py

"""Focused suite: TestFiringRatePlot from former test_viz_plots.py."""

from __future__ import annotations

from tests.viz_plots_support import *  # noqa: F403

class TestFiringRatePlot:
    def test_returns_axes(self, small_network):
        _, sm, _, _ = small_network
        ax = plots.firing_rate_plot(sm, bin_ms=5)
        assert ax is not None
