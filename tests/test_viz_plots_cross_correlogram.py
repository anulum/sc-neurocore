# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCrossCorrelogram from former test_viz_plots.py

"""Focused suite: TestCrossCorrelogram from former test_viz_plots.py."""

from __future__ import annotations

from tests.viz_plots_support import *  # noqa: F403


class TestCrossCorrelogram:
    def test_returns_axes(self, small_network):
        _, sm, _, _ = small_network
        ax = plots.cross_correlogram(sm, 0, 1, max_lag_ms=10)
        assert ax is not None
