# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeTrainComparison from former test_viz_plots.py

"""Focused suite: TestSpikeTrainComparison from former test_viz_plots.py."""

from __future__ import annotations

from tests.viz_plots_support import *  # noqa: F403

class TestSpikeTrainComparison:
    def test_returns_axes(self):
        trains = [np.array([1, 5, 10]), np.array([2, 6, 11]), np.array([3, 7, 12])]
        ax = plots.spike_train_comparison(trains, labels=["A", "B", "C"])
        assert ax is not None
