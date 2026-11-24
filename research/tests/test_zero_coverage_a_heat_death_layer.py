# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHeatDeathLayer from former test_zero_coverage_a.py

"""Focused suite: TestHeatDeathLayer from former test_zero_coverage_a.py."""

from __future__ import annotations

import numpy as np


class TestHeatDeathLayer:
    """The heat-death layer research module."""

    def test_compute_step(self) -> None:
        """A compute step returns an array (the check is on the type only)."""
        from eschaton.heat_death import HeatDeathLayer

        h = HeatDeathLayer(initial_energy=1.0, entropy_rate=0.01)
        bs = np.random.randint(0, 2, 64).astype(np.uint8)
        result = h.compute_step(bs)
        assert isinstance(result, np.ndarray)

    def test_status(self) -> None:
        """The status is a string (the check is on the type only)."""
        from eschaton.heat_death import HeatDeathLayer

        h = HeatDeathLayer()
        assert isinstance(h.status(), str)
