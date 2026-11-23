# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHomeostaticPlasticity from former test_learning_advanced.py

"""Focused suite: TestHomeostaticPlasticity from former test_learning_advanced.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.learning.advanced import (
    HomeostaticPlasticity,
)


class TestHomeostaticPlasticity:
    """Homeostatic plasticity."""

    def test_class_exists(self) -> None:
        """HomeostaticPlasticity should be importable."""
        assert HomeostaticPlasticity is not None

    def test_active_population_rescales_incoming_projection_weights(self) -> None:
        """An over-active population pulls its weights down."""

        # An above-target firing population drives the rate estimate positive, so
        # the controller rescales every incoming projection's weights toward the
        # target rate (clipped to the [0.9, 1.1] per-step band).
        class _Projection:
            def __init__(self) -> None:
                self.data = np.ones(4, dtype=np.float64)

        class _Population:
            def __init__(self) -> None:
                self.voltages = np.array([1.0, 1.0, 1.0, 0.0])  # mostly firing
                self._projections = [_Projection()]

        controller = HomeostaticPlasticity(target_rate=10.0, tau=1.0)
        population = _Population()

        controller.update(population)

        assert controller._rate_estimate is not None
        assert controller._rate_estimate > 0
        # Over-firing relative to target -> weights pulled down to the band floor.
        assert np.allclose(population._projections[0].data, 0.9)
        assert controller._last_scale == 0.9
