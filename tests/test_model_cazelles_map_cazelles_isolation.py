# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCazellesIsolation from former test_model_cazelles_map.py

"""Focused suite: TestCazellesIsolation from former test_model_cazelles_map.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.neurons.models.cazelles_map import (
    CazellesMapNeuron,
)


class TestCazellesIsolation:
    """The CazellesMapNeuron in isolation."""

    def test_construction(self) -> None:
        """The constructor sets the documented initial state."""
        n = CazellesMapNeuron()
        assert n.x == 0.1
        assert n.alpha == 0.0

    def test_step_returns_binary(self) -> None:
        """One step returns a binary spike."""
        n = CazellesMapNeuron()
        assert n.step(0.0) in (0, 1)

    def test_spikes_under_drive(self) -> None:
        """Strong drive produces spikes."""
        n = CazellesMapNeuron()
        spikes = sum(n.step(0.0) for _ in range(5000))
        assert spikes > 50

    def test_source_orbit_visits_fast_and_slow_regimes(self) -> None:
        """The orbit visits both the fast and the slow regime."""
        n = CazellesMapNeuron()
        trace, events = n.simulate(600, backend="python")
        assert events == 7
        assert np.any(trace < n.x1)
        assert np.any(trace >= n.x1)

    def test_x_clipped(self) -> None:
        """The source orbit stays in the Figure-1 domain."""
        n = CazellesMapNeuron()
        for _ in range(10000):
            n.step(0.0)
        assert n.x0 <= n.x <= n.x4

    def test_state_finite(self) -> None:
        """The state stays finite under drive."""
        n = CazellesMapNeuron()
        for _ in range(10000):
            n.step(0.0)
        assert np.isfinite(n.x)

    def test_reset(self) -> None:
        """A reset after drive leaves a valid state."""
        n = CazellesMapNeuron()
        for _ in range(100):
            n.step(0.0)
        n.reset()
        assert n.x == 0.1
