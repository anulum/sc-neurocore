# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestKLIFIsolation from former test_model_klif.py

"""Focused suite: TestKLIFIsolation from former test_model_klif.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.neurons.models.klif import (
    KLIFNeuron,
)


class TestKLIFIsolation:
    """The KLIFNeuron in isolation."""

    def test_defaults(self) -> None:
        """The defaults match the published model."""
        n = KLIFNeuron()
        assert n.v == 0.0 and n.k == 1.0
        assert n.tau == 10.0 and n.v_threshold == 1.0

    def test_precomputed_alpha(self) -> None:
        """The decay factor alpha is precomputed."""
        n = KLIFNeuron()
        assert abs(n.alpha - np.exp(-1.0 / 10.0)) < 1e-14

    def test_step_returns_binary(self) -> None:
        """One step returns a binary spike."""
        assert KLIFNeuron().step(0.0) in (0, 1)

    def test_state_finite_long_run(self) -> None:
        """The state stays finite over a long run."""
        n = KLIFNeuron()
        for _ in range(100_000):
            n.step(1.0)
        assert np.isfinite(n.v)

    def test_reset_restores_default(self) -> None:
        """Reset restores default."""
        n = KLIFNeuron()
        for _ in range(100):
            n.step(1.0)
        n.reset()
        assert n.v == 0.0

    def test_deterministic(self) -> None:
        """The same inputs give the same trace."""
        traces = []
        for _ in range(2):
            n = KLIFNeuron()
            trace = [(n.step(1.0), n.v) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]
