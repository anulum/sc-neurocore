# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestILIFIsolation from former test_model_ilif.py

"""Focused suite: TestILIFIsolation from former test_model_ilif.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.neurons.models.ilif import (
    InhibitoryLIFNeuron,
)


class TestILIFIsolation:
    """The InhibitoryLIFNeuron in isolation."""

    def test_defaults(self) -> None:
        """The defaults match the published model."""
        n = InhibitoryLIFNeuron()
        assert n.v == 0.0 and n.inh_trace == 0.0
        assert n.tau_m == 10.0 and n.tau_inh == 5.0
        assert n.v_threshold == 1.0 and n.inh_strength == 0.5

    def test_precomputed_alphas(self) -> None:
        """alpha_m/alpha_inh precomputed in __post_init__."""
        n = InhibitoryLIFNeuron()
        assert abs(n.alpha_m - np.exp(-1.0 / 10.0)) < 1e-14
        assert abs(n.alpha_inh - np.exp(-1.0 / 5.0)) < 1e-14

    def test_step_returns_binary(self) -> None:
        """One step returns a binary spike."""
        assert InhibitoryLIFNeuron().step(0.0) in (0, 1)

    def test_state_finite_long_run(self) -> None:
        """The state stays finite over a long run."""
        n = InhibitoryLIFNeuron()
        for _ in range(100_000):
            n.step(5.0)
        assert np.isfinite(n.v) and np.isfinite(n.inh_trace)

    def test_reset_restores_defaults(self) -> None:
        """A reset restores the defaults."""
        n = InhibitoryLIFNeuron()
        for _ in range(1000):
            n.step(5.0)
        n.reset()
        assert n.v == 0.0 and n.inh_trace == 0.0

    def test_deterministic(self) -> None:
        """The same inputs give the same trace."""
        traces = []
        for _ in range(2):
            n = InhibitoryLIFNeuron()
            trace = [(n.step(5.0), n.v, n.inh_trace) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]
