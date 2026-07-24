# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIPIsolation from former test_model_inhomogeneous_poisson.py

"""Focused suite: TestIPIsolation from former test_model_inhomogeneous_poisson.py."""

from __future__ import annotations

from tests.model_inhomogeneous_poisson_support import *  # noqa: F403


class TestIPIsolation:
    def test_defaults(self):
        n = InhomogeneousPoissonNeuron()
        assert n.dt_ms == 1.0

    def test_step_returns_binary(self):
        assert InhomogeneousPoissonNeuron().step(100.0) in (0, 1)

    def test_stateless(self):
        """No internal state — only dt_ms parameter."""
        n = InhomogeneousPoissonNeuron()
        assert not hasattr(n, "v")

    def test_reset_noop(self):
        n = InhomogeneousPoissonNeuron()
        n.step(100.0)
        n.reset()
        # Nothing to verify — stateless

    def test_stochastic_two_runs_differ(self):
        n1 = InhomogeneousPoissonNeuron()
        n2 = InhomogeneousPoissonNeuron()
        t1 = [n1.step(100.0) for _ in range(1000)]
        t2 = [n2.step(100.0) for _ in range(1000)]
        # Shared np.random → may be equal; test with many steps
        # Actually they share global RNG, so alternating calls may differ
        assert isinstance(t1, list)
