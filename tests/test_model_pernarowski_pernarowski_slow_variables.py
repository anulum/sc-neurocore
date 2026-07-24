# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPernarowskiSlowVariables from former test_model_pernarowski.py

"""Focused suite: TestPernarowskiSlowVariables from former test_model_pernarowski.py."""

from __future__ import annotations

from tests.model_pernarowski_support import *  # noqa: F403


class TestPernarowskiSlowVariables:
    def test_z_evolves_slowly(self):
        """z (eps2=0.001) should change much more slowly than w (eps1=0.1)."""
        n = PernarowskiNeuron()
        z_initial = n.z
        w_initial = n.w
        for _ in range(100):
            n.step(0.5)
        dz = abs(n.z - z_initial)
        dw = abs(n.w - w_initial)
        assert dw > 10 * dz, f"dw={dw:.6f}, dz={dz:.6f} — z should be much slower than w"

    def test_eps2_affects_dynamics(self):
        """Increasing eps2 speeds up z, changing the burst pattern."""
        n_slow = PernarowskiNeuron(eps2=0.0001)
        n_fast = PernarowskiNeuron(eps2=0.01)
        s_slow, _ = _run_and_collect(n_slow, current=0.5, steps=10000)
        s_fast, _ = _run_and_collect(n_fast, current=0.5, steps=10000)
        # Different eps2 should produce different spike counts
        assert len(s_slow) != len(s_fast), "eps2 change had no effect on spike count"

    def test_z_bounded(self):
        """Ultra-slow variable z should remain bounded."""
        n = PernarowskiNeuron()
        for _ in range(50000):
            n.step(0.5)
        assert abs(n.z) < 5.0, f"z = {n.z:.4f}, expected bounded"
