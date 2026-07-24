# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestThetaBifurcation from former test_model_theta.py

"""Focused suite: TestThetaBifurcation from former test_model_theta.py."""

from __future__ import annotations

from tests.model_theta_support import *  # noqa: F403


class TestThetaBifurcation:
    """Saddle-node bifurcation at I=0 — same as QIF."""

    def test_negative_current_silent(self) -> None:
        """I<0 → stable fixed point. No spikes."""
        for I in [-1.0, -0.5]:
            n = ThetaNeuron()
            spikes = _run(n, current=I, steps=50000)
            assert len(spikes) == 0, f"I={I}: {len(spikes)} spikes"

    def test_zero_current_silent(self) -> None:
        """I=0 → theta stays at 0 (fixed point)."""
        n = ThetaNeuron()
        for _ in range(50000):
            n.step(0.0)
        assert abs(n.theta) < 1e-10

    def test_positive_current_fires(self) -> None:
        """I>0 → periodic spiking."""
        n = ThetaNeuron()
        spikes = _run(n, current=0.5, steps=50000)
        assert len(spikes) >= 50

    def test_continuous_onset(self) -> None:
        """Rate rises continuously from zero at I=0+ (Type-I)."""
        n01 = ThetaNeuron()
        n10 = ThetaNeuron()
        s01 = len(_run(n01, current=0.1, steps=100000))
        s10 = len(_run(n10, current=1.0, steps=100000))
        assert 0 < s01 < s10

    def test_fixed_point_at_negative_I(self) -> None:
        """At I=-0.5, theta should converge to stable FP: θ* = -arccos((1+I)/(1-I))."""
        # For I=-0.5: (1+I)/(1-I) = 0.5/1.5 = 1/3, θ* = -arccos(1/3) ≈ -1.231
        n = ThetaNeuron()
        for _ in range(100000):
            n.step(-0.5)
        theta_analytical = -np.arccos(1.0 / 3.0)
        assert abs(n.theta - theta_analytical) < 0.01, (
            f"theta={n.theta:.4f}, expected={theta_analytical:.4f}"
        )
