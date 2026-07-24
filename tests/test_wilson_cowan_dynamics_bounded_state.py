# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBoundedState from former test_wilson_cowan_dynamics.py

"""Focused suite: TestBoundedState from former test_wilson_cowan_dynamics.py."""

from __future__ import annotations

from tests.wilson_cowan_dynamics_support import *  # noqa: F403


class TestBoundedState:
    """Published 2-term sigmoid range is [-β, 1-β] where β = 1/(1+exp(aθ));
    dynamics inherit that envelope so the physically meaningful state
    range is [-β · τ_max, 1 - β · τ_max] bounded by forward-Euler
    relaxation. Empirically |E|, |I| ≤ 1 + β at defaults."""

    @pytest.mark.parametrize("drive", [-5.0, 0.0, 1.0, 5.0, 20.0])
    def test_bounds_under_drive(self, drive):
        u = WilsonCowanUnit()
        baseline = 1.0 / (1.0 + math.exp(u.a * u.theta))
        lo = -baseline - 1e-9
        hi = 1.0 + baseline + 1e-9
        for _ in range(10_000):
            u.step(drive)
            assert lo <= u.e <= hi, f"E out of bounds at drive={drive}: {u.e}"
            assert lo <= u.i <= hi, f"I out of bounds at drive={drive}: {u.i}"
