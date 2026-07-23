# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPospischilGating from former test_model_pospischil.py

"""Focused suite: TestPospischilGating from former test_model_pospischil.py."""

from __future__ import annotations

from tests.model_pospischil_support import *  # noqa: F403

class TestPospischilGating:
    def test_gating_bounded(self):
        """m, h, n, p should stay approximately in [0, 1]."""
        n = PospischilNeuron()
        for _ in range(50000):
            n.step(10.0)
        for name, val in [("m", n.m), ("h", n.h), ("n", n.n), ("p", n.p)]:
            assert -0.01 <= val <= 1.01, f"{name} = {val:.6f}"

    @pytest.mark.parametrize("dt", [0.01, 0.025, 0.05])
    def test_dt_stability(self, dt: float):
        n = PospischilNeuron(dt=dt)
        for _ in range(20000):
            n.step(10.0)
        assert np.isfinite(n.v)
