# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestParameterSweeps from former test_wilson_cowan_dynamics.py

"""Focused suite: TestParameterSweeps from former test_wilson_cowan_dynamics.py."""

from __future__ import annotations

from tests.wilson_cowan_dynamics_support import *  # noqa: F403


class TestParameterSweeps:
    def test_w_ei_scales_inhibition(self):
        finals = []
        for w_ei in (2.0, 6.0, 10.0, 15.0):
            u = WilsonCowanUnit(w_ei=w_ei)
            for _ in range(5_000):
                u.step(4.0)
            finals.append(u.e)
        # Stronger cross-inhibition from I onto E reduces E's
        # steady-state activity.
        assert finals[0] > finals[-1], f"Increasing w_ei must lower E; got {finals}"
