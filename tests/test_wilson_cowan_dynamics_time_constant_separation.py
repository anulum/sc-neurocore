# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTimeConstantSeparation from former test_wilson_cowan_dynamics.py

"""Focused suite: TestTimeConstantSeparation from former test_wilson_cowan_dynamics.py."""

from __future__ import annotations

from tests.wilson_cowan_dynamics_support import *  # noqa: F403


class TestTimeConstantSeparation:
    """τ_e < τ_i: E settles before I on a step input."""

    def test_e_reaches_target_before_i(self):
        u = WilsonCowanUnit(tau_e=1.0, tau_i=4.0)
        trace_e, trace_i = [], []
        for _ in range(800):
            u.step(5.0)
            trace_e.append(u.e)
            trace_i.append(u.i)
        # Find first time each crosses 50 % of its own final value.
        ef, iff = trace_e[-1], trace_i[-1]
        t_e = next((k for k, v in enumerate(trace_e) if v > ef * 0.5), None)
        t_i = next((k for k, v in enumerate(trace_i) if v > iff * 0.5), None)
        assert t_e is not None and t_i is not None
        assert t_e < t_i, f"E must reach 50 % before I (t_e={t_e}, t_i={t_i})"
