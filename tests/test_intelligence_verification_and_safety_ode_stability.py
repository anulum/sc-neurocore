# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestODEStability from former test_intelligence_verification_and_safety.py

"""Focused suite: TestODEStability from former test_intelligence_verification_and_safety.py."""

from __future__ import annotations

from tests.intelligence_verification_and_safety_support import *  # noqa: F403

class TestODEStability:
    def test_stable(self):
        from sc_neurocore.compiler.intelligence import verify_ode_stability

        r = verify_ode_stability({"v": "a"}, dt=0.1)
        assert r.stable is True

    def test_unstable(self):
        from sc_neurocore.compiler.intelligence import verify_ode_stability

        r = verify_ode_stability(
            {"v": "a"},
            dt=100.0,
            time_constants={"v": 0.5},
        )
        assert r.stable is False

    def test_critical_dt(self):
        from sc_neurocore.compiler.intelligence import verify_ode_stability

        r = verify_ode_stability(
            {"v": "a"},
            dt=0.1,
            time_constants={"v": 10.0},
        )
        assert r.critical_dt == 20.0
