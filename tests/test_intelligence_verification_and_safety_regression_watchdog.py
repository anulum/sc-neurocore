# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRegressionWatchdog from former test_intelligence_verification_and_safety.py

"""Focused suite: TestRegressionWatchdog from former test_intelligence_verification_and_safety.py."""

from __future__ import annotations

from tests.intelligence_verification_and_safety_support import *  # noqa: F403

class TestRegressionWatchdog:
    def test_no_regression(self):
        from sc_neurocore.compiler.intelligence import check_regression

        r = check_regression({"area": 100}, {"area": 102})
        assert r[0].regression is False

    def test_regression(self):
        from sc_neurocore.compiler.intelligence import check_regression

        r = check_regression({"area": 100}, {"area": 120})
        assert r[0].regression is True
        assert r[0].delta_pct == 20.0
