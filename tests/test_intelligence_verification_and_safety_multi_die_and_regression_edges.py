# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMultiDieAndRegressionEdges from former test_intelligence_verification_and_safety.py

"""Focused suite: TestMultiDieAndRegressionEdges from former test_intelligence_verification_and_safety.py."""

from __future__ import annotations

from tests.intelligence_verification_and_safety_support import *  # noqa: F403

class TestMultiDieAndRegressionEdges:
    """Cover the floorplan overflow placement and the zero-baseline regression
    delta that the nominal cases leave untouched."""

    def test_oversized_block_forced_onto_last_die(self):
        from sc_neurocore.compiler.intelligence import plan_multi_die_floorplan

        # A block larger than any die's capacity cannot be placed in the first-fit
        # sweep, so it is forced onto the last die.
        fp = plan_multi_die_floorplan({"huge": 5000}, die_capacity=1000, num_dies=4)
        assert fp.die_assignment["huge"] == 3

    def test_zero_baseline_reports_zero_delta(self):
        from sc_neurocore.compiler.intelligence import check_regression

        # A zero baseline has no defined percentage change, so the delta is 0.
        checks = check_regression({"leak": 0.0}, {"leak": 5.0})
        leak = next(c for c in checks if c.metric == "leak")
        assert leak.delta_pct == 0.0
        assert leak.regression is False
