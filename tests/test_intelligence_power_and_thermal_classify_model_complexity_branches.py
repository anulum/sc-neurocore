# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestClassifyModelComplexityBranches from former test_intelligence_power_and_thermal.py

"""Focused suite: TestClassifyModelComplexityBranches from former test_intelligence_power_and_thermal.py."""

from __future__ import annotations

from tests.intelligence_power_and_thermal_support import *  # noqa: F403


class TestClassifyModelComplexityBranches:
    """Each compute-profile class is reached by its own op-density / coupling
    signature, distinct from the default compute-bound fall-through."""

    def test_compute_bound_high_op_density(self):
        from sc_neurocore.compiler.intelligence import classify_model_complexity

        # One variable carrying five arithmetic ops -> >4 ops/var -> compute_bound.
        m = classify_model_complexity({"v": "a + b - c * d / e + f"})
        assert m.classification == "compute_bound"
        assert m.recommended_paradigm == "fpga"

    def test_memory_bound_many_low_op_variables(self):
        from sc_neurocore.compiler.intelligence import classify_model_complexity

        # Five variables with one op each -> >4 vars and <=2 ops/var -> memory_bound.
        eqs = {"a": "1+0", "b": "1+0", "c": "1+0", "d": "1+0", "e": "1+0"}
        m = classify_model_complexity(eqs)
        assert m.classification == "memory_bound"
        assert m.recommended_paradigm == "in_memory"

    def test_comm_bound_high_cross_coupling(self):
        from sc_neurocore.compiler.intelligence import classify_model_complexity

        # Three mutually-referencing variables -> comm_ratio 2.0 -> comm_bound.
        eqs = {"a": "b+c", "b": "a+c", "c": "a+b"}
        m = classify_model_complexity(eqs)
        assert m.classification == "comm_bound"
        assert m.recommended_paradigm == "cgra"
        assert m.comm_ratio > 1.5
