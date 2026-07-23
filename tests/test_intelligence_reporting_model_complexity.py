# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestModelComplexity from former test_intelligence_reporting.py

"""Focused suite: TestModelComplexity from former test_intelligence_reporting.py."""

from __future__ import annotations

from tests.intelligence_reporting_support import *  # noqa: F403

class TestModelComplexity:
    def test_compute_bound(self):
        from sc_neurocore.compiler.intelligence import (
            classify_model_complexity,
        )

        m = classify_model_complexity({"v": "a * b + c * d - e / f"})
        assert m.classification == "compute_bound"
        assert m.recommended_paradigm == "fpga"

    def test_simple_model(self):
        from sc_neurocore.compiler.intelligence import (
            classify_model_complexity,
        )

        m = classify_model_complexity({"v": "a + b"})
        assert m.compute_ops == 1

    def test_cross_refs(self):
        from sc_neurocore.compiler.intelligence import (
            classify_model_complexity,
        )

        m = classify_model_complexity({"v": "u + w", "u": "v", "w": "v + u"})
        assert m.comm_ratio > 0
