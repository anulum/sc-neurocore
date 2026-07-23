# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWave7Integration from former test_intelligence_power_and_thermal.py

"""Focused suite: TestWave7Integration from former test_intelligence_power_and_thermal.py."""

from __future__ import annotations

from tests.intelligence_power_and_thermal_support import *  # noqa: F403

class TestWave7Integration:
    def test_classify_then_recommend(self):
        from sc_neurocore.compiler.intelligence import (
            classify_model_complexity,
            recommend_target,
        )

        m = classify_model_complexity({"v": "a * b + c * d - e"})
        recs = recommend_target(
            {"v": "a * b + c * d - e"},
            require_class=m.recommended_paradigm,
        )
        assert len(recs) > 0

    def test_recommend_then_risk(self):
        from sc_neurocore.compiler.intelligence import (
            recommend_target,
            score_supply_chain_risk,
        )

        recs = recommend_target({"v": "a + b"}, top_n=1)
        risk = score_supply_chain_risk(recs[0].profile_name)
        assert risk.risk_score >= 0

    def test_bittrue_then_thermal(self):
        from sc_neurocore.compiler.intelligence import (
            generate_bittrue_kernel,
            estimate_thermal_envelope,
        )

        code = generate_bittrue_kernel("sc_lif", {"v": "a + b"})
        assert len(code) > 50
        t = estimate_thermal_envelope(power_mw=50)
        assert t.pass_fail == "PASS"

    def test_cache_workflow(self):
        from sc_neurocore.compiler.intelligence import (
            CompilationCache,
            generate_bittrue_kernel,
        )

        cache = CompilationCache()
        eqs = {"v": "a + b"}
        assert cache.get(eqs, "artix7") is None
        code = generate_bittrue_kernel("sc_lif", eqs)
        cache.put(eqs, "artix7", 16, 8, {"code": code})
        hit = cache.get(eqs, "artix7")
        assert hit["code"] == code
