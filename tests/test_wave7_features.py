# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2020–2026 Miroslav Šotek. All rights reserved.
# SC-NeuroCore — Wave 7 test suite

"""Tests for Wave 7: frontier platforms + 8 compiler intelligence features."""

import pytest


# ═══════════════════════════════════════════════════════════════════════
# 1. Frontier Platform Classes
# ═══════════════════════════════════════════════════════════════════════

class TestFrontierPlatforms:
    """Verify 4 new platform classes and 10 new profiles."""

    @pytest.mark.parametrize("name", [
        "finalspark_neuroplatform", "cortical_labs_dishbrain",
    ])
    def test_biological(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile
        p = get_profile(name)
        assert p.platform_class == "biological"

    @pytest.mark.parametrize("name", [
        "ibm_ecram", "samsung_pcram", "stanford_ecram",
    ])
    def test_electrochemical(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile
        p = get_profile(name)
        assert p.platform_class == "electrochemical"

    @pytest.mark.parametrize("name", [
        "cerebras_wse3_ws", "tesla_dojo3", "tachyum_prodigy",
    ])
    def test_wafer_scale(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile
        p = get_profile(name)
        assert p.platform_class == "wafer_scale"

    @pytest.mark.parametrize("name", [
        "aspinity_aml100", "renesas_analog_ai",
    ])
    def test_analog_mixed(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile
        p = get_profile(name)
        assert p.platform_class == "analog_mixed"

    def test_total_profiles(self):
        from sc_neurocore.compiler.hardware_profiles import list_profile_names
        assert len(list_profile_names()) >= 144

    def test_total_classes(self):
        from sc_neurocore.compiler.hardware_profiles import (
            list_profile_names, get_profile,
        )
        classes = {get_profile(n).platform_class for n in list_profile_names()}
        assert len(classes) >= 19


# ═══════════════════════════════════════════════════════════════════════
# 2. Auto-Target Recommender
# ═══════════════════════════════════════════════════════════════════════

class TestAutoTargetRecommender:

    def test_basic_recommendation(self):
        from sc_neurocore.compiler.advanced_features import recommend_target
        recs = recommend_target({"v": "a + b"})
        assert len(recs) == 5
        assert recs[0].score >= recs[-1].score

    def test_class_filter(self):
        from sc_neurocore.compiler.advanced_features import recommend_target
        recs = recommend_target(
            {"v": "a * b + c"},
            require_class="neuromorphic",
        )
        for r in recs:
            assert "neuromorphic" in r.rationale

    def test_width_filter(self):
        from sc_neurocore.compiler.advanced_features import recommend_target
        recs = recommend_target({"v": "a + b"}, max_data_width=8)
        from sc_neurocore.compiler.hardware_profiles import get_profile
        for r in recs:
            assert get_profile(r.profile_name).data_width <= 8

    def test_top_n(self):
        from sc_neurocore.compiler.advanced_features import recommend_target
        recs = recommend_target({"v": "a + b"}, top_n=3)
        assert len(recs) == 3


# ═══════════════════════════════════════════════════════════════════════
# 3. Partial Reconfiguration Planner
# ═══════════════════════════════════════════════════════════════════════

class TestPartialReconfig:

    def test_basic_plan(self):
        from sc_neurocore.compiler.advanced_features import (
            plan_partial_reconfiguration,
        )
        plan = plan_partial_reconfiguration({"v": "a", "u": "b"})
        assert plan.total_regions == 2
        assert plan.bitstream_count == 2
        assert len(plan.schedule) == 2

    def test_single_var(self):
        from sc_neurocore.compiler.advanced_features import (
            plan_partial_reconfiguration,
        )
        plan = plan_partial_reconfiguration({"v": "a"})
        assert plan.total_regions == 1

    def test_custom_slots(self):
        from sc_neurocore.compiler.advanced_features import (
            plan_partial_reconfiguration,
        )
        plan = plan_partial_reconfiguration(
            {"v": "a", "u": "b", "w": "c"},
            time_slots=4,
        )
        assert plan.bitstream_count == 4


# ═══════════════════════════════════════════════════════════════════════
# 4. Supply Chain Risk
# ═══════════════════════════════════════════════════════════════════════

class TestSupplyChainRisk:

    def test_low_risk(self):
        from sc_neurocore.compiler.advanced_features import (
            score_supply_chain_risk,
        )
        r = score_supply_chain_risk("artix7")
        assert r.risk_score < 50
        assert r.export_control == "EAR99"

    def test_high_risk_biological(self):
        from sc_neurocore.compiler.advanced_features import (
            score_supply_chain_risk,
        )
        r = score_supply_chain_risk("finalspark_neuroplatform")
        assert r.risk_score >= 50
        assert "Emerging tech" in " ".join(r.risk_factors)

    def test_alternatives_exist(self):
        from sc_neurocore.compiler.advanced_features import (
            score_supply_chain_risk,
        )
        r = score_supply_chain_risk("artix7")
        assert len(r.alternatives) > 0


# ═══════════════════════════════════════════════════════════════════════
# 5. Bit-True Simulation Kernel
# ═══════════════════════════════════════════════════════════════════════

class TestBittrueKernel:

    def test_c_kernel(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_bittrue_kernel,
        )
        code = generate_bittrue_kernel("sc_lif", {"v": "a + b"})
        assert "#include <stdint.h>" in code
        assert "sc_lif_state_t" in code
        assert "sat(" in code
        assert "fxmul(" in code

    def test_rust_kernel(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_bittrue_kernel,
        )
        code = generate_bittrue_kernel(
            "sc_lif", {"v": "a + b"}, language="rust",
        )
        assert "pub struct" in code
        assert "fn sat" in code
        assert "clamp" in code

    def test_multi_var(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_bittrue_kernel,
        )
        code = generate_bittrue_kernel(
            "sc_izh", {"v": "a * b", "u": "c + d"},
        )
        assert "int16_t v;" in code
        assert "int16_t u;" in code


# ═══════════════════════════════════════════════════════════════════════
# 6. Model Complexity Classifier
# ═══════════════════════════════════════════════════════════════════════

class TestModelComplexity:

    def test_compute_bound(self):
        from sc_neurocore.compiler.advanced_features import (
            classify_model_complexity,
        )
        m = classify_model_complexity({"v": "a * b + c * d - e / f"})
        assert m.classification == "compute_bound"
        assert m.recommended_paradigm == "fpga"

    def test_simple_model(self):
        from sc_neurocore.compiler.advanced_features import (
            classify_model_complexity,
        )
        m = classify_model_complexity({"v": "a + b"})
        assert m.compute_ops == 1

    def test_cross_refs(self):
        from sc_neurocore.compiler.advanced_features import (
            classify_model_complexity,
        )
        m = classify_model_complexity({"v": "u + w", "u": "v", "w": "v + u"})
        assert m.comm_ratio > 0


# ═══════════════════════════════════════════════════════════════════════
# 7. Compilation Cache
# ═══════════════════════════════════════════════════════════════════════

class TestCompilationCache:

    def test_miss_then_hit(self):
        from sc_neurocore.compiler.advanced_features import CompilationCache
        cache = CompilationCache()
        eqs = {"v": "a + b"}
        assert cache.get(eqs, "artix7") is None
        assert cache.misses == 1
        cache.put(eqs, "artix7", 16, 8, {"verilog": "..."})
        result = cache.get(eqs, "artix7")
        assert result is not None
        assert cache.hits == 1

    def test_different_target_misses(self):
        from sc_neurocore.compiler.advanced_features import CompilationCache
        cache = CompilationCache()
        eqs = {"v": "a + b"}
        cache.put(eqs, "artix7", 16, 8, {"v": "data"})
        assert cache.get(eqs, "loihi2") is None

    def test_size(self):
        from sc_neurocore.compiler.advanced_features import CompilationCache
        cache = CompilationCache()
        cache.put({"v": "a"}, "artix7", 16, 8, {})
        cache.put({"v": "b"}, "artix7", 16, 8, {})
        assert cache.size == 2


# ═══════════════════════════════════════════════════════════════════════
# 8. Thermal Envelope
# ═══════════════════════════════════════════════════════════════════════

class TestThermalEnvelope:

    def test_pass(self):
        from sc_neurocore.compiler.advanced_features import (
            estimate_thermal_envelope,
        )
        t = estimate_thermal_envelope(power_mw=100, theta_ja=25)
        assert t.pass_fail == "PASS"
        assert t.t_junction == 27.5  # 25 + 0.1*25

    def test_fail(self):
        from sc_neurocore.compiler.advanced_features import (
            estimate_thermal_envelope,
        )
        t = estimate_thermal_envelope(
            power_mw=5000, theta_ja=30, t_junction_max=100,
        )
        assert t.pass_fail == "FAIL"
        assert t.thermal_margin < 0

    def test_margin(self):
        from sc_neurocore.compiler.advanced_features import (
            estimate_thermal_envelope,
        )
        t = estimate_thermal_envelope(power_mw=0)
        assert t.thermal_margin == 100.0  # 125 - 25


# ═══════════════════════════════════════════════════════════════════════
# 9. Network Topology Optimizer
# ═══════════════════════════════════════════════════════════════════════

class TestNetworkTopology:

    def test_basic_partition(self):
        from sc_neurocore.compiler.advanced_features import (
            optimize_network_topology,
        )
        adj = {0: [1, 2], 1: [0, 2], 2: [0, 1], 3: [4], 4: [3]}
        plan = optimize_network_topology(adj, num_chips=2)
        assert plan.num_chips == 2
        assert len(plan.chip_assignment) == 5

    def test_all_intra(self):
        from sc_neurocore.compiler.advanced_features import (
            optimize_network_topology,
        )
        adj = {0: [1], 1: [0]}
        plan = optimize_network_topology(adj, num_chips=1)
        assert plan.inter_chip_spikes == 0

    def test_bandwidth_reduction(self):
        from sc_neurocore.compiler.advanced_features import (
            optimize_network_topology,
        )
        adj = {0: [1], 1: [0], 2: [3], 3: [2]}
        plan = optimize_network_topology(adj, num_chips=2)
        assert plan.bandwidth_reduction >= 0.0


# ═══════════════════════════════════════════════════════════════════════
# 10. Cross-Feature E2E
# ═══════════════════════════════════════════════════════════════════════

class TestWave7Integration:

    def test_classify_then_recommend(self):
        from sc_neurocore.compiler.advanced_features import (
            classify_model_complexity, recommend_target,
        )
        m = classify_model_complexity({"v": "a * b + c * d - e"})
        recs = recommend_target(
            {"v": "a * b + c * d - e"},
            require_class=m.recommended_paradigm,
        )
        assert len(recs) > 0

    def test_recommend_then_risk(self):
        from sc_neurocore.compiler.advanced_features import (
            recommend_target, score_supply_chain_risk,
        )
        recs = recommend_target({"v": "a + b"}, top_n=1)
        risk = score_supply_chain_risk(recs[0].profile_name)
        assert risk.risk_score >= 0

    def test_bittrue_then_thermal(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_bittrue_kernel, estimate_thermal_envelope,
        )
        code = generate_bittrue_kernel("sc_lif", {"v": "a + b"})
        assert len(code) > 50
        t = estimate_thermal_envelope(power_mw=50)
        assert t.pass_fail == "PASS"

    def test_cache_workflow(self):
        from sc_neurocore.compiler.advanced_features import (
            CompilationCache, generate_bittrue_kernel,
        )
        cache = CompilationCache()
        eqs = {"v": "a + b"}
        assert cache.get(eqs, "artix7") is None
        code = generate_bittrue_kernel("sc_lif", eqs)
        cache.put(eqs, "artix7", 16, 8, {"code": code})
        hit = cache.get(eqs, "artix7")
        assert hit["code"] == code
