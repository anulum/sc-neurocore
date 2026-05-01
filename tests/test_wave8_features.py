# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2020–2026 Miroslav Šotek. All rights reserved.
# SC-NeuroCore — Wave 8 test suite

"""Tests for Wave 8: frontier platforms + 10 compiler intelligence features."""

import pytest


# ═══════════════════════════════════════════════════════════════════════
# 1. Frontier Platform Classes
# ═══════════════════════════════════════════════════════════════════════

class TestFrontierPlatformsW8:

    @pytest.mark.parametrize("name", [
        "weebit_reram", "crossbar_rram", "adesto_cbram",
    ])
    def test_rram(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile
        assert get_profile(name).platform_class == "rram"

    @pytest.mark.parametrize("name", ["tsmc_cim_n7", "samsung_cim_sf3"])
    def test_sram_cim(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile
        assert get_profile(name).platform_class == "sram_cim"

    @pytest.mark.parametrize("name", ["intel_horse_ridge", "google_cryo_ctrl"])
    def test_cryo_cmos(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile
        assert get_profile(name).platform_class == "cryo_cmos"

    @pytest.mark.parametrize("name", [
        "microsoft_dna_store", "asu_dna_perovskite",
    ])
    def test_dna_molecular(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile
        assert get_profile(name).platform_class == "dna_molecular"

    @pytest.mark.parametrize("name", ["ibm_qnn", "ionq_trapped_ion"])
    def test_quantum_neuro(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile
        assert get_profile(name).platform_class == "quantum_neuro"

    def test_total_profiles(self):
        from sc_neurocore.compiler.hardware_profiles import list_profile_names
        assert len(list_profile_names()) >= 155

    def test_total_classes(self):
        from sc_neurocore.compiler.hardware_profiles import (
            list_profile_names, get_profile,
        )
        classes = {get_profile(n).platform_class for n in list_profile_names()}
        assert len(classes) >= 24


# ═══════════════════════════════════════════════════════════════════════
# 2. NIR Import
# ═══════════════════════════════════════════════════════════════════════

class TestNIRImport:

    def test_lif_import(self):
        from sc_neurocore.compiler.advanced_features import import_nir_graph
        g = import_nir_graph({
            "nodes": {"n0": {"type": "LIF", "tau": 20}},
            "edges": [],
        })
        assert "n0" in g.equations
        assert "20" in g.equations["n0"]

    def test_edges(self):
        from sc_neurocore.compiler.advanced_features import import_nir_graph
        g = import_nir_graph({
            "nodes": {"a": {"type": "LIF"}, "b": {"type": "LIF"}},
            "edges": [["a", "b"]],
        })
        assert ("a", "b") in g.edges

    def test_framework(self):
        from sc_neurocore.compiler.advanced_features import import_nir_graph
        g = import_nir_graph({"nodes": {}, "edges": []}, framework="Norse")
        assert g.framework == "Norse"


# ═══════════════════════════════════════════════════════════════════════
# 3. ODE Stability
# ═══════════════════════════════════════════════════════════════════════

class TestODEStability:

    def test_stable(self):
        from sc_neurocore.compiler.advanced_features import verify_ode_stability
        r = verify_ode_stability({"v": "a"}, dt=0.1)
        assert r.stable is True

    def test_unstable(self):
        from sc_neurocore.compiler.advanced_features import verify_ode_stability
        r = verify_ode_stability(
            {"v": "a"}, dt=100.0, time_constants={"v": 0.5},
        )
        assert r.stable is False

    def test_critical_dt(self):
        from sc_neurocore.compiler.advanced_features import verify_ode_stability
        r = verify_ode_stability(
            {"v": "a"}, dt=0.1, time_constants={"v": 10.0},
        )
        assert r.critical_dt == 20.0


# ═══════════════════════════════════════════════════════════════════════
# 4. Power Intent
# ═══════════════════════════════════════════════════════════════════════

class TestPowerIntent:

    def test_upf_output(self):
        from sc_neurocore.compiler.advanced_features import generate_power_intent
        upf = generate_power_intent("sc_lif")
        assert "set_scope sc_lif" in upf
        assert "PD_NEURON_0" in upf
        assert "set_isolation" in upf

    def test_num_domains(self):
        from sc_neurocore.compiler.advanced_features import generate_power_intent
        upf = generate_power_intent("sc_lif", num_domains=4)
        assert "PD_NEURON_3" in upf


# ═══════════════════════════════════════════════════════════════════════
# 5. Carbon Footprint
# ═══════════════════════════════════════════════════════════════════════

class TestCarbonFootprint:

    def test_basic(self):
        from sc_neurocore.compiler.advanced_features import estimate_carbon_footprint
        c = estimate_carbon_footprint("artix7")
        assert c.manufacturing_kg_co2 > 0
        assert c.total_5yr_kg_co2 > c.manufacturing_kg_co2

    def test_biological_low(self):
        from sc_neurocore.compiler.advanced_features import estimate_carbon_footprint
        c = estimate_carbon_footprint("finalspark_neuroplatform")
        assert c.manufacturing_kg_co2 <= 0.5


# ═══════════════════════════════════════════════════════════════════════
# 6. Debug Probes
# ═══════════════════════════════════════════════════════════════════════

class TestDebugProbes:

    def test_xilinx(self):
        from sc_neurocore.compiler.advanced_features import insert_debug_probes
        p = insert_debug_probes("sc_lif", {"v": "a"})
        assert p.probe_type == "ila"
        assert "v" in p.signals
        assert "create_debug_core" in p.tcl_commands

    def test_intel(self):
        from sc_neurocore.compiler.advanced_features import insert_debug_probes
        p = insert_debug_probes("sc_lif", {"v": "a"}, vendor="intel")
        assert p.probe_type == "signaltap"


# ═══════════════════════════════════════════════════════════════════════
# 7. Memory Map
# ═══════════════════════════════════════════════════════════════════════

class TestMemoryMap:

    def test_basic(self):
        from sc_neurocore.compiler.advanced_features import generate_memory_map
        m = generate_memory_map("sc_lif", {"v": "a", "u": "b"})
        assert m.total_bytes > 0
        assert "addr_dec" in m.decoder_verilog
        assert len(m.entries) > 0

    def test_base_address(self):
        from sc_neurocore.compiler.advanced_features import generate_memory_map
        m = generate_memory_map("sc_lif", {"v": "a"}, base_address=0x2000)
        assert m.base_address == 0x2000


# ═══════════════════════════════════════════════════════════════════════
# 8. Portability Score
# ═══════════════════════════════════════════════════════════════════════

class TestPortability:

    def test_simple_model(self):
        from sc_neurocore.compiler.advanced_features import score_portability
        s = score_portability({"v": "a + b"})
        assert s.score > 50
        assert s.compatible_profiles > 0

    def test_complex_model(self):
        from sc_neurocore.compiler.advanced_features import score_portability
        s = score_portability({"v": "a*b*c/d*e/f*g*h"})
        assert len(s.blockers) > 0


# ═══════════════════════════════════════════════════════════════════════
# 9. Reliability
# ═══════════════════════════════════════════════════════════════════════

class TestReliability:

    def test_nominal(self):
        from sc_neurocore.compiler.advanced_features import predict_reliability
        r = predict_reliability(voltage_v=0.9, temperature_c=25.0)
        assert r.mttf_years > 0
        assert r.failure_mode == "TDDB"

    def test_high_temp(self):
        from sc_neurocore.compiler.advanced_features import predict_reliability
        r = predict_reliability(temperature_c=125.0)
        assert r.failure_mode == "NBTI"
        assert r.temp_accel > 1.0


# ═══════════════════════════════════════════════════════════════════════
# 10. Fault Tree
# ═══════════════════════════════════════════════════════════════════════

class TestFaultTree:

    def test_basic(self):
        from sc_neurocore.compiler.advanced_features import generate_fault_tree
        ft = generate_fault_tree("sc_lif", {"v": "a", "u": "b"})
        assert "SYSTEM_FAILURE" in ft.top_event
        assert len(ft.basic_events) >= 6  # 2 vars * 2 + 2 common
        assert len(ft.mcs) == len(ft.basic_events)

    def test_single_var(self):
        from sc_neurocore.compiler.advanced_features import generate_fault_tree
        ft = generate_fault_tree("sc_lif", {"v": "a"})
        assert len(ft.basic_events) == 4  # 1 var * 2 + 2 common


# ═══════════════════════════════════════════════════════════════════════
# 11. Auto-Testbench
# ═══════════════════════════════════════════════════════════════════════

class TestAutoTestbench:

    def test_cocotb(self):
        from sc_neurocore.compiler.advanced_features import generate_testbench
        tb = generate_testbench("sc_lif", {"v": "a"})
        assert "import cocotb" in tb
        assert "test_sc_lif_reset" in tb

    def test_uvm(self):
        from sc_neurocore.compiler.advanced_features import generate_testbench
        tb = generate_testbench("sc_lif", {"v": "a"}, framework="uvm")
        assert "uvm_test" in tb


# ═══════════════════════════════════════════════════════════════════════
# 12. Cross-Feature E2E
# ═══════════════════════════════════════════════════════════════════════

class TestWave8Integration:

    def test_nir_to_stability(self):
        from sc_neurocore.compiler.advanced_features import (
            import_nir_graph, verify_ode_stability,
        )
        g = import_nir_graph({
            "nodes": {"n0": {"type": "LIF", "tau": 10}},
            "edges": [],
        })
        r = verify_ode_stability(g.equations, dt=0.1)
        assert r.stable is True

    def test_carbon_vs_reliability(self):
        from sc_neurocore.compiler.advanced_features import (
            estimate_carbon_footprint, predict_reliability,
        )
        c = estimate_carbon_footprint("artix7", power_mw=500)
        r = predict_reliability(voltage_v=0.9, temperature_c=85)
        assert c.total_5yr_kg_co2 > 0
        assert r.mttf_years > 0

    def test_fault_tree_then_testbench(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_fault_tree, generate_testbench,
        )
        ft = generate_fault_tree("sc_lif", {"v": "a"})
        tb = generate_testbench("sc_lif", {"v": "a"})
        assert len(ft.mcs) > 0
        assert len(tb) > 100
