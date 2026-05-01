# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2020–2026 Miroslav Šotek. All rights reserved.
# SC-NeuroCore — Wave 6 test suite

"""Tests for Wave 6: new platform classes + 8 overlooked compiler features."""

import pytest


# ═══════════════════════════════════════════════════════════════════════
# 1. New Platform Classes (6 new classes, 18 profiles)
# ═══════════════════════════════════════════════════════════════════════

class TestNewPlatformClasses:
    """Verify all 6 new platform classes and 18 new profiles."""

    @pytest.mark.parametrize("name", [
        "nist_sfq", "northrop_aqfp", "josephson_jj",
    ])
    def test_superconducting_profiles(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile
        p = get_profile(name)
        assert p.platform_class == "superconducting"
        assert p.max_freq_mhz >= 5000  # GHz-class

    @pytest.mark.parametrize("name", [
        "everspin_stt_mram", "samsung_sot_mram",
    ])
    def test_spintronic_profiles(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile
        p = get_profile(name)
        assert p.platform_class == "spintronic"

    @pytest.mark.parametrize("name", ["gf_fefet", "sk_hynix_feram"])
    def test_ferroelectric_profiles(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile
        p = get_profile(name)
        assert p.platform_class == "ferroelectric"

    @pytest.mark.parametrize("name", [
        "samsung_cgra", "qualcomm_npu_cgra", "pact_xtensa",
    ])
    def test_cgra_profiles(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile
        p = get_profile(name)
        assert p.platform_class == "cgra"
        assert p.dsp_block  # CGRAs have PE blocks

    @pytest.mark.parametrize("name", [
        "tsmc_soic", "intel_foveros", "amd_3dv",
    ])
    def test_3d_stacked_profiles(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile
        p = get_profile(name)
        assert p.platform_class == "3d_stacked"

    @pytest.mark.parametrize("name", [
        "rp2040", "esp32_s3", "stm32h7", "nrf5340", "max78000",
    ])
    def test_edge_mcu_profiles(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile
        p = get_profile(name)
        assert p.platform_class == "edge_mcu"

    @pytest.mark.parametrize("name", [
        "sifive_x280", "qualcomm_ventana", "ainekko_rv",
    ])
    def test_riscv_ai_profiles(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile
        p = get_profile(name)
        assert p.platform_class == "accelerator"

    def test_total_profile_count(self):
        from sc_neurocore.compiler.hardware_profiles import list_profile_names
        assert len(list_profile_names()) >= 131

    def test_platform_class_count(self):
        from sc_neurocore.compiler.hardware_profiles import (
            list_profile_names, get_profile,
        )
        classes = {get_profile(n).platform_class
                   for n in list_profile_names()}
        assert len(classes) >= 15


# ═══════════════════════════════════════════════════════════════════════
# 2. Formal Equivalence Sketch
# ═══════════════════════════════════════════════════════════════════════

class TestFormalEquivalence:
    """Formal equivalence proof skeleton."""

    def test_basic_sketch(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_equivalence_sketch,
        )
        s = generate_equivalence_sketch(
            "sc_lif", {"v": "a + b * c"},
        )
        assert s.module_name == "sc_lif"
        assert len(s.proof_steps) >= 5
        assert len(s.assertions) == 1
        assert s.quantisation_bound > 0

    def test_multi_equation(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_equivalence_sketch,
        )
        s = generate_equivalence_sketch(
            "sc_izh", {"v": "a * b + c", "u": "d * e"},
        )
        assert len(s.assertions) == 2
        assert "CONCLUSION" in s.proof_steps[-1]

    def test_sva_format(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_equivalence_sketch,
        )
        s = generate_equivalence_sketch("sc_lif", {"v": "a + b"})
        assert "assert property" in s.assertions[0]
        assert "posedge clk" in s.assertions[0]


# ═══════════════════════════════════════════════════════════════════════
# 3. Multi-Timescale ODE Partitioner
# ═══════════════════════════════════════════════════════════════════════

class TestTimescalePartitioner:
    """Multi-timescale ODE partitioning."""

    def test_single_timescale(self):
        from sc_neurocore.compiler.advanced_features import (
            partition_timescales,
        )
        p = partition_timescales({"v": "a + b"})
        assert len(p.fast_equations) == 1
        assert len(p.slow_equations) == 0

    def test_explicit_separation(self):
        from sc_neurocore.compiler.advanced_features import (
            partition_timescales,
        )
        p = partition_timescales(
            {"v": "a + b", "w": "c + d"},
            time_constants={"v": 1.0, "w": 100.0},
        )
        assert "v" in p.fast_equations
        assert "w" in p.slow_equations
        assert p.slow_clock_div >= 2

    def test_cdc_signals(self):
        from sc_neurocore.compiler.advanced_features import (
            partition_timescales,
        )
        p = partition_timescales(
            {"v": "a + b", "w": "v * c"},
            time_constants={"v": 1.0, "w": 100.0},
        )
        assert "v" in p.cdc_signals

    def test_all_fast(self):
        from sc_neurocore.compiler.advanced_features import (
            partition_timescales,
        )
        p = partition_timescales(
            {"v": "a + b", "w": "c + d"},
            time_constants={"v": 1.0, "w": 2.0},
        )
        assert len(p.slow_equations) == 0


# ═══════════════════════════════════════════════════════════════════════
# 4. Provenance Chain
# ═══════════════════════════════════════════════════════════════════════

class TestProvenanceChain:
    """Cryptographic audit trail."""

    def test_chain_length(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_provenance_chain,
        )
        chain = generate_provenance_chain(
            "sc_lif", {"v": "a + b"},
        )
        assert len(chain) == 3
        assert chain[0].stage == "source_equations"
        assert chain[1].stage == "compilation_config"
        assert chain[2].stage == "verilog_generation"

    def test_hash_chain_linked(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_provenance_chain,
        )
        chain = generate_provenance_chain(
            "sc_lif", {"v": "a + b"},
        )
        assert chain[0].output_hash == chain[1].input_hash
        assert chain[1].output_hash == chain[2].input_hash

    def test_genesis(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_provenance_chain,
        )
        chain = generate_provenance_chain("sc_lif", {"v": "a"})
        assert chain[0].input_hash == "genesis"

    def test_json_format(self):
        import json
        from sc_neurocore.compiler.advanced_features import (
            generate_provenance_chain, format_provenance_json,
        )
        chain = generate_provenance_chain("sc_lif", {"v": "a"})
        j = format_provenance_json(chain)
        data = json.loads(j)
        assert "sc_neurocore_provenance" in data
        assert len(data["sc_neurocore_provenance"]["chain"]) == 3

    def test_deterministic_hashes(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_provenance_chain,
        )
        c1 = generate_provenance_chain("sc_lif", {"v": "a + b"})
        c2 = generate_provenance_chain("sc_lif", {"v": "a + b"})
        assert c1[0].output_hash == c2[0].output_hash


# ═══════════════════════════════════════════════════════════════════════
# 5. Compliance Matrix
# ═══════════════════════════════════════════════════════════════════════

class TestComplianceMatrix:
    """Safety compliance matrix generation."""

    def test_default_standards(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_compliance_matrix,
        )
        entries = generate_compliance_matrix("sc_lif")
        standards = {e.standard for e in entries}
        assert "DO-254" in standards
        assert "IEC 61508" in standards
        assert "ISO 26262" in standards

    def test_all_covered(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_compliance_matrix,
        )
        entries = generate_compliance_matrix(
            "sc_lif",
            has_tmr=True, has_checksum=True,
            has_sva=True, has_provenance=True,
        )
        covered = [e for e in entries if e.status == "covered"]
        assert len(covered) == len(entries)

    def test_gaps_without_tmr(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_compliance_matrix,
        )
        entries = generate_compliance_matrix("sc_lif")
        gaps = [e for e in entries if e.status == "gap"]
        assert len(gaps) > 0

    def test_format_report(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_compliance_matrix, format_compliance_report,
        )
        entries = generate_compliance_matrix("sc_lif", has_tmr=True)
        report = format_compliance_report(entries)
        assert "Compliance Matrix" in report
        assert "DO-254" in report
        assert "✅" in report


# ═══════════════════════════════════════════════════════════════════════
# 6. Energy Harvesting Scheduler
# ═══════════════════════════════════════════════════════════════════════

class TestEnergyScheduler:
    """Energy-aware neuron scheduling."""

    def test_basic_schedule(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_energy_schedule,
        )
        s = generate_energy_schedule(1000)
        assert s.total_neurons == 1000
        assert s.neurons_per_epoch <= 1000
        assert s.duty_cycle > 0

    def test_energy_limited(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_energy_schedule,
        )
        s = generate_energy_schedule(
            1000, energy_budget_uj=1.0, energy_per_neuron_nj=100.0,
        )
        assert s.neurons_per_epoch == 10
        assert s.duty_cycle == 0.01

    def test_priority_neurons(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_energy_schedule,
        )
        s = generate_energy_schedule(
            100, priority_neurons=[50, 51, 52],
        )
        assert s.update_order[0] == 50
        assert s.update_order[1] == 51

    def test_excess_budget(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_energy_schedule,
        )
        s = generate_energy_schedule(
            10, energy_budget_uj=1000.0,
        )
        assert s.neurons_per_epoch == 10
        assert s.duty_cycle == 1.0


# ═══════════════════════════════════════════════════════════════════════
# 7. Side-Channel Lint
# ═══════════════════════════════════════════════════════════════════════

class TestSideChannelLint:
    """Side-channel leakage analysis."""

    def test_clean_expression(self):
        from sc_neurocore.compiler.advanced_features import (
            lint_side_channels,
        )
        findings = lint_side_channels({"v": "a + b"})
        # Should still have spike_out finding
        assert any(f.signal == "spike_out" for f in findings)

    def test_division_flagged(self):
        from sc_neurocore.compiler.advanced_features import (
            lint_side_channels,
        )
        findings = lint_side_channels({"v": "a / b"})
        div_findings = [f for f in findings if "Division" in f.description]
        assert len(div_findings) == 1
        assert div_findings[0].risk_level == "medium"

    def test_branch_flagged(self):
        from sc_neurocore.compiler.advanced_features import (
            lint_side_channels,
        )
        findings = lint_side_channels({"v": "a if x > 0 else b"})
        branch = [f for f in findings if f.risk_level == "high"]
        assert len(branch) >= 1

    def test_multiply_flagged(self):
        from sc_neurocore.compiler.advanced_features import (
            lint_side_channels,
        )
        findings = lint_side_channels({"v": "a * b"})
        mul = [f for f in findings if "Hamming" in f.description]
        assert len(mul) == 1


# ═══════════════════════════════════════════════════════════════════════
# 8. Drift Compensation
# ═══════════════════════════════════════════════════════════════════════

class TestDriftCompensation:
    """Analog drift compensation controller."""

    def test_basic_compensator(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_drift_compensator,
        )
        d = generate_drift_compensator("sc_analog")
        assert "module sc_analog_drift_ctrl" in d.verilog_controller
        assert "endmodule" in d.verilog_controller
        assert d.refresh_interval_ms > 0
        assert d.compensation_method == "periodic_refresh"

    def test_fast_drift(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_drift_compensator,
        )
        d = generate_drift_compensator(
            "sc_rram", drift_rate_per_day=0.1,
            max_drift_tolerance=0.01,
        )
        # Should refresh very frequently
        assert d.refresh_interval_ms < 10_000_000

    def test_verilog_contains_counter(self):
        from sc_neurocore.compiler.advanced_features import (
            generate_drift_compensator,
        )
        d = generate_drift_compensator("sc_mem")
        assert "counter" in d.verilog_controller
        assert "refresh_trigger" in d.verilog_controller
        assert "REFRESH_CYCLES" in d.verilog_controller


# ═══════════════════════════════════════════════════════════════════════
# 9. Heterogeneous Dispatch
# ═══════════════════════════════════════════════════════════════════════

class TestHeterogeneousDispatch:
    """Multi-backend SNN dispatch."""

    def test_two_backends(self):
        from sc_neurocore.compiler.advanced_features import (
            plan_heterogeneous_dispatch,
        )
        plan = plan_heterogeneous_dispatch(
            {"v": "a + b", "u": "c * d"},
            ["fpga", "gpu"],
        )
        assert "fpga" in plan.backends
        assert "gpu" in plan.backends
        assert plan.estimated_speedup > 1.0

    def test_single_backend(self):
        from sc_neurocore.compiler.advanced_features import (
            plan_heterogeneous_dispatch,
        )
        plan = plan_heterogeneous_dispatch(
            {"v": "a + b"}, ["fpga"],
        )
        assert len(plan.sync_barriers) == 0
        assert plan.total_neurons_per_backend["fpga"] == 1000

    def test_three_backends(self):
        from sc_neurocore.compiler.advanced_features import (
            plan_heterogeneous_dispatch,
        )
        plan = plan_heterogeneous_dispatch(
            {"v": "a", "u": "b", "w": "c"},
            ["fpga", "mcu", "gpu"],
            neuron_count=3000,
        )
        assert len(plan.sync_barriers) == 2
        total = sum(plan.total_neurons_per_backend.values())
        assert total == 3000

    def test_neuron_distribution(self):
        from sc_neurocore.compiler.advanced_features import (
            plan_heterogeneous_dispatch,
        )
        plan = plan_heterogeneous_dispatch(
            {"v": "a", "u": "b"},
            ["fpga", "gpu"],
            neuron_count=100,
        )
        total = sum(plan.total_neurons_per_backend.values())
        assert total == 100


# ═══════════════════════════════════════════════════════════════════════
# 10. Cross-Feature E2E Integration
# ═══════════════════════════════════════════════════════════════════════

class TestWave6Integration:
    """Cross-feature integration tests."""

    def test_provenance_then_compliance(self):
        """Provenance chain enables compliance coverage."""
        from sc_neurocore.compiler.advanced_features import (
            generate_provenance_chain, generate_compliance_matrix,
        )
        chain = generate_provenance_chain("sc_lif", {"v": "a + b"})
        assert len(chain) == 3
        matrix = generate_compliance_matrix(
            "sc_lif", has_provenance=True, has_tmr=True,
            has_checksum=True, has_sva=True,
        )
        all_covered = all(e.status == "covered" for e in matrix)
        assert all_covered

    def test_timescale_then_dispatch(self):
        """Partition timescales, then dispatch to backends."""
        from sc_neurocore.compiler.advanced_features import (
            partition_timescales, plan_heterogeneous_dispatch,
        )
        part = partition_timescales(
            {"v": "a + b", "w": "c + d"},
            time_constants={"v": 1.0, "w": 100.0},
        )
        all_eqs = {**part.fast_equations, **part.slow_equations}
        plan = plan_heterogeneous_dispatch(
            all_eqs, ["fpga", "mcu"],
        )
        assert plan.estimated_speedup > 1.0

    def test_equivalence_then_lint(self):
        """Generate proof sketch, then lint for side channels."""
        from sc_neurocore.compiler.advanced_features import (
            generate_equivalence_sketch, lint_side_channels,
        )
        sketch = generate_equivalence_sketch(
            "sc_hh", {"v": "a * b / c + d"},
        )
        findings = lint_side_channels({"v": "a * b / c + d"})
        assert sketch.quantisation_bound > 0
        assert len(findings) >= 3  # div + mul + spike

    def test_energy_schedule_for_mcu(self):
        """Energy schedule on edge MCU profile."""
        from sc_neurocore.compiler.hardware_profiles import get_profile
        from sc_neurocore.compiler.advanced_features import (
            generate_energy_schedule,
        )
        p = get_profile("esp32_s3")
        assert p.platform_class == "edge_mcu"
        s = generate_energy_schedule(500, energy_budget_uj=5.0)
        assert s.neurons_per_epoch <= 500
