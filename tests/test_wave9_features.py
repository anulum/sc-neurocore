# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2020–2026 Miroslav Šotek. All rights reserved.
# SC-NeuroCore — Wave 9 test suite

"""Tests for Wave 9: final frontier platforms + extensibility layer."""

import os
import tempfile
import pytest


# ═══════════════════════════════════════════════════════════════════════
# 1. Final Platform Classes
# ═══════════════════════════════════════════════════════════════════════

class TestFinalPlatforms:

    @pytest.mark.parametrize("name", ["ayar_teraphy", "intel_cpo"])
    def test_optical_io(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile
        assert get_profile(name).platform_class == "optical_io"

    @pytest.mark.parametrize("name", ["mit_phononic", "caltech_mems_nn"])
    def test_acoustic(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile
        assert get_profile(name).platform_class == "acoustic"

    @pytest.mark.parametrize("name", [
        "stanford_microfluidic", "eth_fluidic_logic",
    ])
    def test_fluidic(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile
        assert get_profile(name).platform_class == "fluidic"

    @pytest.mark.parametrize("name", [
        "bae_rad750_sq", "seakr_sbc", "vorago_va10820", "frontgrade_leon5",
    ])
    def test_space_qualified(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile
        assert get_profile(name).platform_class == "space_qualified"

    def test_total_profiles(self):
        from sc_neurocore.compiler.hardware_profiles import list_profile_names
        assert len(list_profile_names()) >= 164

    def test_total_classes(self):
        from sc_neurocore.compiler.hardware_profiles import (
            list_profile_names, get_profile,
        )
        classes = {get_profile(n).platform_class for n in list_profile_names()}
        assert len(classes) >= 28


# ═══════════════════════════════════════════════════════════════════════
# 2. CDC Analyzer
# ═══════════════════════════════════════════════════════════════════════

class TestCDC:

    def test_same_domain(self):
        from sc_neurocore.compiler.advanced_features import analyze_cdc
        r = analyze_cdc({"v": "a + b", "u": "c"})
        assert r.safe is True

    def test_cross_domain(self):
        from sc_neurocore.compiler.advanced_features import analyze_cdc
        r = analyze_cdc(
            {"v": "u + 1", "u": "v - 1"},
            clock_domains={"v": "clk_a", "u": "clk_b"},
        )
        assert r.total_crossings >= 2


# ═══════════════════════════════════════════════════════════════════════
# 3. TOML Profile Loader
# ═══════════════════════════════════════════════════════════════════════

class TestTOMLLoader:

    def test_load(self, tmp_path):
        from sc_neurocore.compiler.advanced_features import load_profiles_from_toml
        from sc_neurocore.compiler.hardware_profiles import get_profile
        toml = tmp_path / "custom.toml"
        toml.write_text(
            '[[profile]]\n'
            'name = "test_custom_chip"\n'
            'vendor = "TestVendor"\n'
            'family = "TestFam"\n'
            'platform_class = "custom"\n'
            'data_width = 16\n'
            'fraction = 8\n'
        )
        loaded = load_profiles_from_toml(str(toml))
        assert "test_custom_chip" in loaded
        p = get_profile("test_custom_chip")
        assert p.vendor == "TestVendor"


# ═══════════════════════════════════════════════════════════════════════
# 4. Multi-Die Floorplanner
# ═══════════════════════════════════════════════════════════════════════

class TestFloorplanner:

    def test_basic(self):
        from sc_neurocore.compiler.advanced_features import plan_multi_die_floorplan
        r = plan_multi_die_floorplan(
            {"cortex_a": 500, "cortex_b": 300, "cortex_c": 400},
            die_capacity=1000,
        )
        assert len(r.die_assignment) == 3
        assert r.total_dies >= 1

    def test_overflow(self):
        from sc_neurocore.compiler.advanced_features import plan_multi_die_floorplan
        r = plan_multi_die_floorplan(
            {"big": 900, "huge": 800},
            die_capacity=1000, num_dies=2,
        )
        assert r.total_dies == 2


# ═══════════════════════════════════════════════════════════════════════
# 5. Regression Watchdog
# ═══════════════════════════════════════════════════════════════════════

class TestRegressionWatchdog:

    def test_no_regression(self):
        from sc_neurocore.compiler.advanced_features import check_regression
        r = check_regression({"area": 100}, {"area": 102})
        assert r[0].regression is False

    def test_regression(self):
        from sc_neurocore.compiler.advanced_features import check_regression
        r = check_regression({"area": 100}, {"area": 120})
        assert r[0].regression is True
        assert r[0].delta_pct == 20.0


# ═══════════════════════════════════════════════════════════════════════
# 6. License Checker
# ═══════════════════════════════════════════════════════════════════════

class TestLicenseChecker:

    def test_compatible(self):
        from sc_neurocore.compiler.advanced_features import check_license_compliance
        r = check_license_compliance("AGPL-3.0", {"numpy": "BSD-3"})
        assert r.compatible is True

    def test_conflict(self):
        from sc_neurocore.compiler.advanced_features import check_license_compliance
        r = check_license_compliance("MIT", {"gpl_lib": "GPL-3.0"})
        assert r.compatible is False
        assert len(r.conflicts) == 1


# ═══════════════════════════════════════════════════════════════════════
# 7. Power State Machine
# ═══════════════════════════════════════════════════════════════════════

class TestPowerFSM:

    def test_default(self):
        from sc_neurocore.compiler.advanced_features import generate_power_state_machine
        v = generate_power_state_machine("sc_lif")
        assert "ACTIVE" in v
        assert "HIBERNATE" in v
        assert "power_fsm" in v

    def test_custom_states(self):
        from sc_neurocore.compiler.advanced_features import generate_power_state_machine
        v = generate_power_state_machine("sc_lif", states=["ON", "OFF"])
        assert "ON" in v
        assert "OFF" in v


# ═══════════════════════════════════════════════════════════════════════
# 8. Platform Discovery Hook
# ═══════════════════════════════════════════════════════════════════════

class TestDiscoveryHook:

    def test_register_and_discover(self):
        from sc_neurocore.compiler.advanced_features import (
            register_platform_hook, discover_platforms, _DISCOVERY_HOOKS,
        )
        from sc_neurocore.compiler.hardware_profiles import HardwareProfile

        def my_hook():
            return [HardwareProfile(
                name="test_discovered_chip",
                vendor="HookVendor",
                family="HookFam",
                platform_class="custom",
                data_width=16, fraction=8,
                overflow="saturate", rounding="nearest",
            )]

        register_platform_hook(my_hook)
        found = discover_platforms()
        assert "test_discovered_chip" in found
        # Cleanup
        _DISCOVERY_HOOKS.pop()


# ═══════════════════════════════════════════════════════════════════════
# 9. Compilation Report
# ═══════════════════════════════════════════════════════════════════════

class TestCompilationReport:

    def test_basic(self):
        from sc_neurocore.compiler.advanced_features import generate_compilation_report
        md = generate_compilation_report("sc_lif", {"v": "a"}, "artix7")
        assert "# SC-NeuroCore Compilation Report" in md
        assert "artix7" in md
        assert "Carbon" in md

    def test_no_carbon(self):
        from sc_neurocore.compiler.advanced_features import generate_compilation_report
        md = generate_compilation_report(
            "sc_lif", {"v": "a"}, "artix7", include_carbon=False,
        )
        assert "Carbon" not in md


# ═══════════════════════════════════════════════════════════════════════
# 10. E2E Integration
# ═══════════════════════════════════════════════════════════════════════

class TestWave9Integration:

    def test_toml_to_report(self, tmp_path):
        from sc_neurocore.compiler.advanced_features import (
            load_profiles_from_toml, generate_compilation_report,
        )
        toml = tmp_path / "e2e.toml"
        toml.write_text(
            '[[profile]]\n'
            'name = "e2e_custom"\n'
            'vendor = "E2EVendor"\n'
            'platform_class = "custom"\n'
            'data_width = 16\n'
            'fraction = 8\n'
        )
        load_profiles_from_toml(str(toml))
        md = generate_compilation_report("sc_lif", {"v": "a"}, "e2e_custom")
        assert "E2EVendor" in md

    def test_cdc_then_floorplan(self):
        from sc_neurocore.compiler.advanced_features import (
            analyze_cdc, plan_multi_die_floorplan,
        )
        r = analyze_cdc({"v": "u", "u": "v"},
                        clock_domains={"v": "clk_a", "u": "clk_b"})
        assert r.total_crossings >= 2
        fp = plan_multi_die_floorplan({"region_a": 500, "region_b": 500})
        assert fp.total_dies >= 1
