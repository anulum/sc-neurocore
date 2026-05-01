# SPDX-License-Identifier: AGPL-3.0-or-later
# © 2020–2026 Miroslav Šotek. All rights reserved.
# SC-NeuroCore — Wave 10 test suite

"""Tests for Wave 10: absolute endgame platforms + moat-sealing features."""

import pytest


# ═══════════════════════════════════════════════════════════════════════
# 1. Final Platform Classes
# ═══════════════════════════════════════════════════════════════════════


class TestMagnonicPlatforms:
    @pytest.mark.parametrize(
        "name",
        [
            "tum_skyrmion",
            "kaist_spinwave",
            "imec_mtj_reservoir",
        ],
    )
    def test_magnonic(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile

        assert get_profile(name).platform_class == "magnonic"


class TestOrganicBioelectronic:
    @pytest.mark.parametrize("name", ["cambridge_oect", "linkoping_organic"])
    def test_organic(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile

        assert get_profile(name).platform_class == "organic_bioelectronic"


class TestRiscVSovereign:
    @pytest.mark.parametrize(
        "name",
        [
            "sifive_x280_ai",
            "esperanto_et_soc",
            "ventana_veyron_ai",
            "tenstorrent_ascalon",
            "andes_ax45mpv",
        ],
    )
    def test_risc_v(self, name):
        from sc_neurocore.compiler.hardware_profiles import get_profile

        assert get_profile(name).platform_class == "risc_v_sovereign"


class TestTotalCoverage:
    def test_min_profiles(self):
        from sc_neurocore.compiler.hardware_profiles import list_profile_names

        assert len(list_profile_names()) >= 175

    def test_min_classes(self):
        from sc_neurocore.compiler.hardware_profiles import (
            list_profile_names,
            get_profile,
        )

        classes = {get_profile(n).platform_class for n in list_profile_names()}
        assert len(classes) >= 31


# ═══════════════════════════════════════════════════════════════════════
# 2. Generic Profile Constructor
# ═══════════════════════════════════════════════════════════════════════


class TestFromConstraints:
    def test_basic(self):
        from sc_neurocore.compiler.hardware_profiles import (
            HardwareProfile,
            get_profile,
        )

        p = HardwareProfile.from_constraints(
            "test_w10_auto",
            vendor="TestVendor",
            platform_class="custom",
        )
        assert p.data_width >= 8
        assert p.fraction >= 1
        retrieved = get_profile("test_w10_auto")
        assert retrieved.vendor == "TestVendor"

    def test_low_power(self):
        from sc_neurocore.compiler.hardware_profiles import HardwareProfile

        p = HardwareProfile.from_constraints(
            "test_w10_lowpow",
            max_power_budget_mw=5,
        )
        assert p.data_width == 8

    def test_explicit_width(self):
        from sc_neurocore.compiler.hardware_profiles import HardwareProfile

        p = HardwareProfile.from_constraints(
            "test_w10_32bit",
            data_width=32,
            fraction=16,
        )
        assert p.data_width == 32
        assert p.fraction == 16


# ═══════════════════════════════════════════════════════════════════════
# 3. Hardware Trojan Lint
# ═══════════════════════════════════════════════════════════════════════


class TestTrojanLint:
    def test_clean(self):
        from sc_neurocore.compiler.advanced_features import lint_hardware_trojans

        r = lint_hardware_trojans({"v": "a + b", "u": "c - d"})
        assert r.risk_level == "LOW"

    def test_conditional(self):
        from sc_neurocore.compiler.advanced_features import lint_hardware_trojans

        r = lint_hardware_trojans({"v": "a if trigger else b"})
        assert r.risk_level in ("MEDIUM", "HIGH")
        assert len(r.suspicious_paths) >= 1


# ═══════════════════════════════════════════════════════════════════════
# 4. SBOM Generator
# ═══════════════════════════════════════════════════════════════════════


class TestSBOM:
    def test_basic(self):
        from sc_neurocore.compiler.advanced_features import generate_sbom

        s = generate_sbom("sc_lif", "artix7")
        assert s.total_components >= 3
        assert s.format == "CycloneDX"

    def test_with_deps(self):
        from sc_neurocore.compiler.advanced_features import generate_sbom

        s = generate_sbom("sc_lif", "artix7", dependencies={"numpy": "1.26.0"})
        assert s.total_components >= 4


# ═══════════════════════════════════════════════════════════════════════
# 5. HIL Calibration
# ═══════════════════════════════════════════════════════════════════════


class TestHILCalibration:
    def test_basic(self):
        from sc_neurocore.compiler.advanced_features import generate_hil_calibration

        r = generate_hil_calibration("sc_lif", {"v": "expr", "u": "expr"})
        assert r.num_parameters == 2
        assert len(r.protocol_steps) >= 5

    def test_custom_ranges(self):
        from sc_neurocore.compiler.advanced_features import generate_hil_calibration

        r = generate_hil_calibration(
            "sc_lif",
            {"v": "expr"},
            parameters={"tau": (-1.0, 1.0)},
        )
        assert r.sweep_ranges["tau"] == (-1.0, 1.0)


# ═══════════════════════════════════════════════════════════════════════
# 6. Digital Twin
# ═══════════════════════════════════════════════════════════════════════


class TestDigitalTwin:
    def test_basic(self):
        from sc_neurocore.compiler.advanced_features import generate_digital_twin

        code = generate_digital_twin("sc_lif", {"v": "-(v)/tau"}, "artix7")
        assert "Twin" in code
        assert "def step" in code
        assert "def compare" in code


# ═══════════════════════════════════════════════════════════════════════
# 7. UCIe Protocol Mapper
# ═══════════════════════════════════════════════════════════════════════


class TestUCIeMapper:
    def test_basic(self):
        from sc_neurocore.compiler.advanced_features import map_ucie_protocol

        r = map_ucie_protocol({"core_a": 64, "core_b": 128})
        assert r.lanes["core_a"] >= 1
        assert r.lanes["core_b"] >= 1
        assert r.total_bandwidth_gbps > 0
        assert "UCIe" in r.protocol_version


# ═══════════════════════════════════════════════════════════════════════
# 8. SEU Scrub Scheduler
# ═══════════════════════════════════════════════════════════════════════


class TestSEUScrubber:
    def test_leo(self):
        from sc_neurocore.compiler.advanced_features import schedule_seu_scrubbing

        s = schedule_seu_scrubbing(1_000_000, orbit_altitude_km=400)
        assert s.interval_ms > 0
        assert s.frames_per_cycle > 0
        assert s.strategy == "hybrid"

    def test_higher_orbit(self):
        from sc_neurocore.compiler.advanced_features import schedule_seu_scrubbing

        leo = schedule_seu_scrubbing(1_000_000, orbit_altitude_km=400)
        geo = schedule_seu_scrubbing(1_000_000, orbit_altitude_km=35786)
        # Higher orbit = more flux = shorter interval
        assert geo.interval_ms < leo.interval_ms


# ═══════════════════════════════════════════════════════════════════════
# 9. IP Obfuscation
# ═══════════════════════════════════════════════════════════════════════


class TestIPObfuscation:
    def test_basic(self):
        from sc_neurocore.compiler.advanced_features import obfuscate_ip

        r = obfuscate_ip("sc_lif", {"v": "a + b"})
        assert r.key_bits == 64
        assert "logic_locking" in r.techniques_applied
        assert r.obfuscated_signals > r.original_signals

    def test_custom_key(self):
        from sc_neurocore.compiler.advanced_features import obfuscate_ip

        r = obfuscate_ip("sc_lif", {"v": "a + b"}, key_length=128)
        assert r.key_bits == 128


# ═══════════════════════════════════════════════════════════════════════
# 10. Model Watermark
# ═══════════════════════════════════════════════════════════════════════


class TestWatermark:
    def test_basic(self):
        from sc_neurocore.compiler.advanced_features import embed_watermark

        r = embed_watermark("sc_lif", {"v": "a"})
        assert r.verifiable is True
        assert len(r.watermark_hash) == 16
        assert r.overhead_percent <= 1.0

    def test_deterministic(self):
        from sc_neurocore.compiler.advanced_features import embed_watermark

        r1 = embed_watermark("sc_lif", {"v": "a"}, owner_id="Lab1")
        r2 = embed_watermark("sc_lif", {"v": "a"}, owner_id="Lab1")
        assert r1.watermark_hash == r2.watermark_hash

    def test_different_owners(self):
        from sc_neurocore.compiler.advanced_features import embed_watermark

        r1 = embed_watermark("sc_lif", {"v": "a"}, owner_id="Lab1")
        r2 = embed_watermark("sc_lif", {"v": "a"}, owner_id="Lab2")
        assert r1.watermark_hash != r2.watermark_hash


# ═══════════════════════════════════════════════════════════════════════
# 11. E2E Integration
# ═══════════════════════════════════════════════════════════════════════


class TestWave10Integration:
    def test_from_constraints_to_report(self):
        from sc_neurocore.compiler.hardware_profiles import HardwareProfile
        from sc_neurocore.compiler.advanced_features import (
            generate_compilation_report,
            generate_sbom,
            embed_watermark,
        )

        p = HardwareProfile.from_constraints(
            "test_w10_e2e",
            vendor="E2E",
            platform_class="custom",
        )
        report = generate_compilation_report(
            "sc_lif",
            {"v": "a"},
            "test_w10_e2e",
        )
        assert "E2E" in report
        sbom = generate_sbom("sc_lif", "test_w10_e2e")
        assert sbom.total_components >= 3
        wm = embed_watermark("sc_lif", {"v": "a"})
        assert wm.verifiable

    def test_space_pipeline(self):
        from sc_neurocore.compiler.advanced_features import (
            lint_hardware_trojans,
            schedule_seu_scrubbing,
            obfuscate_ip,
            generate_sbom,
        )

        trojan = lint_hardware_trojans({"v": "a + b"})
        assert trojan.risk_level == "LOW"
        scrub = schedule_seu_scrubbing(500_000, orbit_altitude_km=800)
        assert scrub.interval_ms > 0
        obf = obfuscate_ip("sc_lif", {"v": "a + b"})
        assert obf.key_bits > 0
        sbom = generate_sbom("sc_lif", "bae_rad750_sq")
        assert sbom.total_components >= 3
