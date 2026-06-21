# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore

import unittest

from sc_neurocore.compiler.intelligence import protect_ip_pqc


class TestModelChecksum:
    """SHA-256 model checksum embedding."""

    def test_checksum_embedded(self):
        from sc_neurocore.compiler.intelligence import embed_model_checksum

        verilog = "// Test module\nmodule sc_lif (...);\nendmodule"
        result = embed_model_checksum(
            verilog,
            equations={"v": "a + b"},
            params={"data_width": 16},
        )
        assert "SHA-256:" in result
        assert "MODEL_HASH" in result
        assert "256'h" in result

    def test_checksum_deterministic(self):
        from sc_neurocore.compiler.intelligence import embed_model_checksum

        v1 = embed_model_checksum(
            "module x; endmodule",
            equations={"v": "a * b"},
        )
        v2 = embed_model_checksum(
            "module x; endmodule",
            equations={"v": "a * b"},
        )
        # Same inputs → same hash
        import re

        h1 = re.search(r"SHA-256: ([0-9a-f]+)", v1).group(1)
        h2 = re.search(r"SHA-256: ([0-9a-f]+)", v2).group(1)
        assert h1 == h2

    def test_different_equations_different_hash(self):
        from sc_neurocore.compiler.intelligence import embed_model_checksum
        import re

        v1 = embed_model_checksum("module x; endmodule", equations={"v": "a+b"})
        v2 = embed_model_checksum("module x; endmodule", equations={"v": "a*b"})
        h1 = re.search(r"SHA-256: ([0-9a-f]+)", v1).group(1)
        h2 = re.search(r"SHA-256: ([0-9a-f]+)", v2).group(1)
        assert h1 != h2

    def test_no_equations_still_works(self):
        from sc_neurocore.compiler.intelligence import embed_model_checksum

        result = embed_model_checksum("module y; endmodule")
        assert "MODEL_HASH" in result


class TestBitstreamEncryption:
    """AES-256 bitstream encryption for secure boot."""

    def test_xilinx_encryption(self):
        from sc_neurocore.compiler.intelligence import (
            generate_bitstream_encryption,
        )

        tcl = generate_bitstream_encryption("sc_lif", vendor="xilinx")
        assert "BITSTREAM.ENCRYPTION.ENCRYPT YES" in tcl
        assert "sc_lif.nky" in tcl
        assert "SECURITY_LEVEL" in tcl

    def test_intel_encryption(self):
        from sc_neurocore.compiler.intelligence import (
            generate_bitstream_encryption,
        )

        tcl = generate_bitstream_encryption("sc_hh", vendor="intel")
        assert "ENCRYPTION_KEY_SOURCE" in tcl
        assert "ENABLE_CONFIGURATION_BITSTREAM_ENCRYPTION ON" in tcl
        assert "ANTI_TAMPER" in tcl

    def test_key_source_efuse(self):
        from sc_neurocore.compiler.intelligence import (
            generate_bitstream_encryption,
        )

        tcl = generate_bitstream_encryption(
            "sc_lif",
            key_source="efuse",
        )
        assert "EFUSE" in tcl

    def test_key_source_bbram(self):
        from sc_neurocore.compiler.intelligence import (
            generate_bitstream_encryption,
        )

        tcl = generate_bitstream_encryption(
            "sc_lif",
            key_source="bbram",
        )
        assert "BBRAM" in tcl

    def test_module_name_in_output(self):
        from sc_neurocore.compiler.intelligence import (
            generate_bitstream_encryption,
        )

        tcl = generate_bitstream_encryption("my_neuron_design")
        assert "my_neuron_design" in tcl


class TestTrojanLint:
    def test_clean(self):
        from sc_neurocore.compiler.intelligence import lint_hardware_trojans

        r = lint_hardware_trojans({"v": "a + b", "u": "c - d"})
        assert r.risk_level == "LOW"

    def test_conditional(self):
        from sc_neurocore.compiler.intelligence import lint_hardware_trojans

        r = lint_hardware_trojans({"v": "a if trigger else b"})
        assert r.risk_level in ("MEDIUM", "HIGH")
        assert len(r.suspicious_paths) >= 1


class TestSBOM:
    def test_basic(self):
        from sc_neurocore.compiler.intelligence import generate_sbom

        s = generate_sbom("sc_lif", "artix7")
        assert s.total_components >= 3
        assert s.format == "CycloneDX"

    def test_with_deps(self):
        from sc_neurocore.compiler.intelligence import generate_sbom

        s = generate_sbom("sc_lif", "artix7", dependencies={"numpy": "1.26.0"})
        assert s.total_components >= 4


class TestIPObfuscation:
    def test_basic(self):
        from sc_neurocore.compiler.intelligence import obfuscate_ip

        r = obfuscate_ip("sc_lif", {"v": "a + b"})
        assert r.key_bits == 64
        assert "logic_locking" in r.techniques_applied
        assert r.obfuscated_signals > r.original_signals

    def test_custom_key(self):
        from sc_neurocore.compiler.intelligence import obfuscate_ip

        r = obfuscate_ip("sc_lif", {"v": "a + b"}, key_length=128)
        assert r.key_bits == 128


class TestWatermark:
    def test_basic(self):
        from sc_neurocore.compiler.intelligence import embed_watermark

        r = embed_watermark("sc_lif", {"v": "a"})
        assert r.verifiable is True
        assert len(r.watermark_hash) == 16
        assert r.overhead_percent <= 1.0

    def test_deterministic(self):
        from sc_neurocore.compiler.intelligence import embed_watermark

        r1 = embed_watermark("sc_lif", {"v": "a"}, owner_id="Lab1")
        r2 = embed_watermark("sc_lif", {"v": "a"}, owner_id="Lab1")
        assert r1.watermark_hash == r2.watermark_hash

    def test_different_owners(self):
        from sc_neurocore.compiler.intelligence import embed_watermark

        r1 = embed_watermark("sc_lif", {"v": "a"}, owner_id="Lab1")
        r2 = embed_watermark("sc_lif", {"v": "a"}, owner_id="Lab2")
        assert r1.watermark_hash != r2.watermark_hash


class TestWave10Integration:
    def test_from_constraints_to_report(self):
        from sc_neurocore.compiler.platforms import HardwareProfile
        from sc_neurocore.compiler.intelligence import (
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
        from sc_neurocore.compiler.intelligence import (
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


class TestPQC(unittest.TestCase):
    def test_basic(self):
        r = protect_ip_pqc("sc_lif", {"v": "a"})
        self.assertTrue(r.quantum_safe)
        self.assertEqual(r.algorithm, "CRYSTALS-Dilithium")
        self.assertEqual(len(r.signature_hex), 32)
        self.assertEqual(r.key_size_bits, 1952)

    def test_security_levels(self):
        r2 = protect_ip_pqc("m", {"v": "a"}, security_level=2)
        r5 = protect_ip_pqc("m", {"v": "a"}, security_level=5)
        self.assertLess(r2.key_size_bits, r5.key_size_bits)

    def test_deterministic(self):
        r1 = protect_ip_pqc("m", {"v": "a"})
        r2 = protect_ip_pqc("m", {"v": "a"})
        self.assertEqual(r1.signature_hex, r2.signature_hex)


class TestSideChannelLint:
    """Side-channel leakage analysis."""

    def test_clean_expression(self):
        from sc_neurocore.compiler.intelligence import (
            lint_side_channels,
        )

        findings = lint_side_channels({"v": "a + b"})
        # Should still have spike_out finding
        assert any(f.signal == "spike_out" for f in findings)

    def test_division_flagged(self):
        from sc_neurocore.compiler.intelligence import (
            lint_side_channels,
        )

        findings = lint_side_channels({"v": "a / b"})
        div_findings = [f for f in findings if "Division" in f.description]
        assert len(div_findings) == 1
        assert div_findings[0].risk_level == "medium"

    def test_branch_flagged(self):
        from sc_neurocore.compiler.intelligence import (
            lint_side_channels,
        )

        findings = lint_side_channels({"v": "a if x > 0 else b"})
        branch = [f for f in findings if f.risk_level == "high"]
        assert len(branch) >= 1

    def test_multiply_flagged(self):
        from sc_neurocore.compiler.intelligence import (
            lint_side_channels,
        )

        findings = lint_side_channels({"v": "a * b"})
        mul = [f for f in findings if "Hamming" in f.description]
        assert len(mul) == 1


class TestSupplyChainRisk:
    def test_low_risk(self):
        from sc_neurocore.compiler.intelligence import (
            score_supply_chain_risk,
        )

        r = score_supply_chain_risk("artix7")
        assert r.risk_score < 50
        assert r.export_control == "EAR99"

    def test_high_risk_biological(self):
        from sc_neurocore.compiler.intelligence import (
            score_supply_chain_risk,
        )

        r = score_supply_chain_risk("finalspark_neuroplatform")
        assert r.risk_score >= 50
        assert "Emerging tech" in " ".join(r.risk_factors)

    def test_alternatives_exist(self):
        from sc_neurocore.compiler.intelligence import (
            score_supply_chain_risk,
        )

        r = score_supply_chain_risk("artix7")
        assert len(r.alternatives) > 0

    def test_itar_for_radiation_hardened_fpga(self):
        """A radiation-hardened FPGA part is flagged ITAR-controlled."""
        from sc_neurocore.compiler.intelligence import score_supply_chain_risk

        r = score_supply_chain_risk("bae_rad750")
        assert r.export_control == "ITAR"
        assert "ITAR" in " ".join(r.risk_factors)

    def test_export_controlled_superconducting(self):
        """A superconducting platform is flagged export-controlled emerging tech."""
        from sc_neurocore.compiler.intelligence import score_supply_chain_risk

        r = score_supply_chain_risk("josephson_jj")
        assert r.export_control == "EAR-controlled"
        assert "superconducting" in " ".join(r.risk_factors).lower()


class TestCarbonFootprint:
    def test_basic(self):
        from sc_neurocore.compiler.intelligence import estimate_carbon_footprint

        c = estimate_carbon_footprint("artix7")
        assert c.manufacturing_kg_co2 > 0
        assert c.total_5yr_kg_co2 > c.manufacturing_kg_co2

    def test_biological_low(self):
        from sc_neurocore.compiler.intelligence import estimate_carbon_footprint

        c = estimate_carbon_footprint("finalspark_neuroplatform")
        assert c.manufacturing_kg_co2 <= 0.5


class TestLicenseChecker:
    def test_compatible(self):
        from sc_neurocore.compiler.intelligence import check_license_compliance

        r = check_license_compliance("AGPL-3.0", {"numpy": "BSD-3"})
        assert r.compatible is True

    def test_conflict(self):
        from sc_neurocore.compiler.intelligence import check_license_compliance

        r = check_license_compliance("MIT", {"gpl_lib": "GPL-3.0"})
        assert r.compatible is False
        assert len(r.conflicts) == 1
