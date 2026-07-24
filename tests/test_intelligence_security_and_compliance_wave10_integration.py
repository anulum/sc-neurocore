# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWave10Integration from former test_intelligence_security_and_compliance.py

"""Focused suite: TestWave10Integration from former test_intelligence_security_and_compliance.py."""

from __future__ import annotations

from tests.intelligence_security_and_compliance_support import *  # noqa: F403


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
