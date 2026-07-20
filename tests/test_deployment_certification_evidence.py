# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Certification evidence deployment contracts

"""Contracts for generated deployment certification evidence."""

from __future__ import annotations


class TestCertificationEvidence:
    """Tests for safety-critical certification evidence generation."""

    def test_do254_xml(self) -> None:
        """DO-254 evidence generates valid XML structure."""
        from sc_neurocore.compiler.deployment import (
            CertificationItem,
            generate_certification_evidence,
        )

        items = [
            CertificationItem("REQ-001", "No overflow", "sc_lif.v", "sc_lif_sva.sv", "PASS"),
            CertificationItem("REQ-002", "Reset clears state", "sc_lif.v", "test_reset", "PASS"),
        ]
        xml = generate_certification_evidence("sc_lif", items)
        assert '<?xml version="1.0"' in xml
        assert "<certification_evidence>" in xml
        assert "<module>sc_lif</module>" in xml
        assert "RTCA DO-254" in xml
        assert "DAL-C" in xml
        assert 'passed="2"' in xml
        assert 'coverage="100.0"' in xml
        assert 'id="REQ-001"' in xml

    def test_iec61508_standard(self) -> None:
        """IEC 61508 standard label."""
        from sc_neurocore.compiler.deployment import (
            CertificationItem,
            generate_certification_evidence,
        )

        xml = generate_certification_evidence(
            "sc_lif",
            [CertificationItem("R1", "test", "d", "v", "PASS")],
            standard="iec61508",
            dal_level="SIL-3",
        )
        assert "IEC 61508" in xml
        assert "SIL-3" in xml

    def test_iso26262_standard(self) -> None:
        """ISO 26262 standard label."""
        from sc_neurocore.compiler.deployment import (
            CertificationItem,
            generate_certification_evidence,
        )

        xml = generate_certification_evidence(
            "sc_lif",
            [CertificationItem("R1", "test", "d", "v", "FAIL")],
            standard="iso26262",
            dal_level="ASIL-D",
        )
        assert "ISO 26262" in xml
        assert "ASIL-D" in xml
        assert 'failed="1"' in xml

    def test_mixed_status_coverage(self) -> None:
        """Coverage calculation with mixed statuses."""
        from sc_neurocore.compiler.deployment import (
            CertificationItem,
            generate_certification_evidence,
        )

        items = [
            CertificationItem("R1", "a", "d", "v", "PASS"),
            CertificationItem("R2", "b", "d", "v", "FAIL"),
            CertificationItem("R3", "c", "d", "v", "UNTESTED"),
        ]
        xml = generate_certification_evidence("sc_lif", items)
        assert 'total="3"' in xml
        assert 'passed="1"' in xml
        assert 'failed="1"' in xml
        assert 'coverage="33.3"' in xml

    def test_empty_items(self) -> None:
        """Empty items list produces valid XML with 0% coverage."""
        from sc_neurocore.compiler.deployment import generate_certification_evidence

        xml = generate_certification_evidence("sc_lif", [])
        assert 'total="0"' in xml
        assert 'coverage="0.0"' in xml
