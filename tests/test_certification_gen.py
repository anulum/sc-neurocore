# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for certification evidence generation

"""Tests for safety-critical certification XML generation."""

from __future__ import annotations

from sc_neurocore.compiler.certification_gen import (
    CertificationItem,
    generate_certification_evidence,
)


class TestCertificationGen:
    """Test XML certification evidence generation."""

    def test_basic_evidence(self) -> None:
        """Should produce a valid XML document."""
        items = [
            CertificationItem("REQ-001", "The neuron shall spike.", "sc_lif", "test_spike"),
        ]
        xml = generate_certification_evidence("sc_lif", items)
        assert '<?xml version="1.0"' in xml
        assert "<certification_evidence>" in xml
        assert "RTCA DO-254" in xml
        assert "REQ-001" in xml

    def test_iec61508_standard(self) -> None:
        """Should support industrial standard."""
        xml = generate_certification_evidence("sc_lif", [], standard="iec61508")
        assert "IEC 61508" in xml

    def test_iso26262_standard(self) -> None:
        """Should support automotive standard."""
        xml = generate_certification_evidence("sc_lif", [], standard="iso26262")
        assert "ISO 26262" in xml

    def test_summary_counts(self) -> None:
        """Should correctly count PASS/FAIL items."""
        items = [
            CertificationItem("R1", "D1", "A1", "T1", status="PASS"),
            CertificationItem("R2", "D2", "A2", "T2", status="FAIL"),
            CertificationItem("R3", "D3", "A3", "T3", status="UNTESTED"),
        ]
        xml = generate_certification_evidence("sc_lif", items)
        assert 'total="3"' in xml
        assert 'passed="1"' in xml
        assert 'failed="1"' in xml
        assert 'coverage="33.3"' in xml

    def test_dal_level(self) -> None:
        """DAL level should propagate."""
        xml = generate_certification_evidence("sc_lif", [], dal_level="DAL-A")
        assert "<level>DAL-A</level>" in xml
