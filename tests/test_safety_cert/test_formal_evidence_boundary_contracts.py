# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBoundaryContracts from former test_formal_evidence.py

"""Focused suite: TestBoundaryContracts from former test_formal_evidence.py."""

from __future__ import annotations

from tests.test_safety_cert.formal_evidence_support import *  # noqa: F403


class TestBoundaryContracts:
    def test_formal_certificate_add_property_and_report_validation(self) -> None:
        cert = FormalProofCertificate()
        prop = FormalProperty("P1", "neuron", "desc", "assert", "proven")
        cert.add_property(prop)
        assert cert.total_count == 1
        prop.property_type = _unsafe("invalid")
        with pytest.raises(ValueError, match="property_type"):
            cert.generate_report()

    def test_proof_test_assessment_boundaries(self) -> None:
        assert (
            ProofTestCoverage.coverage_from_proofs(
                [FormalProperty("P1", "m", "desc", "cover", "proven")]
            )
            == 0.0
        )
        assert ProofTestCoverage.dc_to_sil(0.9) == SILLevel.SIL_2
        assert ProofTestCoverage.dc_to_sil(0.6) == SILLevel.SIL_1

    def test_property_gap_rejects_boolean_proven_count(self) -> None:
        with pytest.raises(ValueError, match="proven_properties"):
            PropertyGap("module", 2, True, ["assert"])
