# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBoundaryContracts from former test_traceability.py

"""Focused suite: TestBoundaryContracts from former test_traceability.py."""

from __future__ import annotations

from tests.test_safety_cert.traceability_support import *  # noqa: F403

class TestBoundaryContracts:
    def test_traceability_unknown_verification_link_is_rejected_without_mutation(self) -> None:
        tm = TraceabilityMatrix()
        assert tm.link_verification("REQ_MISSING", "formal/proof.sby") is False
        assert tm.requirements == {}

    def test_traceability_rejects_corrupted_verification_reference_state(self) -> None:
        tm = TraceabilityMatrix()
        req = Requirement("REQ_001", "Test", SafetyStandard.IEC_61508)
        tm.add_requirement(req)
        req.verification_refs.append(_unsafe(""))
        with pytest.raises(ValueError, match="verification_refs"):
            tm.link_implementation("REQ_001", "hdl/test.v")

    def test_traceability_empty_matrix_reports_zero_fraction(self) -> None:
        assert TraceabilityMatrix().coverage == 0.0

    def test_implemented_count_and_explicit_timestamp_validation(self) -> None:
        matrix = TraceabilityMatrix()
        matrix.add_requirement(
            Requirement(
                "REQ_001",
                "description",
                SafetyStandard.IEC_61508,
                implementation_refs=["rtl/neuron.sv"],
                status="implemented",
            )
        )
        assert matrix.implemented_count == 1
        assert matrix.verified_count == 0
        with pytest.raises(ValueError, match="generated_at"):
            matrix.generate_report(generated_at="")
