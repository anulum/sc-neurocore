# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFormalDigestBoundary from former test_formal_evidence.py

"""Focused suite: TestFormalDigestBoundary from former test_formal_evidence.py."""

from __future__ import annotations

from tests.test_safety_cert.formal_evidence_support import *  # noqa: F403


class TestFormalDigestBoundary:
    @pytest.mark.parametrize(
        ("field", "value", "match"),
        [
            ("prop_id", "", "prop_id"),
            ("description", "", "descriptions"),
            ("property_type", "invalid", "property_type"),
            ("status", "invalid", "statuses"),
            ("engine", "", "engines"),
            ("depth", True, "depths"),
            ("sby_file", None, "sby_file"),
        ],
    )
    def test_content_digest_rejects_corrupted_material_fields(
        self,
        field: str,
        value: object,
        match: str,
    ) -> None:
        prop = FormalProperty("P1", "neuron", "description", "assert", "proven")
        setattr(prop, field, value)
        with pytest.raises(ValueError, match=match):
            FormalProofCertificate([prop]).content_sha256()

    def test_hash_and_report_reject_empty_explicit_timestamp(self) -> None:
        cert = FormalProofCertificate(
            [FormalProperty("P1", "neuron", "description", "assert", "proven")]
        )
        with pytest.raises(ValueError, match="generated_at"):
            cert.compute_hash(generated_at="")
        cert.compute_hash(generated_at="2026-07-12T18:30:00+00:00")
        assert "Formal Proof Certificate" in cert.generate_report()
        with pytest.raises(ValueError, match="generated_at"):
            cert.generate_report(generated_at="")
