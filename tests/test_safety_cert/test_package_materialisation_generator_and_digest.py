# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (generator_and_digest) from former test_package_materialisation.py

from __future__ import annotations

from tests.test_safety_cert.package_materialisation_support import *  # noqa: F403


def test_default_generator_is_fail_closed() -> None:
    """Missing evidence must remain visible rather than being fabricated."""
    package = _package(explicit_evidence=False)
    assert package.checklist_coverage == 0.0
    assert all(item.status == "not_addressed" and not item.evidence for item in package.checklist)
    assert "Coverage: 0.0% (0/1)" in package.traceability_report
    assert "hdl/neuron.v" not in package.traceability_report
    assert "Status: not assessed" in package.fmeda_report
    assert "Status: not assessed" in package.wcet_report
    assert package.package_hash == package.content_sha256()[:32]


def test_explicit_evidence_flows_through_every_report() -> None:
    """Caller evidence and assumptions must reach their owning artifacts."""
    package = _package()
    assert "Coverage: 100.0% (1/1)" in package.traceability_report
    assert "rtl/neuron.sv" in package.traceability_report
    assert "| REQ_001 | IEC 61508 | SIL 2 | verified | 1 | 1 |" in package.traceability_report
    assert "12.5 FIT" in package.fmeda_report
    assert "Input-derived bound" in package.wcet_report
    assert package.checklist_coverage == pytest.approx(1 / 7)
    assert "evidence/formal-review.md" in package.checklist_report()


def test_fixed_timestamp_makes_package_content_reproducible() -> None:
    """Equivalent inputs must yield identical package and artifact digests."""
    first = _package()
    second = _package()
    assert first.artifacts() == second.artifacts()
    assert first.content_sha256() == second.content_sha256()
    assert first.package_hash == second.package_hash


def test_formal_digest_covers_all_material_fields_and_tool_version() -> None:
    """Changing any material proof field must change the full digest."""
    base = _property()
    variants = [
        replace(base, prop_id="P-SAFE-002"),
        replace(base, module="encoder"),
        replace(base, description="Reset dominates"),
        replace(base, property_type="cover"),
        replace(base, status="failed"),
        replace(base, engine="SymbiYosys 2.5.0"),
        replace(base, depth=64),
        replace(base, sby_file="formal/other.sby"),
    ]
    digests = {FormalProofCertificate([base], tool_version="sby-2.4").content_sha256()}
    digests.update(
        FormalProofCertificate([variant], tool_version="sby-2.4").content_sha256()
        for variant in variants
    )
    digests.add(FormalProofCertificate([base], tool_version="sby-2.5").content_sha256())
    assert len(digests) == len(variants) + 2
