# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (audit_report) from former test_scnir_compatibility.py

from __future__ import annotations

from tests.scnir_compatibility_support import *  # noqa: F403


def test_scnir_compatibility_audit_report_summarises_evidence() -> None:
    from sc_neurocore.ir import build_scnir_compatibility_audit

    report = build_scnir_compatibility_audit(evidence_root=REPO_ROOT)

    matrix = json.loads(json.dumps(scnir_compatibility_matrix_dicts(), sort_keys=True))
    evidence_paths = sorted(
        {path for row in scnir_compatibility_matrix() for path in row.audit_evidence}
    )
    matrix_digest = hashlib.sha256(
        (json.dumps(matrix, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    ).hexdigest()
    first_evidence = REPO_ROOT / evidence_paths[0]
    first_digest = hashlib.sha256(first_evidence.read_bytes()).hexdigest()

    assert report["schema_version"] == "sc-neurocore.scnir.compatibility-audit.v0.2"
    assert report["status"] == "valid"
    assert report["evidence_root"] == str(REPO_ROOT.resolve())
    assert report["primitive_count"] == len(matrix)
    support_level_counts = cast(dict[str, int], report["support_level_counts"])
    assert support_level_counts["metadata_and_hdl"] >= 1
    assert report["closure_status"] == "closed_for_local_handoff"
    assert report["closure_blocker_count"] == 0
    assert report["parser_only_primitives"] == []
    assert report["metadata_only_primitives"] == []
    assert report["boundary_primitives"] == ["Input", "Output"]
    assert report["closed_handoff_primitives"] == sorted(
        row.nir_primitive
        for row in scnir_compatibility_matrix()
        if row.support_level == "metadata_and_hdl"
    )
    assert report["requires_external_hardware_evidence"] is True
    assert report["external_hardware_evidence_status"] == "not_claimed"
    assert report["audit_evidence_file_count"] == len(evidence_paths)
    assert report["audit_evidence_paths"] == evidence_paths
    assert report["matrix_sha256"] == matrix_digest
    audit_evidence_files = cast(list[dict[str, object]], report["audit_evidence_files"])
    assert audit_evidence_files[0] == {
        "path": evidence_paths[0],
        "sha256": first_digest,
        "size_bytes": first_evidence.stat().st_size,
    }
    assert report["matrix"] == matrix
