# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio evidence bundle model-scan payloads

"""Model-scan inclusion and rejection contracts inside evidence bundles."""

from __future__ import annotations

from tests.studio_evidence_bundle_support import *  # noqa: F403

def test_write_studio_evidence_bundle_includes_model_scan(tmp_path: Path) -> None:
    context = StudioJobContext(
        job_id="sj_scan",
        work_dir=tmp_path / "scan",
        cancel_event=threading.Event(),
        max_artifact_bytes=1024 * 1024,
    )

    result = write_studio_evidence_bundle(
        context,
        model_scan_payloads=(_model_scan_response(),),
        clock=lambda: datetime(2026, 6, 20, tzinfo=UTC),
    )
    payload = result.to_public_dict()
    summary = cast(dict[str, object], payload["summary"])
    entry_type_counts = cast(dict[str, int], summary["entry_type_counts"])
    classification_counts = cast(dict[str, int], summary["evidence_classification_counts"])

    assert "evidence/model-scans/000.json" in result.artifact_paths
    assert "model_scan_result" in json.dumps(payload)
    assert entry_type_counts["model_scan_result"] == 1
    assert classification_counts["analysis"] == 1

@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda p: p.__setitem__("schema_version", "wrong"), "unsupported scan metadata"),
        (lambda p: p.pop("scan_metadata"), "requires scan metadata"),
        (
            lambda p: cast(dict[str, object], p["scan_metadata"]).__setitem__(
                "evidence_classification", "synthesis"
            ),
            "classified as analysis evidence",
        ),
        (
            lambda p: cast(dict[str, object], p["scan_metadata"]).__setitem__("status", "failed"),
            "completed evidence status",
        ),
    ],
)
def test_write_studio_evidence_bundle_rejects_invalid_model_scan(
    tmp_path: Path,
    mutate: Callable[[dict[str, object]], object],
    match: str,
) -> None:
    context = StudioJobContext(
        job_id="sj_scan_bad",
        work_dir=tmp_path / "scan-bad",
        cancel_event=threading.Event(),
        max_artifact_bytes=1024 * 1024,
    )
    payload = _model_scan_response()
    mutate(payload)

    with pytest.raises(ValueError, match=match):
        write_studio_evidence_bundle(context, model_scan_payloads=(payload,))
