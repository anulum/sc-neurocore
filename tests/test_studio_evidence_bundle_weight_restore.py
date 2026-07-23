# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio evidence bundle weight-restore payloads

"""Weight-restore and attach evidence inclusion/rejection contracts."""

from __future__ import annotations

from tests.studio_evidence_bundle_support import *  # noqa: F403

def test_write_studio_evidence_bundle_includes_weight_restore(tmp_path: Path) -> None:
    context = StudioJobContext(
        job_id="sj_restore",
        work_dir=tmp_path / "restore",
        cancel_event=threading.Event(),
        max_artifact_bytes=1024 * 1024,
    )

    result = write_studio_evidence_bundle(
        context,
        weight_restore_payloads=(_weight_restore_response(),),
        clock=lambda: datetime(2026, 6, 20, tzinfo=UTC),
    )
    payload = result.to_public_dict()
    summary = cast(dict[str, object], payload["summary"])
    entry_type_counts = cast(dict[str, int], summary["entry_type_counts"])
    classification_counts = cast(dict[str, int], summary["evidence_classification_counts"])

    assert "evidence/training-weight-restores/000.json" in result.artifact_paths
    assert entry_type_counts["training_weight_restore_result"] == 1
    assert classification_counts["training"] == 1

@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda p: p.__setitem__("schema_version", "wrong"), "schema is unsupported"),
        (
            lambda p: p.__setitem__("evidence_classification", "analysis"),
            "classification is invalid",
        ),
        (lambda p: p.__setitem__("status", "failed"), "must be completed"),
        (lambda p: p.pop("materialization"), "requires materialization"),
        (
            lambda p: cast(dict[str, object], p["materialization"]).__setitem__(
                "weights_sha256", "z" * 64
            ),
            "weights_sha256 is invalid",
        ),
    ],
)
def test_write_studio_evidence_bundle_rejects_invalid_weight_restore(
    tmp_path: Path,
    mutate: Callable[[dict[str, object]], object],
    match: str,
) -> None:
    context = StudioJobContext(
        job_id="sj_restore_bad",
        work_dir=tmp_path / "restore-bad",
        cancel_event=threading.Event(),
        max_artifact_bytes=1024 * 1024,
    )
    payload = _weight_restore_response()
    mutate(payload)

    with pytest.raises(ValueError, match=match):
        write_studio_evidence_bundle(context, weight_restore_payloads=(payload,))

def test_write_studio_evidence_bundle_includes_weight_restore_attach(tmp_path: Path) -> None:
    context = StudioJobContext(
        job_id="sj_attach",
        work_dir=tmp_path / "attach",
        cancel_event=threading.Event(),
        max_artifact_bytes=1024 * 1024,
    )

    result = write_studio_evidence_bundle(
        context,
        weight_restore_attach_payloads=(_weight_restore_attach_response(),),
        clock=lambda: datetime(2026, 6, 20, tzinfo=UTC),
    )
    payload = result.to_public_dict()
    summary = cast(dict[str, object], payload["summary"])
    entry_type_counts = cast(dict[str, int], summary["entry_type_counts"])
    classification_counts = cast(dict[str, int], summary["evidence_classification_counts"])

    assert "evidence/training-weight-restore-attaches/000.json" in result.artifact_paths
    assert entry_type_counts["training_weight_restore_attach_result"] == 1
    assert classification_counts["training"] == 1

@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda p: p.__setitem__("schema_version", "wrong"), "schema is unsupported"),
        (lambda p: p.__setitem__("mode", "hot_swap"), "mode is unsupported"),
        (
            lambda p: p.__setitem__("evidence_classification", "analysis"),
            "classification is invalid",
        ),
        (lambda p: p.pop("target_job_id"), "target_job_id"),
        (
            lambda p: p.__setitem__("architecture_fingerprint", "nope"),
            "fingerprint is invalid",
        ),
    ],
)
def test_write_studio_evidence_bundle_rejects_invalid_weight_restore_attach(
    tmp_path: Path,
    mutate: Callable[[dict[str, object]], object],
    match: str,
) -> None:
    context = StudioJobContext(
        job_id="sj_attach_bad",
        work_dir=tmp_path / "attach-bad",
        cancel_event=threading.Event(),
        max_artifact_bytes=1024 * 1024,
    )
    payload = _weight_restore_attach_response()
    mutate(payload)

    with pytest.raises(ValueError, match=match):
        write_studio_evidence_bundle(context, weight_restore_attach_payloads=(payload,))
