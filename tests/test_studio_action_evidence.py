# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio action evidence manifest tests

"""Tests for Studio action evidence manifests."""

from __future__ import annotations

import hashlib
import json
import math
import threading
from pathlib import Path
from typing import cast

import pytest

from sc_neurocore.studio.platform import STUDIO_ACTION_EVIDENCE_SCHEMA_VERSION
from sc_neurocore.studio.platform.action_evidence import (
    EvidenceClassification,
    EvidenceStatus,
    write_studio_action_evidence_manifest,
)
from sc_neurocore.studio.platform.jobs import StudioJobContext


def test_write_studio_action_evidence_manifest_is_path_free(tmp_path: Path) -> None:
    """Action evidence records route, classification, artifacts, and payload hash."""

    context = StudioJobContext(
        job_id="sj_test",
        work_dir=tmp_path,
        cancel_event=threading.Event(),
        max_artifact_bytes=4096,
    )
    result = {"success": True, "target": "ice40"}
    result_artifact = context.write_artifact("synthesis/result.json", json.dumps(result))

    evidence = write_studio_action_evidence_manifest(
        context,
        action_kind="studio.synthesis.run",
        result=result,
        result_artifact=result_artifact,
        evidence_artifact_path="synthesis/evidence.json",
        evidence_classification="synthesis",
        replay_route="POST /api/synth/run",
        request_id="req-1",
        principal_id="operator-1",
    )

    expected_hash = hashlib.sha256(
        json.dumps(result, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    payload = evidence.to_public_dict()
    assert payload["schema_version"] == STUDIO_ACTION_EVIDENCE_SCHEMA_VERSION
    assert payload["action_kind"] == "studio.synthesis.run"
    assert payload["evidence_classification"] == "synthesis"
    assert payload["job_id"] == "sj_test"
    assert payload["payload_sha256"] == expected_hash
    assert payload["principal_id"] == "operator-1"
    assert payload["replay_route"] == "POST /api/synth/run"
    assert payload["request_id"] == "req-1"
    assert payload["status"] == "completed"
    assert payload["artifacts"] == [result_artifact.to_public_dict()]
    assert evidence.artifact.relative_path == "synthesis/evidence.json"
    assert str(tmp_path) not in json.dumps(payload)


def test_write_studio_action_evidence_manifest_records_error_status(tmp_path: Path) -> None:
    """Action evidence can describe failed terminal worker actions."""

    context = StudioJobContext(
        job_id="sj_failed",
        work_dir=tmp_path,
        cancel_event=threading.Event(),
        max_artifact_bytes=4096,
    )
    result = {"error": "solver_failed", "status": "failed"}
    result_artifact = context.write_artifact("training/status.json", json.dumps(result))

    evidence = write_studio_action_evidence_manifest(
        context,
        action_kind="studio.training.run",
        result=result,
        result_artifact=result_artifact,
        evidence_artifact_path="training/evidence.json",
        evidence_classification="training",
        replay_route="POST /api/training/start",
        status="failed",
        error_message="solver_failed",
    )

    payload = evidence.to_public_dict()
    assert payload["evidence_classification"] == "training"
    assert payload["error_message"] == "solver_failed"
    assert payload["status"] == "failed"


def test_write_studio_action_evidence_manifest_rejects_invalid_fields(
    tmp_path: Path,
) -> None:
    """Controlled evidence fields fail closed before writing a manifest."""

    context = StudioJobContext(
        job_id="sj_invalid_fields",
        work_dir=tmp_path,
        cancel_event=threading.Event(),
        max_artifact_bytes=4096,
    )
    result_artifact = context.write_artifact("synthesis/result.json", "{}")

    invalid_classification = cast(EvidenceClassification, "screenshots")
    invalid_status = cast(EvidenceStatus, "partial")

    for kwargs, message in (
        ({"action_kind": "bad action"}, "action kind"),
        ({"evidence_classification": invalid_classification}, "classification"),
        ({"status": invalid_status}, "status"),
        ({"replay_route": "POST api/synth/run"}, "replay route"),
    ):
        with pytest.raises(ValueError, match=message):
            write_studio_action_evidence_manifest(
                context,
                action_kind=cast(str, kwargs.get("action_kind", "studio.synthesis.run")),
                result={"ok": True},
                result_artifact=result_artifact,
                evidence_artifact_path=f"synthesis/{message}.json",
                evidence_classification=cast(
                    EvidenceClassification,
                    kwargs.get("evidence_classification", "synthesis"),
                ),
                replay_route=cast(str, kwargs.get("replay_route", "POST /api/synth/run")),
                status=cast(EvidenceStatus, kwargs.get("status", "completed")),
            )


def test_write_studio_action_evidence_manifest_rejects_non_portable_result(
    tmp_path: Path,
) -> None:
    """Payload digests reject non-finite JSON values instead of stringifying."""

    context = StudioJobContext(
        job_id="sj_invalid_payload",
        work_dir=tmp_path,
        cancel_event=threading.Event(),
        max_artifact_bytes=4096,
    )
    result_artifact = context.write_artifact("training/status.json", "{}")

    with pytest.raises(ValueError, match="Out of range float values"):
        write_studio_action_evidence_manifest(
            context,
            action_kind="studio.training.run",
            result={"loss": math.nan},
            result_artifact=result_artifact,
            evidence_artifact_path="training/evidence.json",
            evidence_classification="training",
            replay_route="POST /api/training/start",
        )
