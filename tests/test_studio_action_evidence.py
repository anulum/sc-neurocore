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
import threading
from pathlib import Path

from sc_neurocore.studio.platform import STUDIO_ACTION_EVIDENCE_SCHEMA_VERSION
from sc_neurocore.studio.platform.action_evidence import (
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
        json.dumps(result, sort_keys=True, default=str).encode("utf-8")
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
