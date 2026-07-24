# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio evidence bundle test support

"""Shared fixtures and payload builders for Studio evidence-bundle tests."""

from __future__ import annotations

import json

import math

import threading

from collections.abc import Callable

from datetime import datetime, timezone

from pathlib import Path

from typing import cast

import pytest

from fastapi import FastAPI

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app

from sc_neurocore.studio.platform import (
    STUDIO_EVIDENCE_BUNDLE_SCHEMA_VERSION,
    AuditEvent,
    JsonlAuditSink,
    StudioJobArtifact,
    StudioJobContext,
    StudioJobManager,
    StudioJobRecord,
    StudioRuntimeSettings,
    write_studio_evidence_bundle,
)

from sc_neurocore.studio.platform.action_evidence import (
    write_studio_action_evidence_manifest,
)

from sc_neurocore.studio.project import save_project

UTC = timezone.utc


def _simulation_payload() -> dict[str, object]:
    """Return a minimal Studio simulation response carrying run metadata."""

    return {
        "current_trace": [1.0, 1.0],
        "dt": 0.1,
        "n_steps": 2,
        "run_metadata": {
            "dt": 0.1,
            "evidence_classification": "simulation",
            "input_sha256": "1" * 64,
            "n_steps": 2,
            "result_sha256": "2" * 64,
            "sample_count": 2,
            "schema_version": "studio.simulation-run.v1",
            "source": "ode",
            "spike_count": 0,
            "status": "completed",
            "state_variables": ["v"],
        },
        "spike_count": 0,
        "spikes": [],
        "states": {"v": [0.0, 0.1]},
        "time": [0.0, 0.1],
    }


def _analysis_payload() -> dict[str, object]:
    """Return a minimal Studio analysis response carrying analysis metadata."""

    return {
        "analysis_metadata": {
            "analysis_type": "fi_curve",
            "evidence_classification": "analysis",
            "input_sha256": "3" * 64,
            "output_keys": ["currents", "rates"],
            "result_sha256": "4" * 64,
            "schema_version": "studio.analysis-result.v1",
            "source": "ode",
            "status": "completed",
        },
        "currents": [0.0, 1.0],
        "rates": [0.0, 10.0],
    }


def _project_payload() -> dict[str, object]:
    """Return a minimal saved Studio project payload."""

    return {
        "name": "demo",
        "saved_at": 1_782_000_000.0,
        "state": {"duration": 10},
        "version": "0.3.0",
    }


def _action_evidence_payload(job_id: str) -> dict[str, object]:
    """Return a minimal worker action-evidence manifest payload."""

    return {
        "action_kind": "studio.compile",
        "artifacts": [
            {
                "relative_path": "compiler/result.json",
                "sha256": "5" * 64,
                "size_bytes": 128,
            }
        ],
        "evidence_classification": "compile",
        "generated_at_utc": "2026-06-20T00:00:00Z",
        "job_id": job_id,
        "payload_sha256": "6" * 64,
        "principal_id": None,
        "replay_route": "POST /api/compile",
        "request_id": None,
        "schema_version": "studio.action-evidence.v1",
        "status": "completed",
    }


def _default_flow_run_payload() -> dict[str, object]:
    """Return a minimal default-flow run response with reproducibility hashes."""

    return {
        "action_order": ["auto_tune_adaptive_precision"],
        "evidence_classification": "default_flow",
        "executed_count": 1,
        "execution_time_ms": 1.0,
        "flow_id": "studio_default_adaptive_precision_v1",
        "preset_id": "fpga_precision",
        "reproducibility_manifest": {
            "hash_algorithm": "sha256",
            "inputs_fingerprint_sha256": "7" * 64,
            "run_fingerprint_sha256": "8" * 64,
        },
        "results": [],
        "schema_version": "sc-neurocore.studio.default-flow-run.v1",
        "status": "completed",
    }


def _default_flow_attestation_payload() -> dict[str, object]:
    """Return a minimal default-flow attestation for the test run payload."""

    return {
        "attestation_fingerprint_sha256": "9" * 64,
        "evidence_classification": "default_flow",
        "flow_id": "studio_default_adaptive_precision_v1",
        "inputs_fingerprint_sha256": "7" * 64,
        "plan_fingerprint_sha256": "a" * 64,
        "preset_id": "fpga_precision",
        "run_fingerprint_sha256": "8" * 64,
        "schema_version": "sc-neurocore.studio.default-flow-attestation.v1",
        "status": "completed",
    }


def _client_with_evidence_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[FastAPI, TestClient]:
    """Create a Studio app with durable job, audit, and project roots."""

    projects_root = tmp_path / "projects"
    monkeypatch.setattr("sc_neurocore.studio.project._PROJECTS_DIR", str(projects_root))
    audit_path = tmp_path / "audit" / "studio.jsonl"
    settings = StudioRuntimeSettings(
        audit_log_path=str(audit_path),
        job_root_path=str(tmp_path / "jobs"),
        job_default_timeout_seconds=10.0,
    )
    app = create_app(settings)
    sink = JsonlAuditSink(audit_path)
    sink.record(
        AuditEvent(
            action="studio.test",
            route="/api/test",
            principal_id="operator",
            decision="allow",
            reason="authorized",
            request_id="req-test",
            timestamp_utc="2026-06-20T00:00:00Z",
        )
    )
    return app, TestClient(app, base_url="http://127.0.0.1")


def _job_manager(app: FastAPI) -> StudioJobManager:
    """Return the app-local Studio job manager."""

    return cast(StudioJobManager, app.state.studio_job_manager)


def _json_artifact(
    manager: StudioJobManager,
    job_id: str,
    relative_path: str,
) -> dict[str, object]:
    """Read one JSON job artifact through the verified artifact API."""

    payload = manager.read_artifact(job_id, relative_path)
    decoded = json.loads(payload.payload.decode("utf-8"))
    assert isinstance(decoded, dict)
    return cast(dict[str, object], decoded)


def _model_scan_response() -> dict[str, object]:
    """Return a minimal Studio model-scan response carrying scan metadata."""

    scan_metadata = {
        "current": 10.0,
        "duration": 100.0,
        "evidence_classification": "analysis",
        "input_sha256": "5" * 64,
        "model_count": 2,
        "pattern_counts": {"tonic": 2},
        "result_sha256": "6" * 64,
        "schema_version": "studio.model-scan.v1",
        "status": "completed",
    }
    return {
        "models": [{"name": "AdExNeuron", "pattern": "tonic"}],
        "scan_metadata": scan_metadata,
        "schema_version": "studio.model-scan.v1",
    }


def _weight_restore_response() -> dict[str, object]:
    """Return a minimal Studio training weight-restore evidence response."""

    return {
        "schema_version": "studio.training.weight-restore.v1",
        "evidence_classification": "training",
        "status": "completed",
        "source_job_id": "sj_training",
        "source_status": "completed",
        "materialization": {
            "architecture": "64->10",
            "config_sha256": "7" * 64,
            "format": "torch_state_dict",
            "framework": "pytorch",
            "loaded_key_count": 2,
            "metadata_sha256": "8" * 64,
            "parameter_count": 8,
            "schema_version": "studio.training.weight-materialization.v1",
            "source_job_id": "sj_training",
            "weights_sha256": "9" * 64,
        },
    }


def _weight_restore_attach_response() -> dict[str, object]:
    """Return a minimal Studio training weight-restore attach evidence response."""

    return {
        "schema_version": "studio.training.weight-restore-attach.v1",
        "evidence_classification": "training",
        "status": "completed",
        "mode": "warm_start",
        "source_job_id": "sj_training",
        "target_job_id": "sj_attach",
        "target_architecture": "64->10",
        "target_parameter_count": 8,
        "architecture_fingerprint": "a" * 64,
        "materialization": {
            "architecture": "64->10",
            "config_sha256": "7" * 64,
            "format": "torch_state_dict",
            "framework": "pytorch",
            "loaded_key_count": 2,
            "metadata_sha256": "8" * 64,
            "parameter_count": 8,
            "schema_version": "studio.training.weight-materialization.v1",
            "source_job_id": "sj_training",
            "weights_sha256": "9" * 64,
        },
    }


__all__ = [
    "annotations",
    "json",
    "math",
    "threading",
    "Callable",
    "datetime",
    "timezone",
    "Path",
    "cast",
    "pytest",
    "FastAPI",
    "TestClient",
    "create_app",
    "STUDIO_EVIDENCE_BUNDLE_SCHEMA_VERSION",
    "AuditEvent",
    "JsonlAuditSink",
    "StudioJobArtifact",
    "StudioJobContext",
    "StudioJobManager",
    "StudioJobRecord",
    "StudioRuntimeSettings",
    "write_studio_evidence_bundle",
    "write_studio_action_evidence_manifest",
    "save_project",
    "UTC",
    "_simulation_payload",
    "_analysis_payload",
    "_project_payload",
    "_action_evidence_payload",
    "_default_flow_run_payload",
    "_default_flow_attestation_payload",
    "_client_with_evidence_state",
    "_job_manager",
    "_json_artifact",
    "_model_scan_response",
    "_weight_restore_response",
    "_weight_restore_attach_response",
]
