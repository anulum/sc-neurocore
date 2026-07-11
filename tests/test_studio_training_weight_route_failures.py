# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio training-weight route failure tests

"""Exercise training-weight failures through the public Studio API."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.api import training_weights
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import (
    StudioJobArtifactUnavailable,
    StudioJobManager,
    StudioJobRecord,
    StudioJobStatus,
    StudioRuntimeSettings,
)


class _RestorePlan:
    """Minimal restore plan used to isolate HTTP failure mapping."""

    def to_public_dict(self) -> dict[str, object]:
        """Return a valid path-free restore-plan payload."""

        return {}


def _build_client(tmp_path: Path) -> TestClient:
    """Return a TestClient backed by an isolated Studio job root."""

    application = create_app(
        StudioRuntimeSettings(
            audit_log_path=str(tmp_path / "audit" / "studio.jsonl"),
            job_root_path=str(tmp_path / "jobs"),
            job_default_timeout_seconds=1.0,
        )
    )
    return TestClient(application, base_url="http://127.0.0.1")


def _patch_restore_prerequisites(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch source metadata so restore tests reach artifact or worker handling."""

    monkeypatch.setattr(
        training_weights,
        "get_training_status",
        lambda *_args, **_kwargs: {"status": "completed", "weight_checkpoint": {}},
    )
    monkeypatch.setattr(
        training_weights,
        "build_training_weight_restore_plan",
        lambda **_kwargs: _RestorePlan(),
    )


def test_weight_restore_rejects_missing_source_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed stored status maps to a stable conflict response."""

    monkeypatch.setattr(
        training_weights,
        "get_training_status",
        lambda *_args, **_kwargs: {"status": None, "weight_checkpoint": {}},
    )
    response = _build_client(tmp_path).post(
        "/api/studio/training/weight-restore",
        json={"source_job_id": "sj_source"},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "training_status_unavailable"


@pytest.mark.parametrize(
    ("error_type", "expected_status", "expected_detail"),
    [
        (KeyError, 404, "training_weight_artifact_not_found"),
        (
            StudioJobArtifactUnavailable,
            409,
            "training_weight_artifact_unavailable",
        ),
    ],
)
def test_weight_restore_maps_artifact_read_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
    expected_status: int,
    expected_detail: str,
) -> None:
    """Confined artifact failures never leak internal paths or exception text."""

    _patch_restore_prerequisites(monkeypatch)

    def _read_artifact(
        self: StudioJobManager,
        job_id: str,
        relative_path: str,
    ) -> object:
        del self, job_id, relative_path
        raise error_type("private/path")

    monkeypatch.setattr(StudioJobManager, "read_artifact", _read_artifact)
    response = _build_client(tmp_path).post(
        "/api/studio/training/weight-restore",
        json={"source_job_id": "sj_source"},
    )

    assert response.status_code == expected_status
    assert response.json()["detail"] == expected_detail
    assert "private/path" not in response.text


@pytest.mark.parametrize(
    ("status", "expected_status", "expected_detail"),
    [
        ("pending", 503, "studio_job_wait_exceeded"),
        ("timed_out", 504, "studio_job_timed_out"),
        ("failed", 500, "studio_job_failed"),
    ],
)
def test_weight_restore_maps_worker_terminal_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: StudioJobStatus,
    expected_status: int,
    expected_detail: str,
) -> None:
    """Every non-success worker terminal state has a stable HTTP mapping."""

    _patch_restore_prerequisites(monkeypatch)

    def _read_artifact(
        self: StudioJobManager,
        job_id: str,
        relative_path: str,
    ) -> SimpleNamespace:
        del self, job_id, relative_path
        return SimpleNamespace(payload=b"{}")

    def _submit(self: StudioJobManager, **_kwargs: object) -> StudioJobRecord:
        del self
        return StudioJobRecord(
            job_id="restore-route-error",
            kind="training",
            owner="studio-training-weight-restore",
            request_id=None,
            status="pending",
            execution_model="thread",
            created_at_utc="2026-07-11T00:00:00Z",
        )

    def _wait(
        self: StudioJobManager,
        job_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> StudioJobRecord:
        del self, timeout_seconds
        return StudioJobRecord(
            job_id=job_id,
            kind="training",
            owner="studio-training-weight-restore",
            request_id=None,
            status=status,
            execution_model="thread",
            created_at_utc="2026-07-11T00:00:00Z",
        )

    monkeypatch.setattr(StudioJobManager, "read_artifact", _read_artifact)
    monkeypatch.setattr(StudioJobManager, "submit", _submit)
    monkeypatch.setattr(StudioJobManager, "wait", _wait)
    response = _build_client(tmp_path).post(
        "/api/studio/training/weight-restore",
        json={"source_job_id": "sj_source"},
    )

    assert response.status_code == expected_status
    assert response.json()["detail"] == expected_detail


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_detail"),
    [
        ("training_weight_artifact_not_found", 404, "training_weight_artifact_not_found"),
        ("unexpected_backend_failure", 500, "training_weight_attach_failed"),
    ],
)
def test_warm_attach_maps_backend_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: str,
    expected_status: int,
    expected_detail: str,
) -> None:
    """Warm-attach backend failures retain the public error vocabulary."""

    def _start_training_attach(
        source_job_id: str,
        config: dict[str, Any],
        job_manager: StudioJobManager,
        *,
        expected_config_sha256: str | None = None,
    ) -> dict[str, Any]:
        del source_job_id, config, job_manager, expected_config_sha256
        return {"error": error}

    monkeypatch.setattr(training_weights, "start_training_attach", _start_training_attach)
    response = _build_client(tmp_path).post(
        "/api/studio/training/weight-restore/attach",
        json={"source_job_id": "sj_source", "config": {}},
    )

    assert response.status_code == expected_status
    assert response.json()["detail"] == expected_detail


@pytest.mark.parametrize(
    ("result", "raises_value_error", "expected_status", "expected_detail"),
    [
        ({}, True, 422, "invalid_expected_digest"),
        (
            {"error": "training_weight_artifact_not_found"},
            False,
            404,
            "training_weight_artifact_not_found",
        ),
        (
            {"error": "unexpected_backend_failure"},
            False,
            500,
            "training_weight_attach_failed",
        ),
    ],
)
def test_live_attach_maps_backend_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result: dict[str, object],
    raises_value_error: bool,
    expected_status: int,
    expected_detail: str,
) -> None:
    """Live-attach validation and backend failures map without detail leakage."""

    def _request_live_training_weight_attach(
        target_job_id: str,
        source_job_id: str,
        job_manager: StudioJobManager,
        *,
        expected_config_sha256: str | None = None,
    ) -> dict[str, Any]:
        del target_job_id, source_job_id, job_manager, expected_config_sha256
        if raises_value_error:
            raise ValueError("invalid_expected_digest")
        return cast(dict[str, Any], result)

    monkeypatch.setattr(
        training_weights,
        "request_live_training_weight_attach",
        _request_live_training_weight_attach,
    )
    response = _build_client(tmp_path).post(
        "/api/studio/training/weight-restore/attach/live",
        json={"target_job_id": "sj_target", "source_job_id": "sj_source"},
    )

    assert response.status_code == expected_status
    assert response.json()["detail"] == expected_detail
