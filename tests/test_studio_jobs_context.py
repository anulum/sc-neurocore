# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio job sandbox contract tests

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")


from sc_neurocore.studio.platform.jobs import (
    StudioJobArtifactUnavailable,
    StudioJobContext,
    StudioJobManager,
)


def test_studio_job_context_appends_and_publishes_live_event_artifact(
    tmp_path: Path,
) -> None:
    """Live JSONL artifacts can be appended first and manifested once."""

    work_dir = tmp_path / "job"
    work_dir.mkdir()
    context = StudioJobContext(
        job_id="sj_live_events",
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=4096,
    )

    context.append_artifact_event("events/live.jsonl", {"event": "epoch", "data": {"n": 1}})
    artifact = context.publish_existing_artifact("events/live.jsonl")

    payload = (work_dir / "events" / "live.jsonl").read_bytes()
    assert json.loads(payload.decode("utf-8")) == {"data": {"n": 1}, "event": "epoch"}
    assert artifact.relative_path == "events/live.jsonl"
    assert artifact.size_bytes == len(payload)
    assert artifact.sha256 == hashlib.sha256(payload).hexdigest()
    assert context.artifacts == (artifact,)


def test_studio_job_context_rejects_live_event_artifact_escape(tmp_path: Path) -> None:
    """Live JSONL artifact writes use the same confinement as normal artifacts."""

    work_dir = tmp_path / "job"
    work_dir.mkdir()
    context = StudioJobContext(
        job_id="sj_live_escape",
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=4096,
    )

    with pytest.raises(ValueError, match="escapes"):
        context.append_artifact_event("../escape.jsonl", {"event": "bad"})

    assert not (tmp_path / "escape.jsonl").exists()


def test_studio_job_context_rejects_invalid_live_event_payload(
    tmp_path: Path,
) -> None:
    """Live JSONL event writes reject non-JSON payloads."""

    work_dir = tmp_path / "job"
    work_dir.mkdir()
    context = StudioJobContext(
        job_id="sj_live_invalid_payload",
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=4096,
    )

    with pytest.raises(ValueError, match="payload must be JSON"):
        context.append_artifact_event("events/live.jsonl", {"bad": object()})


def test_studio_job_context_rejects_oversized_live_event_artifact(
    tmp_path: Path,
) -> None:
    """Live JSONL event writes enforce per-artifact byte ceilings."""

    work_dir = tmp_path / "job"
    work_dir.mkdir()
    context = StudioJobContext(
        job_id="sj_live_too_large",
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=8,
    )

    with pytest.raises(ValueError, match="exceeds configured size"):
        context.append_artifact_event("events/live.jsonl", {"event": "epoch"})


def test_studio_job_context_rejects_missing_or_oversized_existing_artifact(
    tmp_path: Path,
) -> None:
    """Publishing existing artifacts validates availability and byte ceilings."""

    work_dir = tmp_path / "job"
    work_dir.mkdir()
    context = StudioJobContext(
        job_id="sj_existing_artifact",
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=4,
    )
    artifact_path = work_dir / "events" / "live.jsonl"
    artifact_path.parent.mkdir()
    artifact_path.write_text("too-large", encoding="utf-8")

    with pytest.raises(ValueError, match="unavailable"):
        context.publish_existing_artifact("events/missing.jsonl")
    with pytest.raises(ValueError, match="exceeds configured size"):
        context.publish_existing_artifact("events/live.jsonl")


def test_studio_job_manager_tails_live_artifact_before_manifest(
    tmp_path: Path,
) -> None:
    """Live artifact reads are path-confined and available before completion."""

    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=1.0,
    )
    release = threading.Event()

    def task(context: StudioJobContext) -> dict[str, object]:
        context.append_artifact_event("training/events.jsonl", {"event": "epoch"})
        release.wait(timeout=1.0)
        return {"ok": True}

    record = manager.submit(
        kind="training",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )

    payload = b""
    offset = 0
    for _ in range(20):
        payload, offset = manager.read_live_artifact_bytes(
            record.job_id,
            "training/events.jsonl",
            offset=0,
        )
        if payload:
            break
        time.sleep(0.05)
    release.set()
    manager.wait(record.job_id, timeout_seconds=2.0)

    assert json.loads(payload.decode("utf-8")) == {"event": "epoch"}
    assert offset == len(payload)
    with pytest.raises(KeyError):
        manager.read_live_artifact_bytes(record.job_id, "../escape.jsonl", offset=0)


def test_studio_job_manager_rejects_invalid_live_artifact_read_bounds(
    tmp_path: Path,
) -> None:
    """Live artifact reads validate offsets and return empty missing tails."""

    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=1.0,
    )

    def task(_context: StudioJobContext) -> dict[str, object]:
        return {"ok": True}

    record = manager.submit(
        kind="training",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )
    manager.wait(record.job_id, timeout_seconds=2.0)

    with pytest.raises(ValueError, match="offset"):
        manager.read_live_artifact_bytes(record.job_id, "events/missing.jsonl", offset=-1)
    with pytest.raises(ValueError, match="read size"):
        manager.read_live_artifact_bytes(
            record.job_id,
            "events/missing.jsonl",
            offset=0,
            max_bytes=0,
        )
    assert manager.read_live_artifact_bytes(
        record.job_id,
        "events/missing.jsonl",
        offset=7,
    ) == (b"", 7)


def test_studio_job_manager_rejects_tampered_manifest_artifact(tmp_path: Path) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"synthesis"}),
        default_timeout_seconds=1.0,
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("reports/result.txt", b"original")
        return {"ok": True}

    record = manager.submit(
        kind="synthesis",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )
    manager.wait(record.job_id, timeout_seconds=2.0)
    (tmp_path / "jobs" / record.job_id / "reports" / "result.txt").write_bytes(b"tampered")

    with pytest.raises(StudioJobArtifactUnavailable, match="integrity"):
        manager.read_artifact(record.job_id, "reports/result.txt")


def test_studio_job_manager_rejects_artifact_path_traversal(tmp_path: Path) -> None:
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"compiler"}),
        default_timeout_seconds=1.0,
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("../escape.txt", b"bad")
        return {"unreachable": True}

    record = manager.submit(
        kind="compiler",
        owner="operator-1",
        request_id="req-1",
        task=task,
    )
    completed = manager.wait(record.job_id, timeout_seconds=2.0)

    assert completed.status == "failed"
    assert completed.error == "Studio job artifact path escapes the job directory."
    assert not (tmp_path / "escape.txt").exists()


def test_studio_job_context_rejects_symlink_artifact_escape(tmp_path: Path) -> None:
    work_dir = tmp_path / "job"
    outside_dir = tmp_path / "outside"
    work_dir.mkdir()
    outside_dir.mkdir()
    (work_dir / "linked").symlink_to(outside_dir, target_is_directory=True)
    context = StudioJobContext(
        job_id="sj_test",
        work_dir=work_dir,
        cancel_event=threading.Event(),
        max_artifact_bytes=1024,
    )

    context.check_cancelled()
    with pytest.raises(ValueError, match="escapes"):
        context.write_artifact("linked/escape.txt", b"bad")
