# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio job ledger artifact custody

"""A completed job's artifact manifest is sealed and survives a restart."""

from __future__ import annotations

from contextlib import closing
import json
from pathlib import Path
import sqlite3
from typing import Any

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_rows import StudioJobLedgerCorrupt
from sc_neurocore.studio.platform.jobs import StudioJobManager
from sc_neurocore.studio.platform.jobs_context import StudioJobContext
from sc_neurocore.studio.platform.jobs_models import (
    StudioJobRejected,
)


def _job_manager(root: Path) -> StudioJobManager:
    """Open a manager over the shared root, reconciling on construction."""
    return StudioJobManager(
        root=root, allowed_kinds=frozenset({"analysis"}), default_timeout_seconds=30.0
    )


def _ledger(root: Path) -> StudioJobLedger:
    """Open a second writer over the same durable root."""
    return StudioJobLedger(root=root)


class TestArtifactCustody:
    @pytest.mark.parametrize(
        "replacement",
        [
            {"result": {"written": False}},
            {"artifacts": ()},
            {"error": "late error"},
            {"started_at_utc": "2026-01-01T00:00:00Z"},
            {"finished_at_utc": "2026-01-01T00:00:00Z"},
        ],
        ids=["result", "manifest", "error", "start", "finish"],
    )
    def test_late_writer_cannot_change_a_real_completed_job(
        self, tmp_path: Path, replacement: dict[str, Any]
    ) -> None:
        """Independent ledger writers cannot revise the real runner's sealed evidence."""
        root = tmp_path / "jobs"
        manager = _job_manager(root)

        def task(context: StudioJobContext) -> dict[str, object]:
            context.write_artifact("result.bin", b"original payload")
            return {"written": True}

        submitted = manager.submit(kind="analysis", owner="alice", request_id=None, task=task)
        sealed = manager.wait(submitted.job_id, timeout_seconds=5.0)
        assert sealed.status == "completed"
        writer = _ledger(root)
        history = writer.transitions(sealed.job_id)
        with pytest.raises(StudioJobRejected, match="cannot rewrite"):
            writer.transition(sealed.job_id, "completed", **replacement)
        assert manager.record(sealed.job_id) == sealed
        assert writer.transitions(sealed.job_id) == history
        assert manager.read_artifact(sealed.job_id, "result.bin").payload == b"original payload"
        assert (
            writer.transition(
                sealed.job_id,
                "completed",
                started_at_utc=sealed.started_at_utc,
                finished_at_utc=sealed.finished_at_utc,
                error=sealed.error,
                result=sealed.result,
                artifacts=sealed.artifacts,
            )
            == sealed
        )

    def test_a_completed_job_keeps_its_manifest_across_a_restart(self, tmp_path: Path) -> None:
        root = tmp_path / "jobs"
        manager = _job_manager(root)

        def task(context: StudioJobContext) -> dict[str, object]:
            context.write_artifact("result.bin", b"payload")
            return {"written": True}

        record = manager.submit(kind="analysis", owner="alice", request_id="req-3", task=task)
        done = manager.wait(record.job_id, 30.0)
        assert done.status == "completed"
        assert done.artifacts != ()

        restarted = _job_manager(root)
        recovered = restarted.record(record.job_id)

        assert recovered.artifacts == done.artifacts
        assert restarted.read_artifact(record.job_id, "result.bin").payload == b"payload"


def _canonical_training_config() -> str:
    from sc_neurocore.studio.training_contract import resolve_training_config

    resolved = resolve_training_config({}).to_public_dict()
    return json.dumps(resolved, sort_keys=True, separators=(",", ":"))


@pytest.mark.parametrize(
    "kind,stored,message",
    [
        ("analysis", "{}", "non-training job"),
        ("training", "x" * 4097, "exceeds 4096 bytes"),
        ("training", "[1]", "not an object"),
        ("training", '{"epochs": -1}', "is invalid"),
        ("training", "canonical-with-space", "not canonical"),
    ],
)
def test_damaged_stored_training_configuration_is_refused(
    tmp_path: Path, kind: str, stored: str, message: str
) -> None:
    """A damaged training snapshot in a stored row is reported, never silently read."""
    ledger = _ledger(tmp_path)
    try:
        job_id = "sj_00000000000000c1"
        ledger.create(
            job_id=job_id,
            kind=kind,
            actor="alice",
            workspace="default",
            request_id=None,
            idempotency_key=None,
            experiment_sha256=None,
            admission=None,
            execution_model="thread",
        )
        if stored == "canonical-with-space":
            stored = _canonical_training_config().replace(",", ", ", 1)
        with closing(sqlite3.connect(ledger.path, isolation_level=None)) as other:
            other.execute("UPDATE jobs SET training_config=? WHERE job_id=?", (stored, job_id))
        with pytest.raises(StudioJobLedgerCorrupt, match=message):
            ledger.record(job_id)
    finally:
        ledger.close()
