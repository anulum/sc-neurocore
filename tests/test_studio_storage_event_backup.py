# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Event archive backup and restored authority reads

"""Preserve full event custody and checkpoint bytes in a quiescent archive copy."""

from pathlib import Path
import shutil
import sqlite3

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.storage_isolated_jobs import IsolatedJobManager
from tests.studio_storage_generation_runs import api_runtime
from tests.studio_storage_generation_runs import authority as authority
from tests.studio_storage_generation_runs import ledger as ledger
from tests.studio_storage_generation_support import Authority
from tests.test_studio_storage_event_training import base as base
from tests.test_studio_storage_event_training import launcher as launcher
from tests.test_studio_storage_event_training import event_config as event_config
from tests.test_studio_storage_event_concurrency import submit_event_training
from tests.test_studio_storage_isolated_jobs import _configuration
from tests.test_studio_storage_isolated_jobs import manager as manager
from tests.test_studio_storage_isolated_jobs import request_on


def test_restored_event_archive_serves_full_contract_and_identical_checkpoint(
    manager: IsolatedJobManager,
    ledger: StudioJobLedger,
    event_config: dict[str, object],
    tmp_path: Path,
) -> None:
    """SQLite and sealed bytes restore together without depending on the spool."""
    job = submit_event_training(manager, event_config, "event-backup-completed")
    with request_on("/api/training/start"):
        completed = manager.wait(job.job_id, timeout_seconds=120)
    assert completed.status == "completed", completed.error
    with request_on("/api/studio/jobs/{job_id}/artifacts/{artifact_path:path}", "GET"):
        checkpoint = manager.read_artifact(completed.job_id, "training/model_state.pt")
    root = tmp_path / "restored"
    root.mkdir(mode=0o700)
    archive = root / "authority"
    archive.mkdir(mode=0o700)
    with sqlite3.connect(archive / ledger.path.name) as destination:
        ledger.connection().backup(destination)
        assert destination.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    shutil.copytree(ledger.path.parent / completed.job_id, archive / completed.job_id)
    restored_ledger = StudioJobLedger(root=archive)
    restored_authority = Authority(restored_ledger)
    runtime = api_runtime(root, restored_authority)
    restored = IsolatedJobManager(
        runtime,
        _configuration(root),
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=120.0,
    )
    try:
        with request_on("/api/studio/jobs", "GET"):
            assert restored.record(completed.job_id) == completed
            assert restored.list_records() == (completed,)
            assert restored.record(completed.job_id).training_config == event_config
        with request_on("/api/studio/jobs/{job_id}/artifacts/{artifact_path:path}", "GET"):
            assert restored.read_artifact(completed.job_id, "training/model_state.pt") == checkpoint
        assert not (root / "spool").exists()
    finally:
        restored_authority.join()
        runtime.live.close()
        restored_ledger.close()
