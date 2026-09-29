# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Large event declarations in the bounded job ledger

"""Exercise real ledger durability, migration and tamper refusal for event inputs."""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import sqlite3

from starlette.testclient import TestClient
from pathlib import Path

import pytest

from sc_neurocore.datasets.encoders import EventBinning
from sc_neurocore.datasets.manifest import build_manifest
from sc_neurocore.datasets.splits import group_split
from sc_neurocore.studio.event_training_contract import EventTrainingContract
from sc_neurocore.studio.event_training_data import DATASET_ROOT_ENV
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import StudioRuntimeSettings
from tests.test_studio_event_training_native import _train
from tests.test_studio_event_training_process import _wait_terminal
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_ledger_schema import LEDGER_FILENAME, SCHEMA_V1
from sc_neurocore.studio.platform.jobs_ledger_rows import StudioJobLedgerCorrupt
from sc_neurocore.studio.training_contract import resolve_training_config
from tests.event_dataset_support import write_shd


def _configuration(root: Path, count: int = 120) -> dict[str, object]:
    """Create actual SHD input whose portable declaration exceeds the row limit."""
    write_shd(root, {"train": [index % 6 for index in range(count)], "test": [6]})
    manifest = build_manifest("shd", root, version="generated-format-fixture")
    contract = EventTrainingContract(
        manifest,
        group_split(manifest, fractions={"train": 0.7, "evaluation": 0.3}, seed=7),
        EventBinning(1.0, 4, 700, 1, "merge"),
        "train",
        "evaluation",
    )
    return resolve_training_config(
        {
            "dataset": "shd",
            "epochs": 1,
            "batch_size": 3,
            "hidden": [4],
            "timesteps": 4,
            "seed": 7,
            "event_data": contract.to_dict(),
        }
    ).to_public_dict()


def _create(ledger: StudioJobLedger, config: dict[str, object], job_id: str = "job") -> None:
    """Admit the public resolved snapshot through the actual ledger transaction."""
    ledger.create(
        job_id=job_id,
        kind="training",
        actor="verification",
        workspace="default",
        request_id=None,
        idempotency_key=None,
        experiment_sha256=None,
        admission=None,
        execution_model="process",
        training_config=config,
    )


def test_large_event_configuration_survives_reopen_without_relaxing_row_limit(
    tmp_path: Path,
) -> None:
    """Full input returns unchanged while storage keeps independent byte bounds."""
    config = _configuration(tmp_path / "recordings")
    assert len(json.dumps(config).encode()) > 4096
    root = tmp_path / "jobs"
    ledger = StudioJobLedger(root=root)
    try:
        _create(ledger, config)
        row = ledger.connection().execute("SELECT * FROM jobs WHERE job_id='job'").fetchone()
        assert len(row["training_config"].encode()) <= 4096
        assert len(row["training_event_data"].encode()) > 4096
        assert ledger.record("job").training_config == config
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            ledger.connection().execute(
                "UPDATE jobs SET training_event_data='{}' WHERE job_id='job'"
            )
        with pytest.raises(ValueError, match="4096-byte"):
            _create(ledger, {**config, "hidden": [1] * 2000}, "oversized")
        assert len(ledger.list_records()) == 1
    finally:
        ledger.close()
    reopened = StudioJobLedger(root=root)
    try:
        assert reopened.record("job").training_config == config
    finally:
        reopened.close()


def test_altered_event_reference_is_reported_as_corruption(tmp_path: Path) -> None:
    """A valid event blob cannot satisfy an altered digest in its own job row."""
    config = _configuration(tmp_path / "recordings")
    ledger = StudioJobLedger(root=tmp_path / "jobs")
    try:
        _create(ledger, config)
        row = ledger.connection().execute("SELECT training_config FROM jobs").fetchone()
        compact = json.loads(row[0])
        compact["event_data_reference"]["sha256"] = "0" * 64
        ledger.connection().execute(
            "UPDATE jobs SET training_config=? WHERE job_id='job'",
            (json.dumps(compact, sort_keys=True, separators=(",", ":")),),
        )
        with pytest.raises(StudioJobLedgerCorrupt, match="reference"):
            ledger.record("job")
    finally:
        ledger.close()


def test_v7_ledger_migration_preserves_legacy_configuration(tmp_path: Path) -> None:
    """An actual old schema gains separate custody without rewriting old snapshots."""
    root = tmp_path / "jobs"
    root.mkdir()
    old_schema = SCHEMA_V1.replace(
        "    training_event_data TEXT CHECK (training_event_data IS NULL OR length(CAST(training_event_data AS BLOB)) <= 67108864),\n",
        "",
    )
    assert old_schema != SCHEMA_V1
    config = _configuration(tmp_path / "legacy-recordings", count=2)
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"))
    assert len(encoded.encode()) <= 4096
    with sqlite3.connect(root / LEDGER_FILENAME) as connection:
        connection.executescript(old_schema)
        connection.execute("INSERT INTO schema_meta VALUES ('schema_version','7')")
        connection.execute("INSERT INTO schema_meta VALUES ('schema_name','studio.job-ledger.v7')")
        connection.execute(
            "INSERT INTO jobs(job_id,kind,actor,workspace,admission,training_config,"
            "execution_model,status,created_at_utc,artifacts,sequence) "
            "VALUES ('job','training','verification','default','{}',?,'process',"
            "'completed','2026-09-26T00:00:00Z','[]',0)",
            (encoded,),
        )
    ledger = StudioJobLedger(root=root)
    try:
        assert ledger.record("job").training_config == config
        assert (
            ledger.connection().execute("SELECT training_config FROM jobs").fetchone()[0] == encoded
        )
        assert (
            ledger.connection().execute("SELECT training_event_data FROM jobs").fetchone()[0]
            is None
        )
    finally:
        ledger.close()


def test_large_event_checkpoint_resumes_after_application_reopen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real HTTP children retain the complete contract across restart and exact resume."""
    root = tmp_path / "recordings"
    config = _configuration(root)
    monkeypatch.setenv(DATASET_ROOT_ENV, str(root))
    settings = StudioRuntimeSettings(
        job_root_path=str(tmp_path / "jobs"), job_default_timeout_seconds=60
    )
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        job_id, status = _train(client, config)
        assert status["status"] == "completed", status
        response = client.get(f"/api/training/checkpoint/{job_id}")
        assert response.status_code == 200, response.text
        assert response.json()["config"]["event_data"] == config["event_data"]
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        response = client.get(f"/api/training/checkpoint/{job_id}")
        assert response.status_code == 200, response.text
        assert response.json()["config"]["event_data"] == config["event_data"]
        response = client.post(
            "/api/studio/training/weight-restore/attach",
            json={
                "source_job_id": job_id,
                "mode": "exact_resume",
                "config": {**config, "epochs": 2},
            },
        )
        assert response.status_code == 200, response.text
        assert response.json()["mode"] == "exact_resume"
        resumed_id = response.json()["job_id"]
        resumed = _wait_terminal(client, resumed_id)
        assert resumed["status"] == "completed", resumed
        baseline_id, baseline = _train(client, {**config, "epochs": 2})
        assert baseline["status"] == "completed", baseline
        states = []
        for state_job_id in (resumed_id, baseline_id):
            artifact = client.get(
                f"/api/studio/jobs/{state_job_id}/artifacts/training/model_state.pt"
            )
            assert artifact.status_code == 200, artifact.text
            assert (
                hashlib.sha256(artifact.content).hexdigest()
                == artifact.headers["x-studio-artifact-sha256"]
            )
            state = tmp_path / f"{state_job_id}.pt"
            state.write_bytes(artifact.content)
            states.append(state)
        comparator = (
            Path(__file__).resolve().parents[1] / "studio/frontend/e2e/event_training_state.py"
        )
        subprocess.run(
            [sys.executable, str(comparator), str(states[0]), str(states[1])],
            check=True,
            capture_output=True,
            timeout=30,
        )


def test_a_validated_snapshot_is_reused_without_sharing_objects(tmp_path: Path) -> None:
    """Repeated reads of unchanged bytes skip re-validation but hand out fresh values."""
    import time

    from sc_neurocore.studio.platform import jobs_ledger_rows

    config = _configuration(tmp_path / "shd", count=600)
    ledger = StudioJobLedger(root=tmp_path / "jobs")
    try:
        _create(ledger, config)
        row = ledger.connection().execute("SELECT * FROM jobs WHERE job_id='job'").fetchone()
        value, event = row["training_config"], row["training_event_data"]
        jobs_ledger_rows._VALIDATED.clear()
        started = time.perf_counter()
        first = jobs_ledger_rows.training_config_from_json(value, kind="training", event_data=event)
        validating = time.perf_counter() - started
        started = time.perf_counter()
        again = jobs_ledger_rows.training_config_from_json(value, kind="training", event_data=event)
        reusing = time.perf_counter() - started
        assert again == first == config and again is not first
        assert reusing < validating
        first["epochs"] = 99
        event_data = again["event_data"]
        assert isinstance(event_data, dict)
        event_data["train_split"] = "altered"
        third = jobs_ledger_rows.training_config_from_json(value, kind="training", event_data=event)
        assert third == config
    finally:
        ledger.close()


def test_reuse_never_admits_changed_bytes_or_a_changed_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reuse key is the stored bytes and the admission limit, so both still refuse."""
    from sc_neurocore.studio.event_training_budget import EVENT_INPUT_LIMIT_ENV
    from sc_neurocore.studio.platform import jobs_ledger_rows

    config = _configuration(tmp_path / "shd")
    ledger = StudioJobLedger(root=tmp_path / "jobs")
    try:
        _create(ledger, config)
        row = ledger.connection().execute("SELECT * FROM jobs WHERE job_id='job'").fetchone()
        value, event = row["training_config"], row["training_event_data"]
        assert jobs_ledger_rows.training_config_from_json(value, kind="training", event_data=event)
        with pytest.raises(StudioJobLedgerCorrupt):
            jobs_ledger_rows.training_config_from_json(
                value, kind="training", event_data=event.replace('"train"', '"trian"', 1)
            )
        monkeypatch.setenv(EVENT_INPUT_LIMIT_ENV, "1")
        with pytest.raises(StudioJobLedgerCorrupt, match="invalid"):
            jobs_ledger_rows.training_config_from_json(value, kind="training", event_data=event)
    finally:
        ledger.close()


def test_the_reuse_store_is_bounded() -> None:
    """Old entries leave once the store holds its maximum."""
    from sc_neurocore.studio.platform import jobs_ledger_rows

    jobs_ledger_rows._VALIDATED.clear()
    base = resolve_training_config({"hidden": [1]}).to_public_dict()
    for width in range(1, jobs_ledger_rows._VALIDATED_ENTRIES + 6):
        snapshot = json.dumps({**base, "hidden": [width]}, sort_keys=True, separators=(",", ":"))
        jobs_ledger_rows.training_config_from_json(snapshot, kind="training")
    assert len(jobs_ledger_rows._VALIDATED) == jobs_ledger_rows._VALIDATED_ENTRIES
    assert all(
        key[0] != json.dumps({**base, "hidden": [1]}, sort_keys=True, separators=(",", ":"))
        for key in jobs_ledger_rows._VALIDATED
    )
