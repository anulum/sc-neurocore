# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Event training through isolated storage and launcher

"""Train actual recordings with operator settings through real worker generations."""

from collections.abc import Iterator
import json
import hashlib
from pathlib import Path

import pytest

from sc_neurocore.studio.event_training_data import DATASET_ROOT_ENV
from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.storage_isolated_jobs import IsolatedJobManager
from sc_neurocore.studio.platform.training_process import TRAINING_PROCESS_TASK
from tests.studio_storage_generation_runs import authority as authority
from tests.studio_storage_generation_runs import api_runtime
from tests.studio_storage_generation_runs import ledger as ledger
from tests.studio_storage_launcher_support import Launcher, launcher_base, shutdown, start
from tests.studio_storage_generation_support import Authority, FRAME
from tests.test_studio_storage_isolated_jobs import _configuration as boundary
from tests.test_studio_storage_isolated_jobs import manager as manager
from tests.test_studio_storage_isolated_jobs import request_on
from tests.test_studio_training_config_storage import _configuration


@pytest.fixture
def base() -> Iterator[Path]:
    """Keep real Unix sockets within the platform's path-length bound."""
    with launcher_base() as root:
        yield root


@pytest.fixture(params=[120, 1000])
def event_config(tmp_path: Path, request: pytest.FixtureRequest) -> dict[str, object]:
    """Build a full declaration from actual SHD HDF5 recordings."""
    return _configuration(tmp_path / "recordings", count=int(request.param))


@pytest.fixture
def launcher(
    base: Path,
    tmp_path: Path,
    event_config: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Launcher]:
    """Give the launcher an explicit root and hostile inherited alternatives."""
    with monkeypatch.context() as inherited:
        inherited.setenv(DATASET_ROOT_ENV, str(tmp_path / "wrong-recordings"))
        inherited.setenv("SC_NEUROCORE_STUDIO_EVENT_INPUT_MAX_BYTES", "1")
        inherited.setenv("SC_NEUROCORE_DATASET_GO_LIBRARY", str(tmp_path / "absent.so"))
        running = start(base, event_input={"dataset_root": str(tmp_path / "recordings")})
    try:
        yield running
    finally:
        shutdown(running)


@pytest.mark.parametrize("lost_finish", [False, True])
def test_operator_event_root_trains_and_preserves_large_contract(
    base: Path,
    authority: Authority,
    manager: IsolatedJobManager,
    ledger: StudioJobLedger,
    event_config: dict[str, object],
    lost_finish: bool,
) -> None:
    """Worker uses the trusted root and budget, retaining full input in read views."""
    assert len(json.dumps(event_config).encode()) > 4096
    if lost_finish:
        authority.lose["finish"] = 1
    with request_on("/api/training/start"):
        submitted = manager.submit_process_task(
            kind="training",
            owner="studio-training",
            request_id="event-training",
            idempotency_key="event-training-replay",
            task_path=TRAINING_PROCESS_TASK,
            payload=event_config,
            training_config=event_config,
        )
        completed = manager.wait(submitted.job_id, timeout_seconds=120.0)
    assert completed.status == "completed", completed.error
    assert authority.seen.count("finish") == (2 if lost_finish else 1)
    assert completed.training_config == event_config
    assert ledger.record(submitted.job_id) == completed
    assert any(item.relative_path == "training/model_state.pt" for item in completed.artifacts)
    assert any(item.size_bytes > FRAME for item in completed.artifacts)
    with request_on("/api/studio/jobs", "GET"):
        assert manager.record(submitted.job_id) == completed
        assert manager.list_records() == (completed,)
    assert manager.generation_failures == {}
    with request_on("/api/studio/jobs/status", "GET"):
        assert manager.unreaped_workers == ()
    with request_on("/api/studio/jobs/{job_id}/artifacts/{artifact_path:path}", "GET"):
        saved = manager.read_artifact(submitted.job_id, "training/model_state.pt")
    assert hashlib.sha256(saved.payload).hexdigest() == saved.artifact.sha256
    assert len(saved.payload) == saved.artifact.size_bytes
    recovered = IsolatedJobManager(
        api_runtime(base, authority),
        boundary(base),
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=120.0,
    )
    with request_on("/api/studio/jobs", "GET"):
        assert recovered.record(completed.job_id) == completed
        assert recovered.list_records() == (completed,)
    with request_on("/api/training/start"):
        replayed = recovered.submit_process_task(
            kind="training",
            owner="studio-training",
            request_id="event-training",
            idempotency_key="event-training-replay",
            task_path=TRAINING_PROCESS_TASK,
            payload=event_config,
            training_config=event_config,
        )
    assert replayed == completed
    assert ledger.list_records() == (completed,)
    with request_on("/api/studio/jobs/{job_id}/artifacts/{artifact_path:path}", "GET"):
        assert recovered.read_artifact(completed.job_id, "training/model_state.pt") == saved
    assert recovered.generation_failures == {}
