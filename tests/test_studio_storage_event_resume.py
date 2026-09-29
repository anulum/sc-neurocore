# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exact event resume through isolated storage

"""Compare full resumed states from actual storage, launcher and worker paths."""

import hashlib
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from sc_neurocore.datasets.encoders import EventBinning
from sc_neurocore.studio.event_training_contract import resolve_event_training_contract
from sc_neurocore.studio.platform.jobs_models import StudioJobRecord
from sc_neurocore.studio.platform.storage_isolated_jobs import IsolatedJobManager
from sc_neurocore.studio.platform.training_process import (
    TRAINING_PROCESS_TASK,
    TRAINING_ATTACH_PROCESS_TASK,
    TRAINING_ATTACH_SEED_METADATA_PATH,
    TRAINING_ATTACH_SEED_WEIGHTS_PATH,
)
from sc_neurocore.studio.platform.training_weights import (
    build_training_weight_restore_plan,
    training_architecture_fingerprint,
)
from tests.studio_storage_generation_runs import api_runtime
from tests.studio_storage_generation_runs import authority as authority
from tests.studio_storage_generation_runs import ledger as ledger
from tests.studio_storage_generation_support import Authority, FRAME
from tests.studio_storage_launcher_support import Launcher
from tests.test_studio_storage_event_training import base as base
from tests.test_studio_storage_event_training import event_config as event_config
from tests.test_studio_storage_event_training import launcher as launcher
from tests.test_studio_storage_isolated_jobs import _configuration, request_on

ATTACH_ROUTE = "/api/studio/training/weight-restore/attach"
ARTIFACT_ROUTE = "/api/studio/jobs/{job_id}/artifacts/{artifact_path:path}"


def _train(manager: IsolatedJobManager, config: dict[str, object]) -> StudioJobRecord:
    """Complete an actual named training generation, refusing any worker error."""
    with request_on("/api/training/start"):
        job = manager.submit_process_task(
            kind="training",
            owner="studio-training",
            request_id="event-resume-baseline",
            task_path=TRAINING_PROCESS_TASK,
            payload=config,
            training_config=config,
        )
        completed = manager.wait(job.job_id, timeout_seconds=120)
    assert completed.status == "completed", completed.error
    return completed


def _checkpoint(manager: IsolatedJobManager, job_id: str) -> bytes:
    """Read the complete sealed checkpoint over the real artifact protocol."""
    with request_on(ARTIFACT_ROUTE, "GET"):
        saved = manager.read_artifact(job_id, "training/model_state.pt")
    assert hashlib.sha256(saved.payload).hexdigest() == saved.artifact.sha256
    return saved.payload


def test_exact_event_resume_retains_model_optimiser_and_rng_through_seed_frames(
    base: Path,
    tmp_path: Path,
    launcher: Launcher,
    authority: Authority,
    event_config: dict[str, object],
) -> None:
    """Actual resumed epochs match an uninterrupted run; changed input is refused."""
    assert launcher.process.poll() is None
    seed_budget = 1 << 20
    authority.services = replace(authority.services, max_seed_bytes=seed_budget)
    runtime = api_runtime(base, authority)
    boundary = _configuration(base).model_copy(update={"max_seed_bytes": seed_budget})
    manager = IsolatedJobManager(
        runtime,
        boundary,
        allowed_kinds=frozenset({"training"}),
        default_timeout_seconds=120,
    )
    source = _train(manager, event_config)
    assert source.result is not None
    checkpoint = source.result["weight_checkpoint"]
    assert isinstance(checkpoint, dict)
    plan = build_training_weight_restore_plan(
        source_job_id=source.job_id,
        source_status=source.status,
        weight_checkpoint=checkpoint,
    )
    continued = {**event_config, "epochs": 2}
    with request_on(ATTACH_ROUTE):
        weights = manager.read_artifact(source.job_id, "training/model_state.pt").payload
        metadata = manager.read_artifact(source.job_id, "training/model_state.json").payload
    assert len(weights) <= seed_budget
    if len(str(event_config)) > FRAME:
        assert len(weights) > FRAME
    seeds = {
        TRAINING_ATTACH_SEED_WEIGHTS_PATH: weights,
        TRAINING_ATTACH_SEED_METADATA_PATH: metadata,
    }

    def resume(
        config: dict[str, object], *, seed_inputs: dict[str, bytes] | None = None
    ) -> StudioJobRecord:
        """Submit integrity-bound seeds through the reviewed exact-resume task."""
        with request_on(ATTACH_ROUTE):
            job = manager.submit_process_task(
                kind="training",
                owner="studio-training-attach",
                request_id="event-exact-resume",
                task_path=TRAINING_ATTACH_PROCESS_TASK,
                payload={
                    "config": config,
                    "restore_plan": plan.to_public_dict(),
                    "architecture_fingerprint": training_architecture_fingerprint(config),
                    "mode": "exact_resume",
                },
                training_config=config,
                seed_inputs=seeds if seed_inputs is None else seed_inputs,
            )
            return manager.wait(job.job_id, timeout_seconds=120)

    resumed = resume(continued)
    assert resumed.status == "completed", resumed.error
    uninterrupted = _train(manager, continued)
    resumed_path = tmp_path / "resumed.pt"
    baseline_path = tmp_path / "uninterrupted.pt"
    resumed_path.write_bytes(_checkpoint(manager, resumed.job_id))
    baseline_path.write_bytes(_checkpoint(manager, uninterrupted.job_id))
    subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[1] / "studio/frontend/e2e/event_training_state.py"
            ),
            str(resumed_path),
            str(baseline_path),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    contract = resolve_event_training_contract(continued["event_data"], dataset="shd", timesteps=4)
    changed = replace(contract, encoder=EventBinning(2.0, 4, 700, 1, "merge"))
    refused = resume({**continued, "event_data": changed.to_dict()})
    assert refused.status == "failed"
    assert not any(item.relative_path == "training/model_state.pt" for item in refused.artifacts)
    corrupted = resume(
        continued,
        seed_inputs={
            **seeds,
            TRAINING_ATTACH_SEED_WEIGHTS_PATH: weights[:-1] + bytes([weights[-1] ^ 1]),
        },
    )
    assert corrupted.status == "failed"
    assert not any(item.relative_path == "training/model_state.pt" for item in corrupted.artifacts)
    assert manager.generation_failures == {}
    with request_on("/api/studio/jobs/status", "GET"):
        assert manager.unreaped_workers == ()
