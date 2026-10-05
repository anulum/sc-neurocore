# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio QCFS conversion training route

"""A conversion run trains a QCFS ANN and is judged on the network it converts to.

The HTTP cases run real jobs through the public Studio surface and then
rebuild the source network from the sealed weights and the validation split
from the seed, so the sealed report is checked against objects it claims to
describe rather than against itself.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import StudioJobManager, StudioRuntimeSettings
from sc_neurocore.studio.training_refusals import TrainingRefusal
from sc_neurocore.studio.training_contract import (
    TrainingConfigError,
    resolve_training_config,
)

torch = pytest.importorskip("torch")

#: Source-owned refusal of a conversion run whose validation split is empty.
EMPTY_VALIDATION_REFUSAL = (
    "The validation split served no samples at this batch size; a conversion "
    "cannot be judged on none. Choose a smaller batch size."
)

_RUN = {
    "model_kind": "qcfs_conversion",
    "dataset": "synthetic",
    "epochs": 2,
    "batch_size": 32,
    "timesteps": 4,
    "hidden": [16],
    "lr": 0.01,
    "seed": 11,
}


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> TestClient:
    """Return a Studio client backed by its own durable job root."""
    root = tmp_path_factory.mktemp("conversion-studio")
    settings = StudioRuntimeSettings(
        job_root_path=str(root / "jobs"),
        audit_log_path=str(root / "audit" / "studio.jsonl"),
        job_default_timeout_seconds=300.0,
    )
    return TestClient(create_app(settings), base_url="http://127.0.0.1")


def _finish(client: TestClient, job_id: str) -> dict[str, Any]:
    """Poll a job's public status until it ends."""
    deadline = time.monotonic() + 300.0
    status: dict[str, Any] = {}
    while time.monotonic() < deadline:
        status = client.get(f"/api/training/status/{job_id}").json()
        if status.get("status") in {"completed", "failed", "stopped"}:
            return status
        time.sleep(0.2)
    pytest.fail(f"job {job_id} did not finish: {status}")


def _artifact(client: TestClient, job_id: str, path: str) -> bytes:
    """Download one sealed job artifact."""
    response = client.get(f"/api/studio/jobs/{job_id}/artifacts/{path}")
    assert response.status_code == 200, response.text
    return response.content


class TestContract:
    """A conversion request resolves to one recorded contract or is refused by field."""

    def test_a_conversion_request_records_its_kind_and_not_cell_settings(self) -> None:
        """The resolved conversion request names its kind and omits spiking-cell settings."""
        resolved = resolve_training_config(_RUN)
        assert resolved.model_kind == "qcfs_conversion"
        public = resolved.to_public_dict()
        assert public["model_kind"] == "qcfs_conversion"
        assert not {"surrogate", "learn_beta", "learn_threshold"} & set(public)
        assert resolve_training_config(public) == resolved

    def test_a_spiking_request_records_what_it_always_did(self) -> None:
        """An explicit spiking request has the same public form as the default one."""
        explicit = resolve_training_config({"model_kind": "spiking"}).to_public_dict()
        assert explicit == resolve_training_config({}).to_public_dict()
        assert "model_kind" not in explicit

    @pytest.mark.parametrize(
        "change,field",
        [
            ({"model_kind": "ann"}, "model_kind"),
            ({"dataset": "shd"}, "dataset"),
            ({"surrogate": "atan_surrogate"}, "surrogate"),
            ({"learn_beta": False}, "learn_beta"),
            ({"learn_threshold": True}, "learn_threshold"),
            ({"timesteps": 2**32}, "timesteps"),
            (
                {"preregistration": {"metric": "conversion_accuracy_drop", "threshold": 2}},
                "preregistration",
            ),
        ],
    )
    def test_a_request_the_route_would_not_honour_is_refused(
        self, change: dict[str, Any], field: str
    ) -> None:
        """Each setting the conversion route cannot honour is refused with its field."""
        with pytest.raises(TrainingConfigError) as refusal:
            resolve_training_config({**_RUN, **change})
        assert refusal.value.field == field

    def test_a_target_profile_is_resolved_to_its_registered_name(self) -> None:
        """A target profile is stored under its registered lower-case name."""
        resolved = resolve_training_config({**_RUN, "target_profile": "LOIHI2"})
        assert resolved.target_profile == "loihi2"
        public = resolved.to_public_dict()
        assert public["target_profile"] == "loihi2"
        assert resolve_training_config(public) == resolved
        assert "target_profile" not in resolve_training_config(_RUN).to_public_dict()

    @pytest.mark.parametrize(
        "request_body,reason",
        [
            ({"target_profile": "loihi2"}, "only a qcfs_conversion run"),
            ({**_RUN, "target_profile": "abacus"}, "not a registered hardware profile"),
            ({**_RUN, "target_profile": 7}, "must be a profile name"),
        ],
    )
    def test_a_target_the_run_cannot_calibrate_for_is_refused(
        self, request_body: dict[str, Any], reason: str
    ) -> None:
        """A target outside the conversion route or the registry is refused with a reason."""
        with pytest.raises(TrainingConfigError) as refusal:
            resolve_training_config(request_body)
        assert refusal.value.field == "target_profile" and reason in refusal.value.reason

    def test_the_accuracy_drop_criterion_belongs_to_the_conversion_route(self) -> None:
        """The accuracy-drop criterion is accepted for conversion and refused for spiking."""
        criterion = {"metric": "conversion_accuracy_drop", "threshold": 0.05}
        assert resolve_training_config({**_RUN, "preregistration": criterion}).preregistration
        with pytest.raises(TrainingConfigError) as refusal:
            resolve_training_config({"preregistration": criterion})
        assert refusal.value.field == "preregistration"

    def test_a_conversion_job_never_takes_attached_weights(self) -> None:
        """A conversion job refuses initial weights and resume state alike."""
        from sc_neurocore.studio.training import TrainingJob
        from sc_neurocore.studio.training_resume import TrainingResumeState

        with pytest.raises(TrainingConfigError, match="fresh weights"):
            TrainingJob(_RUN, initial_state_dict={})
        resume = TrainingResumeState(
            schema_version="studio.training-resume.v1",
            epochs_completed=1,
            architecture="64->16->10",
            config={},
            optimiser_state={},
            rng_state={},
            dataset_fingerprint="sha256:unavailable",
        )
        with pytest.raises(TrainingConfigError, match="fresh weights"):
            TrainingJob(_RUN, resume_state=resume)


class TestHttpRoute:
    """Real conversion runs are started, finished and read through the Studio routes."""

    def test_a_finished_run_is_judged_on_its_converted_network(self, client: TestClient) -> None:
        """The final metrics and the sealed report describe the converted network."""
        from sc_neurocore.conversion.loss_report import (
            LOSS_REPORT_SCHEMA_VERSION,
            data_sha256,
            source_sha256,
        )
        from sc_neurocore.conversion.checkpoint_network import build_qcfs_classifier
        from sc_neurocore.studio._training_datasets import _make_synthetic, _seed_everything

        criterion = {"metric": "conversion_accuracy_drop", "threshold": 1.0}
        started = client.post("/api/training/start", json={**_RUN, "preregistration": criterion})
        assert started.status_code == 200, started.text
        job_id = started.json()["job_id"]
        status = _finish(client, job_id)
        assert status["status"] == "completed", status

        report = json.loads(_artifact(client, job_id, "training/conversion_report.json"))
        metrics = status["final_metrics"]
        assert report["schema_version"] == LOSS_REPORT_SCHEMA_VERSION
        assert report["timesteps"] == 4 and report["input_mode"] == "constant"
        assert report["backend"] in {"numpy", "rust", "go", "mojo", "julia"}
        assert metrics["val_accuracy"] == round(report["converted_accuracy"], 4)
        assert metrics["source_val_accuracy"] == round(report["source_accuracy"], 4)
        assert metrics["conversion_accuracy_drop"] == round(report["accuracy_drop"], 4)
        verdict = status["preregistration_verdict"]
        assert verdict["metric"] == "conversion_accuracy_drop"
        assert verdict["observed"] == report["accuracy_drop"] and verdict["passed"] is True

        checkpoint = torch.load(
            io.BytesIO(_artifact(client, job_id, "training/model_state.pt")), weights_only=True
        )
        assert checkpoint["config"]["model_kind"] == "qcfs_conversion"
        assert "resume_state" not in checkpoint
        rebuilt = build_qcfs_classifier(64, (16,), 10, 4)
        rebuilt.load_state_dict(checkpoint["model_state_dict"])
        assert source_sha256(rebuilt) == report["source_sha256"]

        _seed_everything(_RUN["seed"])
        _, test_loader, _, _ = _make_synthetic(_RUN["batch_size"], rates=True)
        inputs = torch.cat([data for data, _ in test_loader]).numpy().astype(np.float64)
        labels = torch.cat([target for _, target in test_loader]).numpy().astype(np.int64)
        assert report["samples"] == len(labels)
        assert data_sha256(inputs, labels) == report["data_sha256"]

    def test_every_listed_target_profile_resolves(self, client: TestClient) -> None:
        """Every hardware profile the route lists is accepted by the contract."""
        from sc_neurocore.compiler.platforms import get_profile, list_profiles

        listed = client.get("/api/training/target-profiles").json()
        assert [entry["name"] for entry in listed] == sorted(p.name for p in list_profiles())
        assert all(get_profile(entry["name"]).name == entry["name"] for entry in listed)
        loihi = next(entry for entry in listed if entry["name"] == "loihi2")
        assert loihi["q_format"] == "Q11.12" and loihi["signed"] is True

    def test_a_named_target_seals_its_calibration(self, client: TestClient) -> None:
        """A run with a target seals a target report bound to its conversion report."""
        started = client.post("/api/training/start", json={**_RUN, "target_profile": "ecp5"})
        assert started.status_code == 200, started.text
        job_id = started.json()["job_id"]
        status = _finish(client, job_id)
        assert status["status"] == "completed", status
        target = json.loads(_artifact(client, job_id, "training/target_report.json"))
        conversion = json.loads(_artifact(client, job_id, "training/conversion_report.json"))
        assert target["profile"]["name"] == "ecp5"
        assert target["converted_sha256"] == conversion["converted_sha256"]
        assert target["data_sha256"] == conversion["data_sha256"]
        assert target["samples"] == conversion["samples"]
        assert target["float_accuracy"] == conversion["converted_accuracy"]
        assert status["final_metrics"]["target_accuracy"] == round(target["quantized_accuracy"], 4)

    def test_a_criterion_on_the_converted_accuracy_can_be_missed(self, client: TestClient) -> None:
        """A criterion the converted accuracy misses is recorded as not passed."""
        criterion = {"metric": "val_accuracy", "threshold": 1.0}
        started = client.post("/api/training/start", json={**_RUN, "preregistration": criterion})
        status = _finish(client, started.json()["job_id"])
        assert status["status"] == "completed"
        assert status["preregistration_verdict"]["passed"] is False

    def test_an_empty_validation_split_fails_with_its_reason(self, client: TestClient) -> None:
        """An empty validation split fails the run and the status says why."""
        started = client.post("/api/training/start", json={**_RUN, "epochs": 1, "batch_size": 200})
        status = _finish(client, started.json()["job_id"])
        # The worker reports the reason as a finite code; the supervisor renders
        # this source-owned text for it, so it reaches the caller whole.
        assert status["status"] == "failed" and status["error"] == EMPTY_VALIDATION_REFUSAL

    def test_a_sandboxed_run_seals_its_report(self, tmp_path: Path) -> None:
        """A sandboxed run writes its conversion and target reports as artefacts."""
        import threading

        from sc_neurocore.studio.platform import StudioJobContext
        from sc_neurocore.studio.training import TrainingJob

        context = StudioJobContext(
            job_id="sj_sealed_report",
            work_dir=tmp_path,
            cancel_event=threading.Event(),
            max_artifact_bytes=1 << 22,
        )
        config = {**_RUN, "epochs": 1, "target_profile": "loihi2"}
        result = TrainingJob(config, job_id=context.job_id).run_blocking(context)
        assert result["training_status"] == "completed"
        report = json.loads((tmp_path / "training" / "conversion_report.json").read_text())
        target = json.loads((tmp_path / "training" / "target_report.json").read_text())
        assert target["compatible"] is True
        assert result["final_metrics"]["target_accuracy"] == round(target["quantized_accuracy"], 4)
        assert result["final_metrics"]["val_accuracy"] == round(report["converted_accuracy"], 4)
        assert "training/conversion_report.json" in {
            artifact.relative_path for artifact in context.artifacts
        }

    def test_an_empty_validation_split_seals_its_reason(self, tmp_path: Path) -> None:
        """The refusal names the empty split and the sealed status repeats it."""
        import threading

        from sc_neurocore.studio.platform import StudioJobContext
        from sc_neurocore.studio.training import TrainingJob

        context = StudioJobContext(
            job_id="sj_empty_validation",
            work_dir=tmp_path,
            cancel_event=threading.Event(),
            max_artifact_bytes=1 << 20,
        )
        job = TrainingJob({**_RUN, "epochs": 1, "batch_size": 200}, job_id=context.job_id)
        with pytest.raises(TrainingRefusal, match="validation split served no samples"):
            job.run_blocking(context)
        sealed = json.loads((tmp_path / "training" / "status.json").read_text())
        assert sealed["status"] == "failed"
        assert sealed["error"] == EMPTY_VALIDATION_REFUSAL
        assert not (tmp_path / "training" / "conversion_report.json").exists()

    def test_a_conversion_run_refuses_a_warm_start(self, client: TestClient) -> None:
        """The weight-restore route refuses a conversion configuration."""
        response = client.post(
            "/api/studio/training/weight-restore/attach",
            json={"source_job_id": "sj_absent", "config": _RUN},
        )
        assert response.status_code == 422
        assert "fresh weights" in response.json()["detail"]

    def test_a_running_conversion_run_refuses_live_weights(self, client: TestClient) -> None:
        """A running conversion job refuses a live weight attach as incompatible."""
        manager = cast(StudioJobManager, cast(Any, client.app).state.studio_job_manager)
        config = resolve_training_config(_RUN).to_public_dict()
        target = manager.submit_process_task(
            kind="training",
            owner="studio-training",
            request_id="req-conversion-target",
            task_path="tests.studio_job_tasks:process_sleep_task",
            payload={"seconds": 4.0, "config": config},
            training_config=config,
        )
        deadline = time.monotonic() + 5.0
        while manager.record(target.job_id).status != "running":
            assert time.monotonic() < deadline, "target never started"
            time.sleep(0.02)
        response = client.post(
            "/api/studio/training/weight-restore/attach/live",
            json={"target_job_id": target.job_id, "source_job_id": target.job_id},
        )
        assert response.status_code == 409
        assert response.json()["detail"] == "architecture_incompatible"


class TestStops:
    """In-process conversion runs end as completed or stopped, never half reported."""

    def _run(self, config: dict[str, Any], cancelled: Any) -> Any:
        from sc_neurocore.studio.training import TrainingJob

        job = TrainingJob(config, cancelled=cancelled)
        job.start()
        deadline = time.monotonic() + 120.0
        while job.status == "running":
            assert time.monotonic() < deadline, "job did not end"
            time.sleep(0.05)
        return job

    def test_an_in_process_run_completes_without_a_sandbox(self) -> None:
        """A run outside the sandbox completes with the full conversion metric set."""
        # A zero clipping norm skips clipping entirely, as on the spiking route.
        job = self._run({**_RUN, "epochs": 1, "max_grad_norm": 0.0}, cancelled=None)
        assert job.status == "completed", job.error
        assert set(job.final_metrics) == {
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "source_val_accuracy",
            "conversion_accuracy_drop",
        }
        assert job.preregistration_verdict is None

    def test_a_stop_during_training_ends_the_run_without_a_report(self) -> None:
        """A stop requested during training ends the run with no final metrics."""
        job = self._run(_RUN, cancelled=lambda: True)
        assert job.status == "stopped" and job.final_metrics is None

    def test_a_stop_after_the_last_epoch_ends_it_before_conversion(self) -> None:
        """A stop at the last batch boundary ends the run before conversion starts."""
        # 409 synthetic training samples in batches of 32 give 12 batch boundaries
        # per epoch; the probe answers yes only at the boundary after the last one.
        calls = {"count": 0}

        def cancelled() -> bool:
            calls["count"] += 1
            return calls["count"] > 12

        job = self._run({**_RUN, "epochs": 1}, cancelled=cancelled)
        assert job.status == "stopped" and job.final_metrics is None
        assert calls["count"] == 13


def test_the_report_artifact_path_is_under_the_training_prefix() -> None:
    """The conversion report is published under the training artefact prefix."""
    from sc_neurocore.studio._training_conversion import CONVERSION_REPORT_ARTIFACT_PATH

    assert Path(CONVERSION_REPORT_ARTIFACT_PATH).parts[0] == "training"
