# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Authored validation of real training checkpoints

"""Validate downloaded real checkpoints and evidence through public contracts."""

import hashlib
import json
import threading
from io import BytesIO
from pathlib import Path
from typing import cast

import pytest
import torch
from fastapi import FastAPI
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import StudioJobManager, StudioRuntimeSettings
from sc_neurocore.studio.platform.jobs import StudioJobContext
from sc_neurocore.studio.platform.training_weight_loader import load_training_weight_state_dict
from sc_neurocore.studio.platform.training_weights import (
    TRAINING_WEIGHT_ARTIFACT_PATH,
    TRAINING_WEIGHT_METADATA_ARTIFACT_PATH,
    StudioTrainingWeightMaterialization,
    build_training_weight_restore_attach_evidence,
    build_training_weight_restore_plan,
    materialize_training_weight_payload,
    training_architecture_fingerprint,
    validate_training_weight_restore_attach_evidence,
    write_training_weight_checkpoint,
)
from sc_neurocore.studio.training_refusals import TrainingRefusal

_CONFIG: dict[str, object] = {
    "dataset": "synthetic",
    "epochs": 1,
    "batch_size": 1024,
    "hidden": [4],
    "timesteps": 1,
}
Checkpoint = tuple[dict[str, object], bytes, bytes]


def _object(value: object) -> dict[str, object]:
    """Read a JSON object after checking its actual key types."""
    assert isinstance(value, dict) and all(isinstance(key, str) for key in value)
    return cast(dict[str, object], value)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Checkpoint:
    """Download a real bounded trainer's sealed weights, metadata and plan."""
    root = tmp_path_factory.mktemp("weight-validation-refusals")
    settings = StudioRuntimeSettings(
        job_root_path=str(root / "jobs"),
        audit_log_path=str(root / "audit.jsonl"),
        job_default_timeout_seconds=60.0,
    )
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        manager = cast(FastAPI, client.app).state.studio_job_manager
        assert isinstance(manager, StudioJobManager)
        started = client.post("/api/training/start", json=_CONFIG)
        assert started.status_code == 200, started.text
        job_id = _object(started.json())["job_id"]
        assert isinstance(job_id, str)
        completed = manager.wait(job_id, timeout_seconds=45.0)
        assert completed.status == "completed", completed.public_error
        summary = _object(_object(completed.result)["weight_checkpoint"])
        plan = build_training_weight_restore_plan(
            source_job_id=job_id, source_status=completed.status, weight_checkpoint=summary
        ).to_public_dict()
        metadata = manager.read_artifact(job_id, TRAINING_WEIGHT_METADATA_ARTIFACT_PATH).payload
        weights = manager.read_artifact(job_id, TRAINING_WEIGHT_ARTIFACT_PATH).payload
        return _object(plan), metadata, weights


def _plan(checkpoint: Checkpoint) -> dict[str, object]:
    """Copy the portable downloaded plan without mutating the shared source."""
    return _object(json.loads(json.dumps(checkpoint[0])))


def _materialize(
    checkpoint: Checkpoint,
    *,
    plan: dict[str, object] | None = None,
    metadata: bytes | None = None,
) -> StudioTrainingWeightMaterialization:
    """Use the public verifier and production restricted Torch deserialiser."""
    return materialize_training_weight_payload(
        restore_plan=_plan(checkpoint) if plan is None else plan,
        metadata_payload=checkpoint[1] if metadata is None else metadata,
        weights_payload=checkpoint[2],
        trusted_loader=load_training_weight_state_dict,
    )


def _bind_metadata(plan: dict[str, object], payload: bytes) -> None:
    """Bind the actual supplied bytes to the caller's copied metadata manifest."""
    artifact = _object(plan["metadata_artifact"])
    artifact["size_bytes"] = len(payload)
    artifact["sha256"] = hashlib.sha256(payload).hexdigest()


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("schema_version", "unsupported", "restore plan schema is unsupported"),
        ("loader_policy", "unsafe", "restore plan loader policy is unsupported"),
        ("artifact_route_template", "/untrusted", "restore plan route template is unsupported"),
        ("source_job_id", "", "restore plan requires source_job_id"),
        ("config_sha256", "caller-text-xyz", "restore plan config digest is invalid"),
        ("parameter_count", -1, "restore plan parameter count is invalid"),
        ("framework", "unsupported", "restore plan framework is unsupported"),
        ("format", "unsupported", "restore plan format is unsupported"),
    ],
)
def test_downloaded_restore_plan_refusals_are_authored(
    checkpoint: Checkpoint, field: str, value: object, reason: str
) -> None:
    """Invalid imported declarations fail before real weight deserialisation."""
    plan = _plan(checkpoint)
    plan[field] = value
    with pytest.raises(TrainingRefusal, match=reason):
        _materialize(checkpoint, plan=plan)


@pytest.mark.parametrize("field", ("metadata_artifact", "weights_artifact"))
@pytest.mark.parametrize(
    ("attribute", "value", "reason"),
    [
        ("size_bytes", 0, "size is invalid"),
        ("sha256", "caller-text-xyz", "digest is invalid"),
        ("size_bytes", 1, "size mismatch"),
    ],
)
def test_downloaded_artifact_contracts_refuse_before_loading(
    checkpoint: Checkpoint, field: str, attribute: str, value: object, reason: str
) -> None:
    """Manifest validation and actual byte mismatches retain authored provenance."""
    plan = _plan(checkpoint)
    _object(plan[field])[attribute] = value
    with pytest.raises(TrainingRefusal, match=reason):
        _materialize(checkpoint, plan=plan)


@pytest.mark.parametrize("field", ("metadata_artifact", "weights_artifact"))
def test_invalid_artifact_shape_is_a_public_plan_refusal(
    checkpoint: Checkpoint, field: str
) -> None:
    """Invalid caller data is rejected before internal validated-artifact checks."""
    plan = _plan(checkpoint)
    plan[field] = []
    with pytest.raises(TrainingRefusal, match=f"checkpoint requires {field}"):
        _materialize(checkpoint, plan=plan)


@pytest.mark.parametrize("payload", (b"\xff", b"{", b"[]"))
def test_digest_valid_invalid_metadata_has_a_fixed_refusal(
    checkpoint: Checkpoint, payload: bytes
) -> None:
    """Real JSON and UTF-8 faults never become generated exception text."""
    plan = _plan(checkpoint)
    _bind_metadata(plan, payload)
    with pytest.raises(TrainingRefusal, match="metadata payload") as caught:
        _materialize(checkpoint, plan=plan, metadata=payload)
    assert "UnicodeDecodeError" not in str(caught.value)
    assert "Expecting" not in str(caught.value)


def test_metadata_cannot_replace_the_authenticated_plan_artifact(
    checkpoint: Checkpoint,
) -> None:
    """A metadata field cannot override the plan that verified those same bytes."""
    plan = _plan(checkpoint)
    metadata = _object(json.loads(checkpoint[1]))
    metadata["metadata_artifact"] = {"untrusted": "caller-text-xyz"}
    payload = json.dumps(metadata).encode()
    _bind_metadata(plan, payload)
    restored = _materialize(checkpoint, plan=plan, metadata=payload)
    assert restored.metadata_sha256 == hashlib.sha256(payload).hexdigest()
    assert restored.weights_sha256 == hashlib.sha256(checkpoint[2]).hexdigest()
    assert restored.state_dict.keys() == load_training_weight_state_dict(checkpoint[2]).keys()


def test_metadata_weights_must_match_the_downloaded_plan(checkpoint: Checkpoint) -> None:
    """Digest-valid metadata cannot name different model weights."""
    plan = _plan(checkpoint)
    metadata = _object(json.loads(checkpoint[1]))
    _object(metadata["weights_artifact"])["sha256"] = "0" * 64
    payload = json.dumps(metadata).encode()
    _bind_metadata(plan, payload)
    with pytest.raises(TrainingRefusal, match="weight artifact does not match restore plan"):
        _materialize(checkpoint, plan=plan, metadata=payload)


def test_external_restricted_tensor_reader_cannot_admit_an_empty_state_key(
    checkpoint: Checkpoint,
) -> None:
    """Actual Torch deserialisation is checked even for an external trusted reader."""
    loaded = _object(torch.load(BytesIO(checkpoint[2]), weights_only=True, map_location="cpu"))
    state = _object(loaded["model_state_dict"])
    state[""] = state.pop(next(iter(state)))
    buffer = BytesIO()
    torch.save(loaded, buffer)
    weights = buffer.getvalue()
    plan = _plan(checkpoint)
    artifact = _object(plan["weights_artifact"])
    artifact["size_bytes"] = len(weights)
    artifact["sha256"] = hashlib.sha256(weights).hexdigest()
    metadata = _object(json.loads(checkpoint[1]))
    metadata["weights_artifact"] = dict(artifact)
    payload = json.dumps(metadata).encode()
    _bind_metadata(plan, payload)

    def read_tensors(data: bytes) -> dict[str, object]:
        """Read actual CPU tensors with restricted unpickling through Torch."""
        decoded = _object(torch.load(BytesIO(data), weights_only=True, map_location="cpu"))
        return _object(decoded["model_state_dict"])

    with pytest.raises(TrainingRefusal, match="invalid state key"):
        materialize_training_weight_payload(
            restore_plan=plan,
            metadata_payload=payload,
            weights_payload=weights,
            trusted_loader=read_tensors,
        )


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("evidence_classification", "other", "classification is invalid"),
        ("status", "running", "must be completed"),
        ("target_parameter_count", -1, "parameter count is invalid"),
        ("materialization.schema_version", "unsupported", "materialization schema is unsupported"),
        ("materialization.config_sha256", "bad", "materialization config digest is invalid"),
        ("materialization.loaded_key_count", -1, "materialization key count is invalid"),
    ],
)
def test_real_materialization_evidence_refusals_are_authored(
    checkpoint: Checkpoint, field: str, value: object, reason: str
) -> None:
    """Evidence from actual tensors is validated at the public import boundary."""
    restored = _materialize(checkpoint)
    evidence = _object(
        build_training_weight_restore_attach_evidence(
            restored,
            mode="warm_start",
            target_job_id="sj_imported",
            target_architecture=restored.architecture,
            target_parameter_count=restored.parameter_count,
            architecture_fingerprint=training_architecture_fingerprint(_CONFIG),
        )
    )
    if field.startswith("materialization."):
        _object(evidence["materialization"])[field.split(".")[1]] = value
    else:
        evidence[field] = value
    with pytest.raises(TrainingRefusal, match=reason):
        validate_training_weight_restore_attach_evidence(evidence)


def test_evidence_builder_refuses_negative_target_count(checkpoint: Checkpoint) -> None:
    """A real materialisation cannot legitimise an invalid target declaration."""
    restored = _materialize(checkpoint)
    with pytest.raises(TrainingRefusal, match="parameter count is invalid"):
        build_training_weight_restore_attach_evidence(
            restored,
            mode="warm_start",
            target_job_id="sj_imported",
            target_architecture=restored.architecture,
            target_parameter_count=-1,
            architecture_fingerprint=training_architecture_fingerprint(_CONFIG),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("parameter_count", -1),
        ("config", object()),
        ("config", {1: "invalid JSON key"}),
    ],
)
def test_real_checkpoint_writer_refuses_invalid_declarations(
    checkpoint: Checkpoint, tmp_path: Path, field: str, value: object
) -> None:
    """The public writer refuses malformed metadata before publishing weights."""
    count = checkpoint[0]["parameter_count"]
    assert isinstance(count, int)
    context = StudioJobContext(
        job_id="sj_refused_writer",
        work_dir=tmp_path,
        cancel_event=threading.Event(),
        max_artifact_bytes=2 * len(checkpoint[2]) + 4096,
    )
    with pytest.raises(TrainingRefusal):
        write_training_weight_checkpoint(
            context,
            weights_payload=checkpoint[2],
            config={"unportable": value} if field == "config" else _CONFIG,
            architecture=str(checkpoint[0]["architecture"]),
            parameter_count=-1 if field == "parameter_count" else count,
            final_metrics=None,
        )
    assert context.artifacts == ()
    assert list(tmp_path.iterdir()) == []
