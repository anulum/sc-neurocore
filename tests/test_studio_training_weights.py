# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio training weight checkpoint tests

"""Tests for Studio Training Monitor weight checkpoint artifacts."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobContext
from sc_neurocore.studio.platform.training_weight_loader import (
    load_training_weight_state_dict,
)
from sc_neurocore.studio.platform.training_weights import (
    STUDIO_TRAINING_TORCH_STATE_DICT_SCHEMA_VERSION,
    STUDIO_TRAINING_WEIGHT_CHECKPOINT_SCHEMA_VERSION,
    STUDIO_TRAINING_WEIGHT_RESTORE_PLAN_SCHEMA_VERSION,
    STUDIO_TRAINING_WEIGHT_RESTORE_SCHEMA_VERSION,
    TRAINING_WEIGHT_ARTIFACT_ROUTE_TEMPLATE,
    TRAINING_WEIGHT_ARTIFACT_PATH,
    TRAINING_WEIGHT_METADATA_ARTIFACT_PATH,
    build_training_weight_restore_evidence,
    build_training_weight_restore_plan,
    materialize_training_weight_payload,
    validate_training_weight_checkpoint_metadata,
    validate_training_weight_restore_evidence,
    write_training_weight_checkpoint,
)


def _context(tmp_path: Path) -> StudioJobContext:
    """Return a confined Studio job context for weight checkpoint tests."""

    return StudioJobContext(
        job_id="sj_weights",
        work_dir=tmp_path,
        cancel_event=threading.Event(),
        max_artifact_bytes=4096,
    )


def test_write_training_weight_checkpoint_publishes_binary_and_metadata(
    tmp_path: Path,
) -> None:
    """Weight checkpoint writer emits path-free manifest metadata."""

    context = _context(tmp_path)
    weights_payload = b"serialized weights"

    summary = write_training_weight_checkpoint(
        context,
        weights_payload=weights_payload,
        config={"dataset": "synthetic", "epochs": 2},
        architecture="64->128->10",
        parameter_count=9610,
        final_metrics={"train_accuracy": 0.75},
    ).to_public_dict()

    assert summary["schema_version"] == STUDIO_TRAINING_WEIGHT_CHECKPOINT_SCHEMA_VERSION
    assert summary["framework"] == "pytorch"
    assert summary["format"] == "torch_state_dict"
    assert summary["parameter_count"] == 9610
    assert summary["weights_artifact"] == {
        "relative_path": TRAINING_WEIGHT_ARTIFACT_PATH,
        "sha256": hashlib.sha256(weights_payload).hexdigest(),
        "size_bytes": len(weights_payload),
    }
    assert [artifact.relative_path for artifact in context.artifacts] == [
        TRAINING_WEIGHT_ARTIFACT_PATH,
        TRAINING_WEIGHT_METADATA_ARTIFACT_PATH,
    ]
    metadata = json.loads((tmp_path / TRAINING_WEIGHT_METADATA_ARTIFACT_PATH).read_text())
    assert metadata["schema_version"] == STUDIO_TRAINING_WEIGHT_CHECKPOINT_SCHEMA_VERSION
    assert metadata["weights_artifact"] == summary["weights_artifact"]


def test_write_training_weight_checkpoint_rejects_invalid_payload(
    tmp_path: Path,
) -> None:
    """Weight checkpoint writer rejects empty or non-portable metadata inputs."""

    context = _context(tmp_path)

    with pytest.raises(ValueError, match="empty"):
        write_training_weight_checkpoint(
            context,
            weights_payload=b"",
            config={"dataset": "synthetic"},
            architecture="64->10",
            parameter_count=1,
            final_metrics=None,
        )
    with pytest.raises(ValueError, match="metrics"):
        write_training_weight_checkpoint(
            context,
            weights_payload=b"weights",
            config={"dataset": "synthetic"},
            architecture="64->10",
            parameter_count=1,
            final_metrics={"bad": float("nan")},
        )


def test_validate_training_weight_checkpoint_metadata_accepts_writer_output(
    tmp_path: Path,
) -> None:
    """Weight checkpoint import validation accepts writer-produced metadata."""

    context = _context(tmp_path)
    summary = write_training_weight_checkpoint(
        context,
        weights_payload=b"weights",
        config={"dataset": "synthetic", "epochs": 2},
        architecture="64->10",
        parameter_count=650,
        final_metrics={"train_accuracy": 0.75},
    ).to_public_dict()

    validated = validate_training_weight_checkpoint_metadata(
        summary,
        expected_config_sha256=str(summary["config_sha256"]),
    )

    assert validated == summary


def test_build_training_weight_restore_plan_returns_digest_bound_route(
    tmp_path: Path,
) -> None:
    """Weight restore plans expose authenticated routes and expected hashes."""

    context = _context(tmp_path)
    summary = write_training_weight_checkpoint(
        context,
        weights_payload=b"weights",
        config={"dataset": "synthetic", "epochs": 2},
        architecture="64->10",
        parameter_count=650,
        final_metrics={"train_accuracy": 0.75},
    ).to_public_dict()

    plan = build_training_weight_restore_plan(
        source_job_id="sj_training",
        source_status="completed",
        weight_checkpoint=summary,
        expected_config_sha256=str(summary["config_sha256"]),
    ).to_public_dict()

    assert plan["schema_version"] == STUDIO_TRAINING_WEIGHT_RESTORE_PLAN_SCHEMA_VERSION
    assert plan["artifact_route_template"] == TRAINING_WEIGHT_ARTIFACT_ROUTE_TEMPLATE
    assert plan["loader_policy"] == "download_from_authenticated_artifact_route_and_verify_sha256"
    assert plan["restore_ready"] is True
    assert plan["source_job_id"] == "sj_training"
    assert plan["source_status"] == "completed"
    assert plan["weights_artifact"] == summary["weights_artifact"]
    assert plan["metadata_artifact"] == summary["metadata_artifact"]


def test_materialize_training_weight_payload_verifies_artifacts_before_loading(
    tmp_path: Path,
) -> None:
    """Trusted weight materialization verifies plan-bound payloads first."""

    context = _context(tmp_path)
    weights_payload = b"weights"
    summary = write_training_weight_checkpoint(
        context,
        weights_payload=weights_payload,
        config={"dataset": "synthetic", "epochs": 2},
        architecture="64->10",
        parameter_count=650,
        final_metrics={"train_accuracy": 0.75},
    ).to_public_dict()
    plan = build_training_weight_restore_plan(
        source_job_id="sj_training",
        source_status="completed",
        weight_checkpoint=summary,
        expected_config_sha256=str(summary["config_sha256"]),
    ).to_public_dict()

    materialization = materialize_training_weight_payload(
        restore_plan=plan,
        metadata_payload=(tmp_path / TRAINING_WEIGHT_METADATA_ARTIFACT_PATH).read_bytes(),
        weights_payload=(tmp_path / TRAINING_WEIGHT_ARTIFACT_PATH).read_bytes(),
        trusted_loader=lambda payload: {"layer.weight": payload},
    )

    public = materialization.to_public_dict()
    assert public["schema_version"] == "studio.training.weight-materialization.v1"
    assert public["source_job_id"] == "sj_training"
    assert public["loaded_key_count"] == 1
    weights_artifact = summary["weights_artifact"]
    metadata_artifact = summary["metadata_artifact"]
    assert isinstance(weights_artifact, dict)
    assert isinstance(metadata_artifact, dict)
    assert public["weights_sha256"] == weights_artifact["sha256"]
    assert public["metadata_sha256"] == metadata_artifact["sha256"]
    assert materialization.state_dict == {"layer.weight": weights_payload}


def test_materialize_training_weight_payload_rejects_tampered_payloads(
    tmp_path: Path,
) -> None:
    """Trusted weight materialization rejects payloads before loader execution."""

    context = _context(tmp_path)
    summary = write_training_weight_checkpoint(
        context,
        weights_payload=b"weights",
        config={"dataset": "synthetic"},
        architecture="64->10",
        parameter_count=650,
        final_metrics=None,
    ).to_public_dict()
    plan = build_training_weight_restore_plan(
        source_job_id="sj_training",
        source_status="completed",
        weight_checkpoint=summary,
    ).to_public_dict()

    with pytest.raises(ValueError, match="digest mismatch"):
        materialize_training_weight_payload(
            restore_plan=plan,
            metadata_payload=(tmp_path / TRAINING_WEIGHT_METADATA_ARTIFACT_PATH).read_bytes(),
            weights_payload=b"tamper!",
            trusted_loader=lambda payload: {"layer.weight": payload},
        )


def test_materialize_training_weight_payload_rejects_invalid_loader_output(
    tmp_path: Path,
) -> None:
    """Trusted weight materialization validates state-dictionary keys."""

    context = _context(tmp_path)
    summary = write_training_weight_checkpoint(
        context,
        weights_payload=b"weights",
        config={"dataset": "synthetic"},
        architecture="64->10",
        parameter_count=650,
        final_metrics=None,
    ).to_public_dict()
    plan = build_training_weight_restore_plan(
        source_job_id="sj_training",
        source_status="completed",
        weight_checkpoint=summary,
    ).to_public_dict()

    with pytest.raises(ValueError, match="state key"):
        materialize_training_weight_payload(
            restore_plan=plan,
            metadata_payload=(tmp_path / TRAINING_WEIGHT_METADATA_ARTIFACT_PATH).read_bytes(),
            weights_payload=(tmp_path / TRAINING_WEIGHT_ARTIFACT_PATH).read_bytes(),
            trusted_loader=lambda payload: {"": payload},
        )


def test_validate_training_weight_checkpoint_metadata_rejects_forged_artifacts(
    tmp_path: Path,
) -> None:
    """Weight checkpoint import validation rejects forged artifact metadata."""

    context = _context(tmp_path)
    summary = write_training_weight_checkpoint(
        context,
        weights_payload=b"weights",
        config={"dataset": "synthetic"},
        architecture="64->10",
        parameter_count=650,
        final_metrics=None,
    ).to_public_dict()
    forged_path = dict(summary)
    forged_path["weights_artifact"] = {
        "relative_path": "../model_state.pt",
        "sha256": "0" * 64,
        "size_bytes": 7,
    }
    forged_digest = dict(summary)
    forged_digest["config_sha256"] = "1" * 64

    with pytest.raises(ValueError, match="path"):
        validate_training_weight_checkpoint_metadata(forged_path)
    with pytest.raises(ValueError, match="config digest mismatch"):
        validate_training_weight_checkpoint_metadata(
            forged_digest,
            expected_config_sha256=str(summary["config_sha256"]),
        )


def test_build_training_weight_restore_plan_rejects_missing_source_metadata(
    tmp_path: Path,
) -> None:
    """Weight restore plans require explicit source job metadata."""

    context = _context(tmp_path)
    summary = write_training_weight_checkpoint(
        context,
        weights_payload=b"weights",
        config={"dataset": "synthetic"},
        architecture="64->10",
        parameter_count=650,
        final_metrics=None,
    ).to_public_dict()

    with pytest.raises(ValueError, match="source_job_id"):
        build_training_weight_restore_plan(
            source_job_id="",
            source_status="completed",
            weight_checkpoint=summary,
        )


def _torch_checkpoint_bytes(
    *,
    schema_version: str = STUDIO_TRAINING_TORCH_STATE_DICT_SCHEMA_VERSION,
    state_dict: dict[str, object] | None = None,
    include_state_dict: bool = True,
) -> bytes:
    """Return a portable torch checkpoint payload like the Training Monitor."""

    from io import BytesIO

    torch = pytest.importorskip("torch")
    payload: dict[str, object] = {
        "config": {"dataset": "synthetic", "epochs": 2},
        "final_metrics": {"train_accuracy": 0.75},
        "model_info": {"architecture": "64->10"},
        "schema_version": schema_version,
    }
    if include_state_dict:
        if state_dict is None:
            state_dict = {
                "fc.weight": torch.zeros(2, 3),
                "fc.bias": torch.zeros(2),
            }
        payload["model_state_dict"] = state_dict
    buffer = BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def test_load_training_weight_state_dict_extracts_model_state() -> None:
    """Trusted loader returns only the string-keyed model state dictionary."""

    pytest.importorskip("torch")
    state_dict = load_training_weight_state_dict(_torch_checkpoint_bytes())

    assert sorted(state_dict.keys()) == ["fc.bias", "fc.weight"]


def test_load_training_weight_state_dict_rejects_non_checkpoint() -> None:
    """Trusted loader rejects payloads that are not checkpoint objects."""

    torch = pytest.importorskip("torch")
    from io import BytesIO

    buffer = BytesIO()
    torch.save(torch.zeros(3), buffer)

    with pytest.raises(ValueError, match="not a checkpoint object"):
        load_training_weight_state_dict(buffer.getvalue())


def test_load_training_weight_state_dict_rejects_unsupported_schema() -> None:
    """Trusted loader rejects checkpoints with an unexpected payload schema."""

    pytest.importorskip("torch")
    payload = _torch_checkpoint_bytes(schema_version="studio.training.torch-state-dict.v0")

    with pytest.raises(ValueError, match="schema is unsupported"):
        load_training_weight_state_dict(payload)


def test_load_training_weight_state_dict_rejects_missing_state_dict() -> None:
    """Trusted loader rejects checkpoints without a model state dictionary."""

    pytest.importorskip("torch")
    payload = _torch_checkpoint_bytes(include_state_dict=False)

    with pytest.raises(ValueError, match="missing a model state dict"):
        load_training_weight_state_dict(payload)


def test_load_training_weight_state_dict_rejects_invalid_state_key() -> None:
    """Trusted loader rejects model state dictionaries with empty keys."""

    torch = pytest.importorskip("torch")
    payload = _torch_checkpoint_bytes(state_dict={"": torch.zeros(1)})

    with pytest.raises(ValueError, match="invalid state key"):
        load_training_weight_state_dict(payload)


def test_load_training_weight_state_dict_rejects_undeserializable() -> None:
    """Trusted loader fails closed on payloads torch cannot deserialize."""

    pytest.importorskip("torch")
    with pytest.raises(ValueError, match="could not be deserialized"):
        load_training_weight_state_dict(b"not a torch payload")


def test_materialize_with_real_torch_loader_roundtrips(tmp_path: Path) -> None:
    """End-to-end materialization loads real torch weights after verification."""

    pytest.importorskip("torch")
    context = _context(tmp_path)
    weights_payload = _torch_checkpoint_bytes()
    summary = write_training_weight_checkpoint(
        context,
        weights_payload=weights_payload,
        config={"dataset": "synthetic", "epochs": 2},
        architecture="64->10",
        parameter_count=8,
        final_metrics={"train_accuracy": 0.75},
    ).to_public_dict()
    plan = build_training_weight_restore_plan(
        source_job_id="sj_training",
        source_status="completed",
        weight_checkpoint=summary,
        expected_config_sha256=str(summary["config_sha256"]),
    ).to_public_dict()

    materialization = materialize_training_weight_payload(
        restore_plan=plan,
        metadata_payload=(tmp_path / TRAINING_WEIGHT_METADATA_ARTIFACT_PATH).read_bytes(),
        weights_payload=(tmp_path / TRAINING_WEIGHT_ARTIFACT_PATH).read_bytes(),
        trusted_loader=load_training_weight_state_dict,
    )

    assert materialization.to_public_dict()["loaded_key_count"] == 2
    assert sorted(materialization.state_dict.keys()) == ["fc.bias", "fc.weight"]


def _materialization(tmp_path: Path):
    """Return a verified materialization for restore-evidence tests."""

    context = _context(tmp_path)
    summary = write_training_weight_checkpoint(
        context,
        weights_payload=b"weights",
        config={"dataset": "synthetic", "epochs": 2},
        architecture="64->10",
        parameter_count=650,
        final_metrics={"train_accuracy": 0.75},
    ).to_public_dict()
    plan = build_training_weight_restore_plan(
        source_job_id="sj_training",
        source_status="completed",
        weight_checkpoint=summary,
        expected_config_sha256=str(summary["config_sha256"]),
    ).to_public_dict()
    return materialize_training_weight_payload(
        restore_plan=plan,
        metadata_payload=(tmp_path / TRAINING_WEIGHT_METADATA_ARTIFACT_PATH).read_bytes(),
        weights_payload=(tmp_path / TRAINING_WEIGHT_ARTIFACT_PATH).read_bytes(),
        trusted_loader=lambda payload: {"layer.weight": payload},
    )


def test_build_training_weight_restore_evidence_wraps_materialization(
    tmp_path: Path,
) -> None:
    """Restore evidence wraps a verified materialization as training evidence."""

    materialization = _materialization(tmp_path)

    evidence = build_training_weight_restore_evidence(
        materialization,
        source_status="completed",
    )

    assert evidence["schema_version"] == STUDIO_TRAINING_WEIGHT_RESTORE_SCHEMA_VERSION
    assert evidence["evidence_classification"] == "training"
    assert evidence["status"] == "completed"
    assert evidence["source_job_id"] == "sj_training"
    assert evidence["source_status"] == "completed"
    materialization_summary = evidence["materialization"]
    assert isinstance(materialization_summary, dict)
    assert "state_dict" not in materialization_summary
    assert validate_training_weight_restore_evidence(evidence) == evidence


def test_build_training_weight_restore_evidence_requires_source_status(
    tmp_path: Path,
) -> None:
    """Restore evidence requires a non-empty source training status."""

    materialization = _materialization(tmp_path)

    with pytest.raises(ValueError, match="source_status"):
        build_training_weight_restore_evidence(materialization, source_status="")


def test_validate_training_weight_restore_evidence_rejects_forged(
    tmp_path: Path,
) -> None:
    """Restore evidence validation rejects forged or incomplete payloads."""

    materialization = _materialization(tmp_path)
    evidence = build_training_weight_restore_evidence(
        materialization,
        source_status="completed",
    )

    bad_schema = dict(evidence)
    bad_schema["schema_version"] = "studio.training.weight-restore.v0"
    with pytest.raises(ValueError, match="schema is unsupported"):
        validate_training_weight_restore_evidence(bad_schema)

    bad_class = dict(evidence)
    bad_class["evidence_classification"] = "analysis"
    with pytest.raises(ValueError, match="classification is invalid"):
        validate_training_weight_restore_evidence(bad_class)

    bad_status = dict(evidence)
    bad_status["status"] = "failed"
    with pytest.raises(ValueError, match="must be completed"):
        validate_training_weight_restore_evidence(bad_status)

    missing_materialization = dict(evidence)
    del missing_materialization["materialization"]
    with pytest.raises(ValueError, match="requires materialization"):
        validate_training_weight_restore_evidence(missing_materialization)

    forged_digest = dict(evidence)
    forged_summary = dict(materialization.to_public_dict())
    forged_summary["weights_sha256"] = "z" * 64
    forged_digest["materialization"] = forged_summary
    with pytest.raises(ValueError, match="weights_sha256 is invalid"):
        validate_training_weight_restore_evidence(forged_digest)
