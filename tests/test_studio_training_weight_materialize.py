# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio training weight materialisation contracts

"""Materialise/load state-dict verification and tamper rejection contracts."""

from __future__ import annotations

from tests.studio_training_weights_support import *  # noqa: F403


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
