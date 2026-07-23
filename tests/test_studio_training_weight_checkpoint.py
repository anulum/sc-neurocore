# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio training weight checkpoint contracts

"""Write/validate checkpoint metadata and restore-plan contracts."""

from __future__ import annotations

from tests.studio_training_weights_support import *  # noqa: F403

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
