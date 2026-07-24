# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio training weight restore evidence contracts

"""Restore/attach evidence builders, fingerprints, and forgery rejection."""

from __future__ import annotations

from tests.studio_training_weights_support import *  # noqa: F403


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


def test_training_architecture_fingerprint_ignores_non_architecture_fields() -> None:
    """The fingerprint covers only fields that change the model state-dict shape."""

    base = {
        "dataset": "synthetic",
        "hidden": [128],
        "learn_beta": False,
        "learn_threshold": False,
    }
    same_arch = {**base, "epochs": 99, "lr": 0.5, "batch_size": 8, "timesteps": 40}

    fingerprint = training_architecture_fingerprint(base)
    assert len(fingerprint) == 64
    assert training_architecture_fingerprint(same_arch) == fingerprint
    assert training_architecture_fingerprint(base) == training_architecture_fingerprint({})


def test_training_architecture_fingerprint_changes_with_architecture() -> None:
    """Architecture-determining changes produce a different fingerprint."""

    base = {"dataset": "synthetic", "hidden": [128]}

    assert training_architecture_fingerprint({**base, "dataset": "mnist"}) != (
        training_architecture_fingerprint(base)
    )
    assert training_architecture_fingerprint({**base, "hidden": [256]}) != (
        training_architecture_fingerprint(base)
    )
    assert training_architecture_fingerprint({**base, "learn_beta": True}) != (
        training_architecture_fingerprint(base)
    )


def test_build_training_weight_restore_attach_evidence_wraps_materialization(
    tmp_path: Path,
) -> None:
    """Attach evidence wraps a verified materialization as training evidence."""

    materialization = _materialization(tmp_path)
    fingerprint = training_architecture_fingerprint({"dataset": "synthetic", "hidden": [10]})

    evidence = build_training_weight_restore_attach_evidence(
        materialization,
        mode="warm_start",
        target_job_id="sj_attach",
        target_architecture="64->10",
        target_parameter_count=650,
        architecture_fingerprint=fingerprint,
    )

    assert evidence["schema_version"] == STUDIO_TRAINING_WEIGHT_RESTORE_ATTACH_SCHEMA_VERSION
    assert evidence["evidence_classification"] == "training"
    assert evidence["status"] == "completed"
    assert evidence["mode"] == "warm_start"
    assert evidence["source_job_id"] == "sj_training"
    assert evidence["target_job_id"] == "sj_attach"
    assert evidence["architecture_fingerprint"] == fingerprint
    assert validate_training_weight_restore_attach_evidence(evidence) == evidence


def test_build_training_weight_restore_attach_evidence_rejects_invalid_inputs(
    tmp_path: Path,
) -> None:
    """Attach evidence construction fails closed on invalid mode or identifiers."""

    materialization = _materialization(tmp_path)
    fingerprint = "a" * 64

    with pytest.raises(ValueError, match="mode is unsupported"):
        build_training_weight_restore_attach_evidence(
            materialization,
            mode="hot_swap",
            target_job_id="sj_attach",
            target_architecture="64->10",
            target_parameter_count=650,
            architecture_fingerprint=fingerprint,
        )
    with pytest.raises(ValueError, match="target_job_id"):
        build_training_weight_restore_attach_evidence(
            materialization,
            mode="warm_start",
            target_job_id="",
            target_architecture="64->10",
            target_parameter_count=650,
            architecture_fingerprint=fingerprint,
        )
    with pytest.raises(ValueError, match="fingerprint is invalid"):
        build_training_weight_restore_attach_evidence(
            materialization,
            mode="warm_start",
            target_job_id="sj_attach",
            target_architecture="64->10",
            target_parameter_count=650,
            architecture_fingerprint="not-a-digest",
        )


def test_validate_training_weight_restore_attach_evidence_rejects_forged(
    tmp_path: Path,
) -> None:
    """Attach evidence validation rejects forged or incomplete payloads."""

    materialization = _materialization(tmp_path)
    evidence = build_training_weight_restore_attach_evidence(
        materialization,
        mode="live",
        target_job_id="sj_attach",
        target_architecture="64->10",
        target_parameter_count=650,
        architecture_fingerprint="b" * 64,
    )

    bad_schema = dict(evidence)
    bad_schema["schema_version"] = "studio.training.weight-restore-attach.v0"
    with pytest.raises(ValueError, match="schema is unsupported"):
        validate_training_weight_restore_attach_evidence(bad_schema)

    bad_mode = dict(evidence)
    bad_mode["mode"] = "hot_swap"
    with pytest.raises(ValueError, match="mode is unsupported"):
        validate_training_weight_restore_attach_evidence(bad_mode)

    bad_fingerprint = dict(evidence)
    bad_fingerprint["architecture_fingerprint"] = "z" * 64
    with pytest.raises(ValueError, match="fingerprint is invalid"):
        validate_training_weight_restore_attach_evidence(bad_fingerprint)

    missing_target = dict(evidence)
    del missing_target["target_job_id"]
    with pytest.raises(ValueError, match="target_job_id"):
        validate_training_weight_restore_attach_evidence(missing_target)

    forged_materialization = dict(evidence)
    summary = dict(materialization.to_public_dict())
    summary["metadata_sha256"] = "z" * 64
    forged_materialization["materialization"] = summary
    with pytest.raises(ValueError, match="metadata_sha256 is invalid"):
        validate_training_weight_restore_attach_evidence(forged_materialization)
