# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio training process task

"""Importable process tasks for the Studio Training Monitor."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

from sc_neurocore.studio.platform.jobs import StudioJobContext
from sc_neurocore.studio.platform.training_weight_loader import (
    load_training_weight_state_dict,
)
from sc_neurocore.studio.platform.training_weights import (
    TRAINING_WEIGHT_RESTORE_ATTACH_EVIDENCE_ARTIFACT_PATH,
    build_training_weight_restore_attach_evidence,
    materialize_training_weight_payload,
)
from sc_neurocore.studio.training import TRAINING_EVENT_LOG_ARTIFACT_PATH, TrainingJob

TRAINING_PROCESS_TASK = "sc_neurocore.studio.platform.training_process:run_training_process_task"
TRAINING_ATTACH_PROCESS_TASK = (
    "sc_neurocore.studio.platform.training_process:run_training_attach_process_task"
)
TRAINING_ATTACH_SEED_WEIGHTS_PATH = "model_state.pt"
TRAINING_ATTACH_SEED_METADATA_PATH = "model_state.json"


def run_training_process_task(
    context: StudioJobContext,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Run one Studio training job in an isolated process.

    Parameters
    ----------
    context:
        Job sandbox used to write path-confined status and evidence artifacts.
    payload:
        JSON object matching the public ``/api/training/start`` configuration
        contract.

    Returns
    -------
    dict[str, object]
        Path-free terminal training metadata.

    Raises
    ------
    ValueError
        If ``payload`` is not a JSON object suitable for a training
        configuration.
    """
    config = _training_config_from_payload(payload)
    job = TrainingJob(
        config,
        job_id=context.job_id,
        cancelled=lambda: context.cancelled,
        event_sink=lambda event: context.append_artifact_event(
            TRAINING_EVENT_LOG_ARTIFACT_PATH,
            event,
        ),
    )
    return job.run_blocking(context)


def run_training_attach_process_task(
    context: StudioJobContext,
    payload: Mapping[str, object],
) -> dict[str, object]:
    """Run a warm-start training job seeded with restored, verified weights.

    The worker reads the confined seed weight and metadata inputs, verifies and
    materializes them through the shared materializer, then runs a training job
    that loads the restored state dictionary at the epoch-zero checkpoint
    boundary before training forward. On success it writes a path-free
    ``studio.training.weight-restore-attach.v1`` evidence artifact recording the
    verified digests, the resolved target architecture, and the architecture
    fingerprint that gated compatibility. A strict load of an incompatible
    architecture fails the job before training begins.

    Parameters
    ----------
    context:
        Job sandbox holding the confined seed inputs and artifact outputs.
    payload:
        JSON object with ``config`` (the target training configuration),
        ``restore_plan`` (the path-free restore plan), and
        ``architecture_fingerprint``.

    Returns
    -------
    dict[str, object]
        Path-free terminal training metadata augmented with the attach evidence.

    Raises
    ------
    ValueError
        If the payload is malformed or the restored weights are incompatible
        with the target architecture.
    """
    config = _training_config_from_payload(_required_mapping(payload, "config"))
    restore_plan = _required_mapping(payload, "restore_plan")
    fingerprint = payload.get("architecture_fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError("Training weight attach payload requires an architecture fingerprint.")

    metadata_payload = context.read_seed_input(TRAINING_ATTACH_SEED_METADATA_PATH)
    weights_payload = context.read_seed_input(TRAINING_ATTACH_SEED_WEIGHTS_PATH)
    materialization = materialize_training_weight_payload(
        restore_plan=restore_plan,
        metadata_payload=metadata_payload,
        weights_payload=weights_payload,
        trusted_loader=load_training_weight_state_dict,
    )

    job = TrainingJob(
        config,
        job_id=context.job_id,
        cancelled=lambda: context.cancelled,
        event_sink=lambda event: context.append_artifact_event(
            TRAINING_EVENT_LOG_ARTIFACT_PATH,
            event,
        ),
        initial_state_dict=materialization.state_dict,
    )
    result = job.run_blocking(context)

    attach_evidence = build_training_weight_restore_attach_evidence(
        materialization,
        mode="warm_start",
        target_job_id=context.job_id,
        target_architecture=materialization.architecture,
        target_parameter_count=materialization.parameter_count,
        architecture_fingerprint=fingerprint,
    )
    context.write_artifact(
        TRAINING_WEIGHT_RESTORE_ATTACH_EVIDENCE_ARTIFACT_PATH,
        json.dumps(attach_evidence, sort_keys=True),
    )
    result["weight_restore_attach"] = cast(object, attach_evidence)
    return result


def _required_mapping(payload: Mapping[str, object], field_name: str) -> dict[str, object]:
    """Return a required JSON object field from a worker payload."""
    value = payload.get(field_name)
    if not isinstance(value, Mapping):
        raise ValueError(f"Training weight attach payload requires {field_name}.")
    return dict(value)


def _training_config_from_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Return a mutable training configuration copied from worker payload."""
    return dict(payload)


__all__ = [
    "TRAINING_ATTACH_PROCESS_TASK",
    "TRAINING_ATTACH_SEED_METADATA_PATH",
    "TRAINING_ATTACH_SEED_WEIGHTS_PATH",
    "TRAINING_PROCESS_TASK",
    "run_training_attach_process_task",
    "run_training_process_task",
]
