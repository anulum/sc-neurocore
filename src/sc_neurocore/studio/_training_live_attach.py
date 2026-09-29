# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Live weight attach into a running training job

"""Verify and load weights attached to a running job between epochs.

A command arrives through the job's control channel. It is applied only when
its restore plan, architecture fingerprint and both seed payloads verify and
the weights load strictly into the running model; every other outcome is an
``attach_rejected`` event and the run continues unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from sc_neurocore.studio.platform.evidence_bundle import JsonValue
from sc_neurocore.studio.platform.jobs import StudioJobArtifactUnavailable, StudioJobContext

Emit = Callable[[str, dict[str, Any]], None]


def poll_live_attach(
    context: StudioJobContext, model: Any, epoch: int, *, job_id: str, emit: Emit
) -> dict[str, JsonValue] | None:
    """Consume one pending control command and apply it when it is an attach.

    Parameters
    ----------
    context : StudioJobContext
        Running job whose control channel is polled.
    model : torch.nn.Module
        The model being trained.
    epoch : int
        Epoch boundary at which the command is considered.
    job_id : str
        The running job, recorded in the attach evidence.
    emit : callable
        The job's event emitter.

    Returns
    -------
    dict or None
        The written attach evidence, or None when nothing was attached.
    """
    try:
        command = context.poll_control_command()
    except ValueError:
        emit("attach_rejected", {"epoch": epoch, "reason": "invalid_command"})
        return None
    if command is None or command.get("action") != "attach_weights":
        return None
    return apply_live_attach(context, model, command, epoch, job_id=job_id, emit=emit)


def apply_live_attach(
    context: StudioJobContext,
    model: Any,
    command: Mapping[str, object],
    epoch: int,
    *,
    job_id: str,
    emit: Emit,
) -> dict[str, JsonValue] | None:
    """Verify and load a live weight attach, rejecting on any failure.

    Parameters
    ----------
    context : StudioJobContext
        Running job holding the control seeds.
    model : torch.nn.Module
        The model being trained; loaded strictly.
    command : mapping
        The attach command with its restore plan, fingerprint and seed paths.
    epoch : int
        Epoch boundary at which the attach happens.
    job_id : str
        The running job, recorded in the attach evidence.
    emit : callable
        The job's event emitter.

    Returns
    -------
    dict or None
        The written attach evidence, or None when the attach was rejected.
    """
    from sc_neurocore.studio.platform.training_weight_loader import (
        load_training_weight_state_dict,
    )
    from sc_neurocore.studio.platform.training_weights import (
        TRAINING_WEIGHT_RESTORE_ATTACH_EVIDENCE_ARTIFACT_PATH,
        build_training_weight_restore_attach_evidence,
        materialize_training_weight_payload,
    )

    restore_plan = command.get("restore_plan")
    fingerprint = command.get("architecture_fingerprint")
    weights_seed = command.get("weights_seed_path")
    metadata_seed = command.get("metadata_seed_path")
    if (
        not isinstance(restore_plan, Mapping)
        or not isinstance(fingerprint, str)
        or not isinstance(weights_seed, str)
        or not isinstance(metadata_seed, str)
    ):
        emit("attach_rejected", {"epoch": epoch, "reason": "invalid_command"})
        return None
    try:
        metadata_payload = context.read_control_seed(metadata_seed)
        weights_payload = context.read_control_seed(weights_seed)
        materialization = materialize_training_weight_payload(
            restore_plan=restore_plan,
            metadata_payload=metadata_payload,
            weights_payload=weights_payload,
            trusted_loader=load_training_weight_state_dict,
        )
        model.load_state_dict(dict(materialization.state_dict), strict=True)
    except (RuntimeError, KeyError, ValueError, StudioJobArtifactUnavailable):
        emit("attach_rejected", {"epoch": epoch, "reason": "incompatible"})
        return None
    evidence = build_training_weight_restore_attach_evidence(
        materialization,
        mode="live",
        target_job_id=job_id,
        target_architecture=materialization.architecture,
        target_parameter_count=materialization.parameter_count,
        architecture_fingerprint=fingerprint,
    )
    context.write_artifact(
        TRAINING_WEIGHT_RESTORE_ATTACH_EVIDENCE_ARTIFACT_PATH,
        json.dumps(evidence, sort_keys=True),
    )
    emit(
        "attach",
        {"epoch": epoch, "mode": "live", "loaded_key_count": len(materialization.state_dict)},
    )
    return evidence
