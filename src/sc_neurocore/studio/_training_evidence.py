# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio training status and evidence sealing

"""Seal how a training job ended, whether it ran or was refused before running.

Every job that ends leaves ``training/status.json`` and an evidence manifest
over it in its sandbox; this module is the one place both are written.
"""

from __future__ import annotations

import json

from sc_neurocore.studio.platform.action_evidence import (
    EvidenceStatus,
    write_studio_action_evidence_manifest,
)
from sc_neurocore.studio.platform.jobs import StudioJobContext
from sc_neurocore.studio.platform.jobs_failures import GENERIC_JOB_FAILURE, public_job_error


def seal_training_status(
    context: StudioJobContext,
    status_payload: dict[str, object],
    *,
    status: EvidenceStatus,
    error_message: str | None,
) -> None:
    """Write the public status artifact and the evidence manifest bound to it.

    Parameters
    ----------
    context : StudioJobContext
        Job sandbox receiving both artifacts.
    status_payload : dict
        The path-free public status.
    status : {'completed', 'failed', 'cancelled'}
        How the job ended, as the evidence manifest records it.
    error_message : str or None
        The failure, when there is one.
    """
    status_artifact = context.write_artifact(
        "training/status.json",
        json.dumps(status_payload, sort_keys=True),
    )
    write_studio_action_evidence_manifest(
        context,
        action_kind="studio.training.run",
        result=status_payload,
        result_artifact=status_artifact,
        evidence_artifact_path="training/evidence.json",
        evidence_classification="training",
        replay_route="POST /api/training/start",
        status=status,
        error_message=error_message,
    )


def write_refused_evidence(context: StudioJobContext, message: str) -> None:
    """Seal failed evidence for a request that was refused before it ran.

    Parameters
    ----------
    context : StudioJobContext
        Job sandbox to write the status and evidence artifacts into.
    message : str
        A qualified job error or legacy text. Only the explicit public projection
        survives; unqualified text uses the fixed job-failure message.

    Notes
    -----
    A refused configuration never builds a model, so there is no weight
    checkpoint and no event log to publish — only the reason. Writing it
    keeps the sandbox's account complete: every job that ends has an
    evidence artifact saying how.
    """
    message = public_job_error(message) or GENERIC_JOB_FAILURE
    seal_training_status(
        context,
        {
            "job_id": context.job_id,
            "status": "failed",
            "error": message,
            "final_metrics": None,
            "weight_checkpoint": None,
        },
        status="failed",
        error_message=message,
    )


__all__ = ["seal_training_status", "write_refused_evidence"]
