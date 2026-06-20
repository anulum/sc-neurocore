# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio training process task

"""Importable process task for the Studio Training Monitor."""

from __future__ import annotations

from collections.abc import Mapping

from sc_neurocore.studio.platform.jobs import StudioJobContext
from sc_neurocore.studio.training import TRAINING_EVENT_LOG_ARTIFACT_PATH, TrainingJob

TRAINING_PROCESS_TASK = (
    "sc_neurocore.studio.platform.training_process:run_training_process_task"
)


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


def _training_config_from_payload(payload: Mapping[str, object]) -> dict[str, object]:
    """Return a mutable training configuration copied from worker payload."""

    return dict(payload)


__all__ = [
    "TRAINING_PROCESS_TASK",
    "run_training_process_task",
]
