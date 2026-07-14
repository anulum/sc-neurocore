# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Historical Studio training compatibility facade

"""Expose the stable Studio Training Monitor service boundary.

Training execution, parent-process control, event codecs, and weight-attach
orchestration live in focused private modules. This facade preserves historical
imports, callable signatures, qualified names, and process-task integration.
"""

from __future__ import annotations

from typing import Any

from sc_neurocore.studio._training_attach import (
    _request_live_training_weight_attach,
    _start_training_attach,
)
from sc_neurocore.studio._training_control import (
    _export_training_checkpoint,
    _get_training_status,
    _import_training_checkpoint,
    _list_jobs,
    _register_job as _register_job,
    _start_training,
    _stop_training,
    _stream_metrics,
)
from sc_neurocore.studio._training_events import TRAINING_EVENT_LOG_ARTIFACT_PATH
from sc_neurocore.studio._training_job import (
    HAS_TORCH,
    _CELL_TYPES as _CELL_TYPES,
    _SURROGATES as _SURROGATES,
    TrainingJob,
)
from sc_neurocore.studio.platform.jobs import StudioJobManager

TrainingJob.__module__ = __name__

__all__ = [
    "HAS_TORCH",
    "TRAINING_EVENT_LOG_ARTIFACT_PATH",
    "TrainingJob",
    "export_training_checkpoint",
    "get_training_status",
    "import_training_checkpoint",
    "list_cell_types",
    "list_jobs",
    "list_surrogates",
    "request_live_training_weight_attach",
    "start_training",
    "start_training_attach",
    "stop_training",
    "stream_metrics",
]


def list_surrogates() -> list[dict[str, Any]]:
    """Return surrogate-gradient choices exposed by the Studio UI.

    Returns
    -------
    list[dict[str, Any]]
        Ordered names with an ``available`` flag reflecting the installed Torch
        training backend.
    """
    return [{"name": name, "available": HAS_TORCH} for name in _SURROGATES]


def list_cell_types() -> list[dict[str, Any]]:
    """Return training cell types exposed by the Studio UI.

    Returns
    -------
    list[dict[str, Any]]
        Ordered cell names with an ``available`` flag reflecting the installed
        Torch training backend.
    """
    return [{"name": name, "available": HAS_TORCH} for name in _CELL_TYPES]


def start_training(
    config: dict[str, Any],
    job_manager: StudioJobManager | None = None,
) -> dict[str, Any]:
    """Start a Studio training job.

    When ``job_manager`` is supplied, execution is delegated to the bounded
    Studio process sandbox. Without it, the historical in-process training
    thread is retained for direct callers.

    Parameters
    ----------
    config : dict[str, Any]
        Training Monitor configuration.
    job_manager : StudioJobManager or None, optional
        Bounded job manager used by the Studio HTTP route.

    Returns
    -------
    dict[str, Any]
        Path-free job identifier and initial ``running`` status.
    """
    return _start_training(config, job_manager)


def start_training_attach(
    source_job_id: str,
    config: dict[str, Any],
    job_manager: StudioJobManager,
    *,
    expected_config_sha256: str | None = None,
) -> dict[str, Any]:
    """Start a warm-start training job seeded with restored, verified weights.

    The source checkpoint and binary artifacts are verified before a bounded
    process job loads them at the epoch-zero boundary. Raw tensors remain inside
    the confined worker and never enter the API response.

    Parameters
    ----------
    source_job_id : str
        Completed source training job that published model weights.
    config : dict[str, Any]
        Target training configuration.
    job_manager : StudioJobManager
        Bounded manager owning artifact reads and process submission.
    expected_config_sha256 : str or None, optional
        Optional digest that the source configuration must match.

    Returns
    -------
    dict[str, Any]
        Warm-start job metadata, or a stable ``error`` code when a source
        precondition is unavailable.

    Raises
    ------
    ValueError
        If source checkpoint metadata or its expected digest is invalid.
    """
    return _start_training_attach(
        source_job_id,
        config,
        job_manager,
        expected_config_sha256=expected_config_sha256,
    )


def request_live_training_weight_attach(
    target_job_id: str,
    source_job_id: str,
    job_manager: StudioJobManager,
    *,
    expected_config_sha256: str | None = None,
) -> dict[str, Any]:
    """Deliver verified weights to a running training job.

    The command is confined to the worker control channel and applied at the
    next epoch boundary. Incompatible artifacts are rejected without stopping
    the target job.

    Parameters
    ----------
    target_job_id : str
        Running target training job.
    source_job_id : str
        Completed source training job that published model weights.
    job_manager : StudioJobManager
        Manager owning artifact reads and control-command delivery.
    expected_config_sha256 : str or None, optional
        Optional digest that the source configuration must match.

    Returns
    -------
    dict[str, Any]
        Attach-request metadata, or a stable ``error`` code for a failed
        precondition.

    Raises
    ------
    ValueError
        If source checkpoint metadata or its expected digest is invalid.
    """
    return _request_live_training_weight_attach(
        target_job_id,
        source_job_id,
        job_manager,
        expected_config_sha256=expected_config_sha256,
    )


def stop_training(
    job_id: str,
    job_manager: StudioJobManager | None = None,
) -> dict[str, Any]:
    """Request cooperative stop for a Studio training job.

    Parameters
    ----------
    job_id : str
        Training Monitor job identifier.
    job_manager : StudioJobManager or None, optional
        Manager used to propagate cancellation into a process worker.

    Returns
    -------
    dict[str, Any]
        ``stopping`` metadata, or an error payload for an unknown job.
    """
    return _stop_training(job_id, job_manager)


def get_training_status(
    job_id: str,
    job_manager: StudioJobManager | None = None,
) -> dict[str, Any]:
    """Return path-free status for one Studio training job.

    Parameters
    ----------
    job_id : str
        Training Monitor job identifier.
    job_manager : StudioJobManager or None, optional
        Manager used to reconcile process state and verified evidence.

    Returns
    -------
    dict[str, Any]
        Current training status, metrics, checkpoint metadata, and optional
        evidence summary; unknown jobs return an error payload.
    """
    return _get_training_status(job_id, job_manager)


def stream_metrics(job_id: str, job_manager: StudioJobManager | None = None) -> Any:
    """Yield Server-Sent Events for one Studio training job.

    Parameters
    ----------
    job_id : str
        Training Monitor job identifier.
    job_manager : StudioJobManager or None, optional
        Manager used to tail process-worker JSONL events.

    Yields
    ------
    str
        One SSE-formatted metric, heartbeat, terminal, or error event.
    """
    yield from _stream_metrics(job_id, job_manager)


def list_jobs() -> list[dict[str, Any]]:
    """Return path-free summaries for known Studio training jobs.

    Returns
    -------
    list[dict[str, Any]]
        Registry-order job identifiers, statuses, and configurations.
    """
    return _list_jobs()


def export_training_checkpoint(
    job_id: str,
    job_manager: StudioJobManager | None = None,
) -> dict[str, Any]:
    """Return a portable checkpoint for one Studio training job.

    Parameters
    ----------
    job_id : str
        Training Monitor job identifier.
    job_manager : StudioJobManager or None, optional
        Manager used to attach verified terminal worker evidence.

    Returns
    -------
    dict[str, Any]
        ``studio.training.checkpoint.v1`` payload, or an error payload for an
        unknown job.
    """
    return _export_training_checkpoint(job_id, job_manager)


def import_training_checkpoint(data: dict[str, Any]) -> dict[str, Any]:
    """Validate a portable checkpoint and return its training configuration.

    Parameters
    ----------
    data : dict[str, Any]
        JSON object submitted to ``/api/training/checkpoint/import``.

    Returns
    -------
    dict[str, Any]
        Validated source metadata, configuration, and optional weight-restore
        plan.

    Raises
    ------
    ValueError
        If the checkpoint schema or any protected digest is invalid.
    """
    return _import_training_checkpoint(data)
