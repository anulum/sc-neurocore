# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — isolated submission of named process jobs

"""Submit a process job through the storage authority and supervise its generation.

The embedded ``submit_process_task`` signature is kept. The task import path
must be one the reviewed catalogue runs on the route the gateway authorised
for the current request; kind and owner must be that task's. The authority
admits the named submission for the delegated requester (an identical
mutation returns the job it already admitted), and this API generation then
supervises the launched worker in a thread of its own. A job already
supervised here, or no longer pending, gets no second worker; a pending job
another generation owns is refused by the authority at registration.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
import math
import secrets
import socket
import threading

from sc_neurocore.studio.platform.jobs_models import StudioJobRejected, StudioProcessJobPayload
from sc_neurocore.studio.platform.jobs_process_protocol import _json_payload
from sc_neurocore.studio.platform.storage_admission_client import (
    read_named_admission_result,
    send_named_admission_request,
)
from sc_neurocore.studio.platform.storage_admission_protocol import StorageNamedAdmissionRequest
from sc_neurocore.studio.platform.storage_configuration import StorageBoundaryConfiguration
from sc_neurocore.studio.platform.storage_generation_exchanges import (
    GenerationJob,
    GenerationRuntime,
)
from sc_neurocore.studio.platform.storage_generation_supervisor import supervise_generation
from sc_neurocore.studio.platform.storage_named_tasks import named_studio_task_for_path
from sc_neurocore.studio.platform.storage_requester import Delegation


class GenerationThreads:
    """The generations this API supervises, with their cancel and done events."""

    def __init__(self, runtime: GenerationRuntime) -> None:
        """Keep the runtime every supervised generation uses."""
        self._runtime = runtime
        self._lock = threading.Lock()
        self._cancel: dict[str, threading.Event] = {}
        self._done: dict[str, threading.Event] = {}
        self._failures: dict[str, BaseException] = {}

    def failures(self) -> dict[str, BaseException]:
        """Return the generations that ended without a final finish reply."""
        with self._lock:
            return dict(self._failures)

    def events(self, job_id: str) -> tuple[threading.Event, threading.Event] | None:
        """Return the cancel and done events of a supervised job, if any."""
        with self._lock:
            cancel = self._cancel.get(job_id)
            done = self._done.get(job_id)
        return None if cancel is None or done is None else (cancel, done)

    def start(self, job: GenerationJob) -> bool:
        """Supervise ``job`` in a new thread; ``False`` if already supervised."""
        cancel, done = threading.Event(), threading.Event()
        with self._lock:
            if job.job_id in self._done:
                return False
            self._cancel[job.job_id], self._done[job.job_id] = cancel, done

        def run() -> None:
            try:
                supervise_generation(self._runtime, job, cancel=cancel)
            except BaseException as exc:
                # An unanswered finish keeps custody at the authority; the
                # record stays live and the failure is kept for operators.
                with self._lock:
                    self._failures[job.job_id] = exc
            finally:
                done.set()

        threading.Thread(target=run, name=f"studio-generation-{job.job_id}", daemon=True).start()
        return True


def submit_named(
    configuration: StorageBoundaryConfiguration,
    delegation: Delegation,
    *,
    kind: str,
    owner: str,
    request_id: str | None,
    task_path: str,
    payload: StudioProcessJobPayload,
    timeout_seconds: float,
    seed_inputs: Mapping[str, bytes],
    workspace: str | None,
    idempotency_key: str | None,
    experiment_sha256: str | None,
    admission: Mapping[str, object] | None,
    training_config: Mapping[str, object] | None,
    connect: Callable[[], socket.socket],
) -> GenerationJob:
    """Admit one named job for the delegated requester.

    Returns
    -------
    GenerationJob
        The admitted job, ready for :meth:`GenerationThreads.start` once its
        record shows it is still pending.

    Raises
    ------
    StudioJobRejected
        Kind, owner, workspace, task or timeout is not the reviewed contract.
    StudioJobQueueFull
        The authority refused for capacity.
    PermissionError, ValueError, TimeoutError, EOFError, OSError
        The admission exchange failed or was refused; an identical submission
        with the same idempotency key is answered with the same job.
    """
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise StudioJobRejected("Studio job timeout must be finite and positive.")
    if workspace is not None and workspace != configuration.workspace:
        raise StudioJobRejected("Studio isolated jobs run only in the configured workspace.")
    try:
        task = named_studio_task_for_path(task_path, authorized_route=delegation.route)
    except ValueError as exc:
        raise StudioJobRejected(str(exc)) from exc
    if (task.kind, task.owner) != (kind, owner):
        raise StudioJobRejected("Studio job kind or owner differs from the reviewed task.")
    payload_json = _json_payload(payload, "Studio process job payload must be JSON.")
    if training_config is not None:
        # The same snapshot check the embedded submission makes before admission.
        from sc_neurocore.studio.training_contract import resolve_training_config

        resolved = resolve_training_config(training_config).to_public_dict()
        if kind != "training" or payload.get("config", payload) != resolved:
            raise StudioJobRejected(
                "Training configuration snapshot does not match the process payload."
            )
    request = StorageNamedAdmissionRequest(
        schema_version="studio.storage.admission.v1",
        operation="admit_named",
        request_id=request_id,
        mutation_id=idempotency_key or secrets.token_hex(16),
        workspace=configuration.workspace,
        requester=delegation.requester,
        task_name=task.name,
        authorized_route=delegation.route,
        payload=json.loads(payload_json),
        seed_manifest={path: len(data) for path, data in seed_inputs.items()},
        execution_timeout_seconds=timeout_seconds,
        queue_wait_seconds=None,
        admission=None if admission is None else dict(admission),
        training_config=None if training_config is None else dict(training_config),
        experiment_sha256=experiment_sha256,
    )
    channel = connect()
    pending = send_named_admission_request(
        channel, request=request, seed_inputs=seed_inputs, configuration=configuration
    )
    return GenerationJob(
        job_id=read_named_admission_result(channel, pending=pending),
        task_name=task.name,
        authorized_route=delegation.route,
        payload=payload_json.encode("utf-8"),
        seeds=dict(seed_inputs),
        timeout_seconds=timeout_seconds,
    )


__all__ = ["GenerationThreads", "submit_named"]
