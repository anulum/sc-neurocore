# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — API supervision of one launched worker generation

"""Drive one admitted job through a launched worker generation to its finish.

The trusted API stages the compute spool, opens the generation's grant
endpoint, asks the launcher for the exact generation, and grants the verified
worker only after the storage authority registered it. It then renews the
delegated lease, watches for cancellation and the execution deadline, stops
the generation and finishes the job with the artefacts read from the spool.
Lost replies follow the rules of :mod:`storage_generation_exchanges`.

Outcomes mirror the embedded supervisor: a worker that exits by itself ends
``completed`` only with a completed result and exit status 0; cancellation
and the deadline stop it; a generation whose processes cannot be confirmed
stopped finishes with ``worker_reaped`` false and keeps its capacity.
"""

from __future__ import annotations

import secrets
import threading
import time

from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.storage_finish_client import spool_finish_request
from sc_neurocore.studio.platform.storage_finish_protocol import (
    FINISH_SCHEMA_VERSION,
    FinishOutcome,
    StorageFinishRequest,
    StorageFinishResponse,
)
from sc_neurocore.studio.platform.storage_generation_exchanges import (
    GenerationExchanges,
    GenerationJob,
    GenerationRuntime,
    StartRefused,
)
from sc_neurocore.studio.platform.storage_spool_staging import StagedGeneration, stage_generation
from sc_neurocore.studio.platform.storage_worker_bootstrap import WorkerDescriptor
from sc_neurocore.studio.platform.storage_worker_grant import (
    GRANT_ENDPOINT_NAME,
    ExpectedWorker,
    WorkerGrantEndpoint,
)

_UNREAPED = "The worker processes were not confirmed stopped."
_Finish = tuple[StorageFinishRequest, tuple[bytes, ...]]
_Verdict = tuple[FinishOutcome | None, str | None, int | None]


class GenerationSupervisor:
    """Supervise one job generation from spool staging to its finish reply."""

    def __init__(
        self, runtime: GenerationRuntime, job: GenerationJob, *, cancel: threading.Event
    ) -> None:
        """Choose the generation; nothing is staged, launched or sent yet."""
        self._runtime = runtime
        self._job = job
        self._cancel = cancel
        self._generation = secrets.token_hex(16)
        self._exchanges = GenerationExchanges(
            runtime, job_id=job.job_id, generation=self._generation
        )

    @property
    def generation(self) -> str:
        """Return the 128-bit launch generation of this supervisor."""
        return self._generation

    def _failed(self, error: str, *, reaped: bool, outcome: FinishOutcome = "failed") -> _Finish:
        """Build an unsuccessful finish that declares no artefacts."""
        request = StorageFinishRequest(
            schema_version=FINISH_SCHEMA_VERSION,
            operation="finish",
            request_id=secrets.token_hex(16),
            workspace=self._runtime.workspace,
            job_id=self._job.job_id,
            outcome=outcome,
            result=None,
            error=error[:1024],
            artifacts=(),
            worker_reaped=reaped,
        )
        return request, ()

    def _request(
        self,
        staged: StagedGeneration,
        outcome: FinishOutcome | None,
        *,
        error: str | None,
        exit_status: int | None,
        reaped: bool,
    ) -> _Finish:
        """Build the finish request from the spool.

        Output the API refuses to read fails a job that ended by itself; a
        verdict the API already reached (cancelled, timed out) is kept.
        """
        try:
            return spool_finish_request(
                staged.work,
                workspace=self._runtime.workspace,
                job_id=self._job.job_id,
                outcome=outcome,
                exit_status=exit_status,
                frame_max_bytes=self._runtime.frame_max_bytes,
                max_artifact_bytes=self._runtime.artifact_total_bytes,
                max_artifact_entries=self._runtime.artifact_entries,
                error=error,
                worker_reaped=reaped,
            )
        except (OSError, ValueError) as exc:
            refused = f"Studio worker output was refused: {exc}"
            return self._failed(
                refused if error is None else f"{error} {refused}",
                reaped=reaped,
                outcome=outcome or "failed",
            )

    def _watch(self) -> _Verdict:
        """Supervise the granted worker until it exits or must be stopped.

        Returns the API's verdict (``None`` when the worker exited by itself),
        its error, and the exit status when the worker exited by itself.
        """
        runtime, exchanges = self._runtime, self._exchanges
        deadline = time.monotonic() + self._job.timeout_seconds
        renew_at = time.monotonic() + runtime.heartbeat_seconds
        while True:
            status = exchanges.launcher("status")
            if status is not None and status.state == "stopped":
                return None, None, status.exit_status
            if status is not None and status.state == "absent":
                return "failed", "Studio worker generation is unknown to its launcher.", None
            if self._cancel.is_set():
                return "cancelled", None, None
            now = time.monotonic()
            if now >= deadline:
                return "timed_out", "Studio job exceeded its timeout.", None
            if now >= renew_at:
                renew_at = now + runtime.heartbeat_seconds
                renewed = exchanges.heartbeat()
                if renewed is not None and renewed.outcome == "cancelling":
                    return "cancelled", None, None
                if renewed is not None and renewed.outcome == "refused":
                    return "failed", f"Studio job lease was refused: {renewed.reason}.", None
            self._cancel.wait(runtime.poll_seconds)

    def _grant(self, endpoint: WorkerGrantEndpoint, pid: int, token: str) -> _Verdict | None:
        """Grant the launched worker; a verdict when it must not run."""
        try:
            endpoint.grant(
                ExpectedWorker(uid=self._runtime.worker_uid, pid=pid, start_token=token),
                deadline=time.monotonic() + self._runtime.grant_timeout_seconds,
                max_refusals=self._runtime.attempts,
                register=self._exchanges.register,
            )
        except StartRefused as refused:
            return refused.outcome, refused.error, None
        except (OSError, ValueError) as exc:
            return "failed", f"Studio worker grant failed: {exc}", None
        return None

    def _supervise(self, staged: StagedGeneration) -> _Finish:
        """Launch, grant and watch the worker; return the request to finish with."""
        exchanges = self._exchanges
        endpoint = WorkerGrantEndpoint(staged.directory, staged.path, GRANT_ENDPOINT_NAME)
        try:
            endpoint.open()
        except (OSError, ValueError) as exc:
            return self._failed(
                f"Studio worker could not start: grant endpoint: {exc}", reaped=True
            )
        try:
            launched = exchanges.launch()
            if launched is None:
                # A launch may have happened; only a confirmed stop reaps it.
                unanswered = "Studio worker could not start: launcher unanswered."
                return self._failed(unanswered, reaped=exchanges.stop() is not None)
            if launched.state == "refused":
                refused = f"Studio worker could not start: launcher refused ({launched.reason})."
                # A conflicting generation of this job may still be running.
                return self._failed(refused, reaped=launched.reason != "conflict")
            verdict: _Verdict | None = None
            if launched.state == "running":
                # A running reply always carries both; the protocol validates
                # it, and a missing value would be refused by ExpectedWorker.
                verdict = self._grant(endpoint, launched.pid or 0, launched.start_token or "")
        finally:
            endpoint.close()
        outcome, error, exit_status = verdict or self._watch()
        stopped = exchanges.stop()
        if stopped is None:
            error = _UNREAPED if error is None else f"{error} {_UNREAPED}"
            outcome = outcome or "failed"
        elif exit_status is None:
            exit_status = stopped.exit_status
        return self._request(
            staged, outcome, error=error, exit_status=exit_status, reaped=stopped is not None
        )

    def run(self) -> StorageFinishResponse:
        """Run the generation to the authority's final finish reply.

        Returns
        -------
        StorageFinishResponse
            ``sealed``/``already_sealed``, or the authority's refusal for a job
            that ended or changed ownership elsewhere.

        Raises
        ------
        TimeoutError
            The finish was never answered; custody stays with the authority.
        """
        runtime, job = self._runtime, self._job
        descriptor = WorkerDescriptor(
            version="studio.worker.descriptor.v1",
            job_id=job.job_id,
            generation=self._generation,
            task_name=job.task_name,
            authorized_route=job.authorized_route,
            supervisor=supervisor_identity(),
            max_artifact_bytes=runtime.max_artifact_bytes,
        )
        try:
            staged = stage_generation(
                runtime.spool_root,
                descriptor,
                payload=job.payload,
                seeds=job.seeds,
                group=runtime.worker_gid,
            )
        except (OSError, ValueError) as exc:
            failed = self._failed(f"Studio worker could not start: spool: {exc}", reaped=True)
            return self._exchanges.finish(*failed)
        runtime.live.attach(job.job_id, staged.work)
        try:
            with staged:
                return self._exchanges.finish(*self._supervise(staged))
        finally:
            runtime.live.retire(job.job_id)


def supervise_generation(
    runtime: GenerationRuntime, job: GenerationJob, *, cancel: threading.Event
) -> StorageFinishResponse:
    """Supervise one admitted job's launched generation to its finish reply.

    Parameters
    ----------
    runtime : GenerationRuntime
        Trusted API settings and the storage connection factory.
    job : GenerationJob
        Admitted job whose delegated lease this API generation owns.
    cancel : threading.Event
        Set by the API to cancel the job; the worker is stopped and the job
        finishes ``cancelled``.

    Returns
    -------
    StorageFinishResponse
        The authority's final reply.

    Raises
    ------
    TimeoutError
        The finish was never answered.
    """
    return GenerationSupervisor(runtime, job, cancel=cancel).run()


__all__ = ["GenerationSupervisor", "supervise_generation"]
