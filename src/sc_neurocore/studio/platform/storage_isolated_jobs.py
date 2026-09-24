# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — isolated Studio jobs facade

"""The Studio job methods the API uses, served by the storage authority.

Each method keeps the embedded manager's signature and outcome shape. Every
durable read and mutation is a storage operation for the requester the
security middleware delegated for the current request; a call outside a
request is refused rather than made for nobody. Live reads and control go
through the spool of generations this API supervises. No local ledger is
created and nothing falls back to embedded storage.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
import time

from pydantic import TypeAdapter, ValidationError

from sc_neurocore.studio.platform.jobs_ledger_schema import TERMINAL_STATUSES
from sc_neurocore.studio.platform.jobs_models import (
    StudioJobArtifactPayload,
    StudioJobListSnapshot,
    StudioJobPurgeSnapshot,
    StudioJobRecord,
    StudioJobRejected,
    StudioJobStatusSnapshot,
    StudioProcessJobPayload,
)
from sc_neurocore.studio.platform.jobs_process_protocol import _json_payload
from sc_neurocore.studio.platform.storage_artifact_client import artifact_request, exchange_artifact
from sc_neurocore.studio.platform.storage_artifact_protocol import ArtifactRoute
from sc_neurocore.studio.platform.storage_cancel_client import cancel_request, exchange_cancel
from sc_neurocore.studio.platform.storage_configuration import StorageBoundaryConfiguration
from sc_neurocore.studio.platform.storage_generation_exchanges import GenerationRuntime
from sc_neurocore.studio.platform.storage_isolated_submit import GenerationThreads, submit_named
from sc_neurocore.studio.platform.storage_purge_client import exchange_purge, purge_request
from sc_neurocore.studio.platform.storage_query_client import QueryReader
from sc_neurocore.studio.platform.storage_record_client import read_storage_record
from sc_neurocore.studio.platform.storage_record_protocol import StorageRecordRequest
from sc_neurocore.studio.platform.storage_requester import Delegation, current_delegation


_ARTIFACT_ROUTE: TypeAdapter[ArtifactRoute] = TypeAdapter(ArtifactRoute)


def _delegation() -> Delegation:
    delegation = current_delegation()
    if delegation is None:
        raise PermissionError("Studio isolated storage acts only for an authorised request.")
    return delegation


class IsolatedJobManager:
    """Serve the API's job methods through the storage authority and launcher."""

    def __init__(
        self,
        runtime: GenerationRuntime,
        configuration: StorageBoundaryConfiguration,
        *,
        allowed_kinds: frozenset[str],
        default_timeout_seconds: float,
    ) -> None:
        """Bind the trusted runtime and boundary; nothing is contacted yet."""
        self._runtime = runtime
        self._configuration = configuration
        self._allowed_kinds = allowed_kinds
        self._default_timeout_seconds = default_timeout_seconds
        self._threads = GenerationThreads(runtime)
        self._reader = QueryReader(
            runtime.connect,
            workspace=configuration.workspace,
            storage_uid=runtime.storage_uid,
            max_bytes=runtime.frame_max_bytes,
            timeout_seconds=runtime.transfer_timeout_seconds,
        )

    def _deadline(self) -> float:
        return time.monotonic() + self._runtime.transfer_timeout_seconds

    def submit_process_task(
        self,
        *,
        kind: str,
        owner: str,
        request_id: str | None,
        task_path: str,
        payload: StudioProcessJobPayload,
        timeout_seconds: float | None = None,
        seed_inputs: Mapping[str, bytes] | None = None,
        workspace: str | None = None,
        idempotency_key: str | None = None,
        experiment_sha256: str | None = None,
        admission: Mapping[str, object] | None = None,
        training_config: Mapping[str, object] | None = None,
    ) -> StudioJobRecord:
        """Admit a named process job and supervise its launched worker here."""
        if kind not in self._allowed_kinds:
            raise StudioJobRejected(f"Studio job kind '{kind}' is not allowed.")
        job = submit_named(
            self._configuration,
            _delegation(),
            kind=kind,
            owner=owner,
            request_id=request_id,
            task_path=task_path,
            payload=payload,
            timeout_seconds=(
                self._default_timeout_seconds if timeout_seconds is None else timeout_seconds
            ),
            seed_inputs=dict(seed_inputs or {}),
            workspace=workspace,
            idempotency_key=idempotency_key,
            experiment_sha256=experiment_sha256,
            admission=admission,
            training_config=training_config,
            connect=self._runtime.connect,
        )
        record = self.record(job.job_id)
        if record.status == "pending":
            self._threads.start(job)
        return record

    @property
    def generation_failures(self) -> dict[str, BaseException]:
        """Return supervised generations that ended without a final finish reply.

        Their jobs keep their delegated lease and reservation at the authority;
        this is the operator's view of what this API generation could not settle.
        """
        return self._threads.failures()

    def record(self, job_id: str) -> StudioJobRecord:
        """Return one record of the configured workspace."""
        request = StorageRecordRequest(
            schema_version="studio.storage.record.v2",
            operation="record",
            request_id=_delegation().request_id,
            job_id=job_id,
            workspace=self._configuration.workspace,
            requester=_delegation().requester,
        )
        return read_storage_record(
            self._runtime.connect(),
            request=request,
            expected_service_uid=self._runtime.storage_uid,
            max_bytes=self._runtime.frame_max_bytes,
            deadline=self._deadline(),
        )

    def wait(self, job_id: str, timeout_seconds: float | None = None) -> StudioJobRecord:
        """Observe the record until terminal or the deadline, as embedded."""
        if timeout_seconds is not None and not math.isfinite(timeout_seconds):
            raise ValueError("Studio job wait timeout must be finite or None.")
        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        events = self._threads.events(job_id)
        record = self.record(job_id)
        while record.status not in TERMINAL_STATUSES:
            delay = 0.05
            if deadline is not None:
                delay = min(delay, max(0.0, deadline - time.monotonic()))
                if delay == 0.0:
                    return record
            if events is None:
                time.sleep(delay)
            else:
                events[1].wait(delay)
            record = self.record(job_id)
        return record

    def list_records(
        self, *, actor: str | None = None, workspace: str | None = None
    ) -> tuple[StudioJobRecord, ...]:
        """Return the workspace's records in creation order, scoped when asked."""
        records = self._reader.records(_delegation().requester)
        return tuple(
            record
            for record in records
            if (actor is None or record.owner == actor)
            and (workspace is None or record.workspace == workspace)
        )

    def list_snapshot(
        self, *, actor: str | None = None, workspace: str | None = None
    ) -> StudioJobListSnapshot:
        """Return a path-free snapshot of every visible job."""
        return StudioJobListSnapshot(records=self.list_records(actor=actor, workspace=workspace))

    def purge_snapshot(
        self, *, limit: int = 100, after: str | None = None
    ) -> StudioJobPurgeSnapshot:
        """Read one operator purge journal page."""
        return self._reader.purges(_delegation().requester, limit=limit, after=after)

    @property
    def unreaped_workers(self) -> tuple[str, ...]:
        """Return jobs whose capacity is held because their workers were not reaped."""
        return tuple(self._reader.status(_delegation().requester).unreaped)

    def status(self) -> StudioJobStatusSnapshot:
        """Return aggregate path-free health from the authority's summary."""
        return self._reader.status(_delegation().requester).snapshot(
            allowed_kinds=self._allowed_kinds,
            default_timeout_seconds=self._default_timeout_seconds,
            max_artifact_bytes=self._runtime.max_artifact_bytes,
        )

    def cancel(self, job_id: str) -> StudioJobRecord:
        """Record the cancellation at the authority; stop a worker supervised here."""
        events = self._threads.events(job_id)
        if events is not None:
            events[0].set()
        record = exchange_cancel(
            self._runtime.connect(),
            cancel_request(
                self._configuration.workspace, job_id, requester=_delegation().requester
            ),
            expected_service_uid=self._runtime.storage_uid,
            max_bytes=self._runtime.frame_max_bytes,
            deadline=self._deadline(),
        )
        if record.status not in TERMINAL_STATUSES and record.status != "cancelling":
            raise StudioJobRejected(
                f"Studio job {job_id} cannot move from '{record.status}' to 'cancelling'."
            )
        return record

    def read_artifact(self, job_id: str, relative_path: str) -> StudioJobArtifactPayload:
        """Read one sealed artefact through the route the request was authorised for."""
        delegation = _delegation()
        try:
            route = _ARTIFACT_ROUTE.validate_python(delegation.route)
        except ValidationError as exc:
            raise PermissionError("this route does not read completed artefacts") from exc
        return exchange_artifact(
            self._runtime.connect(),
            artifact_request(
                self._configuration.workspace,
                job_id,
                relative_path,
                route=route,
                requester=delegation.requester,
            ),
            expected_service_uid=self._runtime.storage_uid,
            max_bytes=self._runtime.frame_max_bytes,
            deadline=self._deadline(),
        )

    def read_live_artifact_bytes(
        self, job_id: str, relative_path: str, *, offset: int, max_bytes: int = 64 * 1024
    ) -> tuple[bytes, int]:
        """Read one bounded slice from a live artefact of a job supervised here."""
        self.record(job_id)
        return self._runtime.live.read(job_id, relative_path, offset=offset, max_bytes=max_bytes)

    def send_control_command(
        self,
        job_id: str,
        *,
        command: Mapping[str, object],
        seed_inputs: Mapping[str, bytes] | None = None,
    ) -> None:
        """Deliver a command and control seeds to a running job supervised here."""
        if self.record(job_id).status != "running":
            raise StudioJobRejected("Studio job is not running.")
        encoded = _json_payload(command, "Studio job control command must be JSON.")
        self._runtime.live.deliver(job_id, encoded.encode("utf-8"), dict(seed_inputs or {}))

    def purge_terminal_record(self, job_id: str) -> StudioJobRecord:
        """Purge one terminal job and its sealed directory at the authority."""
        return exchange_purge(
            self._runtime.connect(),
            purge_request(self._configuration.workspace, job_id, requester=_delegation().requester),
            expected_service_uid=self._runtime.storage_uid,
            max_bytes=self._runtime.frame_max_bytes,
            deadline=self._deadline(),
        )


__all__ = ["IsolatedJobManager"]
