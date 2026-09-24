# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — API exchanges for one launched worker generation

"""The API's launcher and storage exchanges for one job generation.

Each exchange that can lose its reply has one fixed rule. A lost ``launch`` is
resolved by ``status`` of the same generation and repeated only for that
generation, which the launcher answers from its record. ``start`` and
``finish`` are repeated identically; the storage authority answers both
idempotently. A lost heartbeat is renewed at the next interval. A stop that
cannot be confirmed is repeated; a launcher without a record of the generation
never confirms it. Every loop is bounded by ``attempts``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
import secrets
import socket
import time

from sc_neurocore.studio.platform.storage_finish_client import exchange_finish
from sc_neurocore.studio.platform.storage_finish_protocol import (
    FinishOutcome,
    StorageFinishRequest,
    StorageFinishResponse,
)
from sc_neurocore.studio.platform.storage_launcher_client import (
    exchange_launcher_request,
    new_launcher_request,
)
from sc_neurocore.studio.platform.storage_live_spool import LiveSpools
from sc_neurocore.studio.platform.storage_launcher_protocol import (
    LauncherOperation,
    LauncherResponse,
)
from sc_neurocore.studio.platform.storage_supervision_client import exchange_supervision
from sc_neurocore.studio.platform.storage_supervision_protocol import (
    SUPERVISION_SCHEMA_VERSION,
    StorageSupervisionResponse,
    SupervisionHeartbeatRequest,
    SupervisionStartRequest,
)

# Failures after which the peer's action is unknown or the peer unreachable,
# including a malformed reply; each is resolved by its exchange's rule.
_LOST = (OSError, EOFError, ValueError)


@dataclass(frozen=True, slots=True)
class GenerationRuntime:
    """Trusted API settings for supervising launched worker generations.

    ``connect`` returns a new peer-verified stream to the storage authority,
    for example :func:`storage_connection.connect_storage_authority` bound to
    the boundary configuration. ``max_artifact_bytes`` is the worker's
    per-artefact budget; ``artifact_total_bytes`` and ``artifact_entries`` are
    the aggregate budgets the authority enforces. ``attempts`` bounds each
    resolution loop; ``transfer_timeout_seconds`` bounds each single exchange.
    ``live`` is this API generation's registry of live worker directories.
    """

    workspace: str
    spool_root: Path
    storage_uid: int
    frame_max_bytes: int
    transfer_timeout_seconds: float
    launcher_socket: Path
    launcher_uid: int
    worker_uid: int
    worker_gid: int
    max_artifact_bytes: int
    artifact_total_bytes: int
    artifact_entries: int
    grant_timeout_seconds: float
    heartbeat_seconds: float
    poll_seconds: float
    attempts: int
    connect: Callable[[], socket.socket]
    live: LiveSpools


@dataclass(frozen=True, slots=True)
class GenerationJob:
    """One admitted job whose delegated lease this API generation owns."""

    job_id: str
    task_name: str
    authorized_route: str
    payload: bytes
    seeds: Mapping[str, bytes]
    timeout_seconds: float


class StartRefused(Exception):
    """The authority did not register the worker; carries the finish verdict."""

    def __init__(self, outcome: FinishOutcome, error: str | None) -> None:
        """Keep the outcome and error the job must finish with."""
        super().__init__(outcome)
        self.outcome: FinishOutcome = outcome
        self.error = error


class GenerationExchanges:
    """Bounded exchanges of one API generation for one exact job generation."""

    def __init__(self, runtime: GenerationRuntime, *, job_id: str, generation: str) -> None:
        """Bind the runtime to the job and its 128-bit generation."""
        self._runtime = runtime
        self._job_id = job_id
        self._generation = generation

    def _deadline(self) -> float:
        return time.monotonic() + self._runtime.transfer_timeout_seconds

    def launcher(self, operation: LauncherOperation) -> LauncherResponse | None:
        """Send one launcher request; ``None`` when no valid reply arrived."""
        runtime = self._runtime
        request = new_launcher_request(operation, job_id=self._job_id, generation=self._generation)
        try:
            return exchange_launcher_request(
                runtime.launcher_socket,
                request,
                launcher_uid=runtime.launcher_uid,
                deadline=self._deadline(),
            )
        except _LOST:
            return None

    def launch(self) -> LauncherResponse | None:
        """Launch this generation; ``None`` when no attempt was answered."""
        for _ in range(self._runtime.attempts):
            response = self.launcher("launch")
            if response is None:
                response = self.launcher("status")
                if response is None or response.state == "absent":
                    continue
            return response
        return None

    def stop(self) -> LauncherResponse | None:
        """Stop this generation; ``None`` when termination was not confirmed.

        Only ``stopped`` confirms it. ``absent`` does not: a restarted
        launcher has no record of a generation whose descendants may survive.
        """
        for _ in range(self._runtime.attempts):
            response = self.launcher("stop")
            if response is not None and response.state == "stopped":
                return response
            if response is not None and response.state == "absent":
                return None
        return None

    def _supervision(
        self, request: SupervisionStartRequest | SupervisionHeartbeatRequest
    ) -> StorageSupervisionResponse | None:
        runtime = self._runtime
        try:
            return exchange_supervision(
                runtime.connect(),
                request,
                expected_service_uid=runtime.storage_uid,
                max_bytes=runtime.frame_max_bytes,
                deadline=self._deadline(),
            )
        except _LOST:
            return None

    def register(self, worker: str) -> None:
        """Have the authority register the verified worker before its grant.

        Raises
        ------
        StartRefused
            The job is cancelling, the authority refused, or no attempt was
            answered; the worker must not receive its grant.
        """
        request = SupervisionStartRequest(
            schema_version=SUPERVISION_SCHEMA_VERSION,
            operation="start",
            request_id=secrets.token_hex(16),
            workspace=self._runtime.workspace,
            job_id=self._job_id,
            worker=worker,
        )
        for _ in range(self._runtime.attempts):
            response = self._supervision(request)
            if response is None:
                continue
            if response.outcome == "started":
                return
            if response.outcome == "cancelling":
                raise StartRefused("cancelled", None)
            raise StartRefused("failed", f"Studio worker could not start: {response.reason}.")
        raise StartRefused("failed", "Studio worker could not start: registration unanswered.")

    def heartbeat(self) -> StorageSupervisionResponse | None:
        """Renew the delegated lease; ``None`` when the reply was lost."""
        return self._supervision(
            SupervisionHeartbeatRequest(
                schema_version=SUPERVISION_SCHEMA_VERSION,
                operation="heartbeat",
                request_id=secrets.token_hex(16),
                workspace=self._runtime.workspace,
                job_id=self._job_id,
            )
        )

    def finish(
        self, request: StorageFinishRequest, payloads: tuple[bytes, ...]
    ) -> StorageFinishResponse:
        """Deliver the finish, repeating it identically after a lost reply.

        A registered worker the authority still sees alive is stopped again
        before the next attempt.

        Raises
        ------
        TimeoutError
            No final answer arrived within the configured attempts; the job
            keeps its delegated lease and reservation.
        """
        runtime = self._runtime
        for _ in range(runtime.attempts):
            try:
                response = exchange_finish(
                    runtime.connect(),
                    request,
                    payloads,
                    expected_service_uid=runtime.storage_uid,
                    max_bytes=runtime.frame_max_bytes,
                    deadline=self._deadline(),
                )
            except _LOST:
                continue
            if response.reply == "refused" and response.reason == "worker_live":
                self.stop()
                continue
            return response
        raise TimeoutError("Studio job finish was not answered")


__all__ = [
    "GenerationExchanges",
    "GenerationJob",
    "GenerationRuntime",
    "StartRefused",
]
