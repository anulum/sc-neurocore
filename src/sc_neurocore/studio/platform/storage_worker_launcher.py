# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — direct-spawn worker launcher

"""Start and stop fixed worker generations for the trusted API only.

This is the direct-spawn launcher profile: a small privileged service that
accepts bounded ``studio.launcher.v1`` requests from the configured API UID
and starts one fixed bootstrap per admitted job generation under the compute
identity (supplementary groups dropped, privilege gain forbidden, ceilings
applied by the bootstrap). It never imports task code, stores payloads,
returns output or accepts a command, environment or path from a request.

It is a limited qualification profile, not the production isolated backend:
custody is a pidfd-held process tree plus subreaper adoption, not a cgroup,
and resource ceilings are per-process rlimits, not per-job accounting. A
launcher crash kills each worker leader through its parent-death signal but
cannot stop that leader's other descendants; a privileged launcher therefore
refuses to start while any process of the compute identity exists. Spawns
happen on the serving thread, because the parent-death signal follows the
thread that forked the child. An unprivileged launcher can only start workers
under its own identity and never qualifies isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import select
import socket
import subprocess  # nosec B404 - fixed argument vector, no shell.
import time
from types import TracebackType

from sc_neurocore.studio.platform.storage_launcher_configuration import LauncherConfiguration
from sc_neurocore.studio.platform.storage_launcher_endpoint import LauncherEndpoint
from sc_neurocore.studio.platform.storage_launcher_protocol import (
    LAUNCHER_MESSAGE_MAX_BYTES,
    LAUNCHER_PROTOCOL_VERSION,
    LauncherReason,
    LauncherRequest,
    LauncherResponse,
    LauncherState,
    decode_launcher_request,
    encode_launcher_response,
)
from sc_neurocore.studio.platform.storage_peer import require_storage_peer
from sc_neurocore.studio.platform.storage_transport import read_frame, write_frame
from sc_neurocore.studio.platform.storage_worker_spawn import (
    compute_identity_processes,
    generation_spool_ready,
    spawn_worker_bootstrap,
)
from sc_neurocore.studio.platform.storage_worker_tree import (
    TrackedProcess,
    WorkerTree,
    become_child_subreaper,
    process_start_token,
    reap_adopted,
)


@dataclass(slots=True)
class _Generation:
    generation: str
    process: subprocess.Popen[bytes]
    tree: WorkerTree
    stopped: bool = False


class WorkerLauncher:
    """Serve bounded launch/stop/status requests from the configured API UID.

    Call :meth:`start`, then :meth:`serve_once` repeatedly from one long-lived
    thread, then :meth:`stop`. Every generation stays recorded until
    ``max_records`` forces the oldest stopped record out, so retries and lost
    replies resolve to the same outcome.
    """

    def __init__(self, configuration: LauncherConfiguration) -> None:
        """Retain configuration; nothing is bound or spawned yet."""
        self._config = configuration
        self._records: dict[str, _Generation] = {}
        self._listener: socket.socket | None = None
        self._endpoint: LauncherEndpoint | None = None

    @property
    def privileged(self) -> bool:
        """Return whether this launcher can switch to a different compute UID."""
        return os.geteuid() == 0

    def start(self) -> None:
        """Become a subreaper and bind the API-only endpoint.

        Raises
        ------
        RuntimeError
            Already started.
        PermissionError
            The compute identity differs from this process but it is not
            privileged, a privileged launcher finds processes of the compute
            identity it does not own, or the socket parent is unsafe.
        FileExistsError
            The endpoint already exists.
        OSError
            Subreaper setup, bind or listen fails.
        """
        if self._listener is not None:
            raise RuntimeError("worker launcher is already started")
        config = self._config
        if config.worker_uid != os.geteuid() and not self.privileged:
            raise PermissionError("launcher cannot start workers under another identity")
        if self.privileged and config.api_uid == config.worker_uid:
            raise PermissionError(
                "privileged launcher requires distinct API and compute identities"
            )
        if self.privileged and compute_identity_processes(config.worker_uid):
            raise PermissionError("compute identity has processes this launcher does not own")
        become_child_subreaper()
        endpoint = LauncherEndpoint(config.socket_path)
        self._listener = endpoint.open()
        self._endpoint = endpoint

    def maintain(self) -> tuple[int, ...]:
        """Observe trees, collect leader exits and handle adopted processes.

        Returns
        -------
        tuple[int, ...]
            PIDs of unattributed adopted processes that were killed.
        """
        for record in self._records.values():
            self._settle(record)
        trees = [record.tree for record in self._records.values()]
        leaders = {record.tree.leader.pid for record in self._records.values()}
        return reap_adopted(trees, leaders=leaders)

    def _settle(self, record: _Generation) -> None:
        """Bound a generation by its leader: once it exits, stop what it left.

        The worker's lifetime guard and any other descendant belong to the
        generation only while the worker runs, matching the embedded
        supervisor, which reaps the whole group after the worker exits.
        """
        record.tree.observe()
        if record.stopped or record.process.poll() is None:
            return
        record.stopped = record.tree.kill(rounds=self._config.stop_rounds)

    def serve_once(self) -> None:
        """Handle one request, or return after the transfer timeout with none.

        Raises
        ------
        RuntimeError
            The launcher has not started.
        PermissionError
            The connecting peer is not the configured API identity; the
            connection is closed without a response.
        ValueError
            The request frame is malformed; no response is sent.
        TimeoutError
            The peer did not complete its frame in time.
        """
        listener = self._listener
        if listener is None:
            raise RuntimeError("worker launcher is not started")
        self.maintain()
        timeout = self._config.transfer_timeout_seconds
        if not select.select([listener], [], [], timeout)[0]:
            return
        channel, _ = listener.accept()
        with channel:
            deadline = time.monotonic() + timeout
            require_storage_peer(channel, expected_uid=self._config.api_uid)
            request = decode_launcher_request(
                read_frame(channel, max_bytes=LAUNCHER_MESSAGE_MAX_BYTES, deadline=deadline)
            )
            response = self.handle(request)
            write_frame(
                channel,
                encode_launcher_response(response),
                max_bytes=LAUNCHER_MESSAGE_MAX_BYTES,
                deadline=deadline,
            )
        self.maintain()

    def handle(self, request: LauncherRequest) -> LauncherResponse:
        """Apply one decoded request and return the observed outcome.

        Parameters
        ----------
        request : LauncherRequest
            Request already received from the verified API peer.

        Returns
        -------
        LauncherResponse
            Correlated state for the exact job generation.
        """
        if request.operation == "launch":
            return self._launch(request)
        if request.operation == "stop":
            return self._stop(request)
        return self._status(request)

    def _reply(
        self,
        request: LauncherRequest,
        state: LauncherState,
        *,
        leader: TrackedProcess | None = None,
        reason: LauncherReason | None = None,
        exit_status: int | None = None,
    ) -> LauncherResponse:
        return LauncherResponse(
            version=LAUNCHER_PROTOCOL_VERSION,
            request_id=request.request_id,
            operation=request.operation,
            job_id=request.job_id,
            generation=request.generation,
            state=state,
            pid=None if leader is None else leader.pid,
            start_token=None if leader is None else leader.start_token,
            reason=reason,
            exit_status=exit_status,
        )

    def _observed(self, request: LauncherRequest, record: _Generation) -> LauncherResponse:
        self._settle(record)
        if record.stopped:
            return self._reply(
                request,
                "stopped",
                leader=record.tree.leader,
                exit_status=record.process.returncode,
            )
        return self._reply(request, "running", leader=record.tree.leader)

    def _launch(self, request: LauncherRequest) -> LauncherResponse:
        record = self._records.get(request.job_id)
        if record is not None:
            if record.generation != request.generation:
                return self._reply(request, "refused", reason="conflict")
            return self._observed(request, record)
        live = sum(1 for item in self._records.values() if item.tree.live())
        config = self._config
        if live >= config.max_workers:
            return self._reply(request, "refused", reason="capacity")
        if not generation_spool_ready(
            config.spool_root, request.job_id, request.generation, config.api_uid
        ):
            return self._reply(request, "refused", reason="spool")
        if not self._retire_for_new_record():
            return self._reply(request, "refused", reason="capacity")
        try:
            process = spawn_worker_bootstrap(
                self._config,
                job_id=request.job_id,
                generation=request.generation,
                privileged=self.privileged,
            )
        except (OSError, ValueError, subprocess.SubprocessError):
            return self._reply(request, "refused", reason="unavailable")
        try:
            pidfd = os.pidfd_open(process.pid)
            leader = TrackedProcess(
                pid=process.pid, pidfd=pidfd, start_token=process_start_token(process.pid)
            )
        except OSError:
            process.kill()
            process.wait()
            return self._reply(request, "refused", reason="unavailable")
        self._records[request.job_id] = _Generation(
            generation=request.generation, process=process, tree=WorkerTree(leader)
        )
        return self._reply(request, "running", leader=leader)

    def _retire_for_new_record(self) -> bool:
        if len(self._records) < self._config.max_records:
            return True
        for job_id, record in self._records.items():
            if record.stopped and record.process.poll() is not None:
                record.tree.close()
                del self._records[job_id]
                return True
        return False

    def _stop(self, request: LauncherRequest) -> LauncherResponse:
        record = self._records.get(request.job_id)
        if record is None or record.generation != request.generation:
            return self._reply(request, "absent")
        stopped = record.tree.kill(rounds=self._config.stop_rounds)
        self.maintain()
        if not stopped:
            return self._reply(request, "refused", leader=record.tree.leader, reason="survivors")
        record.stopped = True
        # The leader's pidfd reported its exit, so collecting it cannot block.
        exit_status = record.process.wait()
        return self._reply(request, "stopped", leader=record.tree.leader, exit_status=exit_status)

    def _status(self, request: LauncherRequest) -> LauncherResponse:
        record = self._records.get(request.job_id)
        if record is None or record.generation != request.generation:
            return self._reply(request, "absent")
        return self._observed(request, record)

    def stop(self) -> None:
        """Close the endpoint and remove only its own unchanged socket.

        Running workers are not stopped here; the caller stops each generation
        explicitly, and a crashed launcher relies on the documented limits.
        """
        listener = self._listener
        endpoint = self._endpoint
        self._listener = None
        self._endpoint = None
        if listener is None or endpoint is None:
            return
        listener.close()
        endpoint.close()

    def __enter__(self) -> WorkerLauncher:
        """Start the launcher and return its single owner."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the endpoint without suppressing caller failures."""
        self.stop()
