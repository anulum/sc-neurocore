# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Owned desktop backend process

"""Serve the existing application on a descriptor reserved by its desktop parent."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import time
from types import FrameType

import uvicorn

from sc_neurocore import __version__
from sc_neurocore.studio.desktop import DesktopStartRefused
from sc_neurocore.studio.platform.jobs_ledger_schema import TERMINAL_STATUSES
from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.jobs_worker_guard import arm_worker_guard
from sc_neurocore.studio.platform.studio_job_service import StudioJobService


def _cancel_owned_jobs(manager: StudioJobService) -> bool:
    """Cancel this process's jobs under one deadline, retaining foreign custody."""
    identity = supervisor_identity()
    active = [
        record
        for record in manager.list_records()
        if record.lease_owner == identity and record.status not in TERMINAL_STATUSES
    ]
    deadline = time.monotonic() + 5.0
    for record in active:
        manager.cancel(record.job_id)
    settled = [
        manager.wait(record.job_id, timeout_seconds=max(0.0, deadline - time.monotonic()))
        for record in active
    ]
    return (
        all(record.status in TERMINAL_STATUSES for record in settled)
        and not manager.unreaped_workers
    )


def main() -> int:
    """Run a descriptor-owned local backend and collect its lifetime guard.

    Returns
    -------
    int
        Zero after a clean server/job shutdown; nonzero for refused startup or
        unverified worker cleanup. Arguments are internal launcher descriptors,
        not a public network control interface.

    Raises
    ------
    DesktopStartRefused
        The inherited listener is not an IPv4 loopback stream socket or the
        installation has no built Studio interface.
    OSError
        A required descriptor or readiness write failed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket-fd", type=int, required=True)
    parser.add_argument("--ready-fd", type=int, required=True)
    parser.add_argument("--lock-fd", type=int, required=True)
    parser.add_argument("--parent", required=True)
    parser.add_argument("--nonce", required=True)
    args = parser.parse_args()
    guard = arm_worker_guard(args.parent)
    manager: StudioJobService | None = None
    cleaned = False
    try:
        from sc_neurocore.studio.app import create_app
        from sc_neurocore.studio.api.frontend import studio_frontend_candidates, studio_frontend_dir
        from sc_neurocore.studio import app as app_module

        interface = studio_frontend_dir(studio_frontend_candidates(app_module.__file__))
        if interface is None:
            raise DesktopStartRefused("This installation has no built Studio interface.")
        digest = hashlib.sha256((interface / "index.html").read_bytes()).hexdigest()
        application = create_app()
        manager = application.state.studio_job_manager
        with socket.socket(fileno=args.socket_fd) as listener:
            if (
                listener.family != socket.AF_INET
                or listener.getsockname()[0] != "127.0.0.1"
                or listener.type != socket.SOCK_STREAM
            ):
                raise DesktopStartRefused(
                    "The desktop backend requires its owned loopback listener."
                )
            config = uvicorn.Config(
                application,
                log_level="warning",
                timeout_graceful_shutdown=5,
                timeout_notify=0,
            )

            async def ready() -> None:
                """Publish this process's actual package and served UI identity once."""
                config.callback_notify = None
                payload = {
                    "schema_version": "studio.desktop.ready.v1",
                    "pid": os.getpid(),
                    "nonce": args.nonce,
                    "version": __version__,
                    "ui_sha256": digest,
                }
                os.write(args.ready_fd, json.dumps(payload, sort_keys=True).encode() + b"\n")
                os.close(args.ready_fd)

            config.callback_notify = ready
            server = uvicorn.Server(config)

            def stop(_number: int, _frame: FrameType | None) -> None:
                """Retain a clean return when Uvicorn restores and repeats a signal."""
                server.should_exit = True

            signal.signal(signal.SIGTERM, stop)
            signal.signal(signal.SIGINT, stop)
            server.run(sockets=[listener])
        complete = _cancel_owned_jobs(manager)
        cleaned = True
        return 0 if complete else 1
    finally:
        try:
            if manager is not None and not cleaned:
                _cancel_owned_jobs(manager)
        finally:
            guard.terminate()
            guard.wait(timeout=1.0)
            os.close(args.lock_fd)


if __name__ == "__main__":
    raise SystemExit(main())
