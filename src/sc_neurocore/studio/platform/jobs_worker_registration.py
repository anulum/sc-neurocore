# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Parent-owned worker registration handshake

"""Gate managed task import on committed registration, without worker database access."""

from __future__ import annotations

import os
import logging
import select
import subprocess
import threading
import time
from typing import TYPE_CHECKING

from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity

if TYPE_CHECKING:
    from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger

_LOG = logging.getLogger(__name__)


def start_worker_registration(
    ledger: StudioJobLedger,
    job_id: str,
    process: subprocess.Popen[bytes],
    expected_supervisor: str,
) -> threading.Thread:
    """Capture the owned child before polling, then commit asynchronously and grant once.

    The supervisor continues timeout/cancellation monitoring while SQLite may
    wait for its writer lock. EOF is refusal. Identity arguments are observations
    of the trusted parent, not an interface for untrusted network assertions.
    """
    stream = process.stdin
    if stream is None:
        raise ValueError("Managed worker needs a private registration pipe.")
    # Observation never raises: unreadable metadata yields start token 0,
    # which registration refuses.
    identity = supervisor_identity(process.pid)

    def register() -> None:
        from sc_neurocore.studio.platform.jobs_worker_custody import register_worker

        try:
            with stream:
                register_worker(ledger, job_id, expected_supervisor, identity, process.pid)
                stream.write(b"ready\n")
                stream.flush()
        except Exception as exc:
            # A missing grant fails closed in the worker. The supervisor owns
            # result/error observation and group reaping, including late EOF.
            _LOG.warning("Studio worker registration refused: %s", type(exc).__name__)
        finally:
            ledger.close()

    try:
        thread = threading.Thread(target=register, daemon=True, name=f"studio-register-{job_id}")
        thread.start()
    except BaseException:
        stream.close()
        raise
    return thread


def await_worker_registration(descriptor: int) -> None:
    """Require the exact parent grant within three seconds; EOF or malformed input refuses."""
    expected = b"ready\n"
    received = b""
    deadline = time.monotonic() + 3.0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([descriptor], [], [], remaining)[0]:
            raise RuntimeError("Worker registration timed out.")
        part = os.read(descriptor, len(expected) + 1 - len(received))
        if not part and received == expected:
            return
        received += part
        if not part or not expected.startswith(received):
            raise RuntimeError("Worker registration was not confirmed.")
