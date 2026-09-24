# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Independent worker group lifetime guard

"""Keep an independent interpreter in the owned group until supervisor cleanup."""

from __future__ import annotations

from contextlib import suppress
import os
import select
import signal
import subprocess
import sys
import time

from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_is_alive


def _await_ready(descriptor: int) -> None:
    """Read only the readiness token, with a three-second monotonic deadline."""
    expected = b"ready\n"
    received = b""
    deadline = time.monotonic() + 3.0
    while received != expected:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([descriptor], [], [], remaining)[0]:
            raise RuntimeError("Worker lifetime guard readiness timed out.")
        chunk = os.read(descriptor, len(expected) - len(received))
        received += chunk
        if not chunk or not expected.startswith(received):
            raise RuntimeError("Worker lifetime guard failed to arm.")


def arm_worker_guard(supervisor: str) -> subprocess.Popen[bytes]:
    """Arm before importing task code; refuse execution without a ready guard.

    The guard shares the worker's dedicated group. It survives worker GIL stalls
    and is stopped with the group by normal supervisor reaping. It never signals
    a group supplied by an unrelated process: its own membership retains custody.
    """
    group = os.getpgrp()
    if group != os.getpid():
        raise ValueError("Worker guard requires a dedicated session leader.")
    process = subprocess.Popen(
        [sys.executable, "-m", __name__, supervisor, str(group)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    assert process.stdout is not None
    try:
        with process.stdout:
            _await_ready(process.stdout.fileno())
    except BaseException:
        # A rejected handshake must not abandon the direct guard child.
        # The outer supervisor still owns whole-group cleanup if this fails.
        process.kill()
        process.wait(timeout=1.0)
        raise
    return process


def _stop_group(group: int) -> None:
    """Leave the worker's group, then kill every process still in it.

    Leaving first lets the guard report its own exit status; a group that has
    already emptied needs no signal.
    """
    os.setpgid(0, 0)
    with suppress(ProcessLookupError):
        os.killpg(group, signal.SIGKILL)


def main() -> int:
    """Stop the worker's group on supervisor death or prolonged unknown liveness.

    Poll every 100 ms; tolerate unknown metadata for at most one second. Exit
    with status 1 after stopping the group. This guards local supervised
    compute, not hostile processes escaping the session or a system where the
    guard itself cannot be scheduled.
    """
    if len(sys.argv) != 3:
        return 2
    supervisor, group_text = sys.argv[1:]
    group = int(group_text)
    if group <= 0 or os.getpgrp() != group or os.getpid() == group:
        return 2
    if supervisor_is_alive(supervisor) is not True:
        _stop_group(group)
        return 1
    print("ready", flush=True)
    sys.stdout.close()
    unknown_since: float | None = None
    while True:
        alive = supervisor_is_alive(supervisor)
        if alive is True:
            unknown_since = None
        elif alive is None and unknown_since is None:
            unknown_since = time.monotonic()
        if alive is False or (
            unknown_since is not None and time.monotonic() - unknown_since >= 1.0
        ):
            _stop_group(group)
            return 1
        time.sleep(0.1)


if __name__ == "__main__":
    raise SystemExit(main())
