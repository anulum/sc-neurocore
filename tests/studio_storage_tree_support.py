# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — worker process tree test support

"""Real leader processes and observation helpers shared by the tree test modules."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from sc_neurocore.studio.platform.jobs_process_protocol import _process_worker_environment
from sc_neurocore.studio.platform.storage_worker_tree import (
    TrackedProcess,
    WorkerTree,
    process_start_token,
)

PREFIX = (
    "import json, os, signal, sys, time\n"
    "from sc_neurocore.studio.platform.storage_worker_tree import become_child_subreaper\n"
)


def leader(body: str) -> subprocess.Popen[str]:
    """Start a leader that reports the PIDs it created as one JSON line."""
    return subprocess.Popen(
        [sys.executable, "-c", PREFIX + body],
        env=_process_worker_environment(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def tree_for(child: subprocess.Popen[str]) -> WorkerTree:
    """Track the leader ``child`` through a pidfd and its recorded start time."""
    pidfd = os.pidfd_open(child.pid)
    return WorkerTree(
        TrackedProcess(pid=child.pid, pidfd=pidfd, start_token=process_start_token(child.pid))
    )


def reported(child: subprocess.Popen[str]) -> list[int]:
    """Read the descendant PIDs that the leader ``child`` printed as one JSON list."""
    assert child.stdout is not None
    line = child.stdout.readline()
    assert line, "leader exited before reporting its descendants"
    value = json.loads(line)
    assert isinstance(value, list)
    return [int(item) for item in value]


def observe_until(tree: WorkerTree, pids: list[int]) -> None:
    """Observe ``tree`` until it owns every PID in ``pids``, failing after ten seconds."""
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        tree.observe()
        if all(tree.owns(pid) for pid in pids):
            return
        time.sleep(0.02)
    raise AssertionError(f"tree did not observe {pids}")


def gone(pid: int, start_token: str | None = None) -> bool:
    """Return whether the process ``pid`` named has exited.

    A zombie (``Z``), a task being reaped (``X``) and a missing ``/proc`` entry
    have all exited. With ``start_token``, a live process whose start time
    differs is a later process that reused the PID after the named one was
    reaped, so the named process has exited too.
    """
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as handle:
            fields = handle.read().rsplit(")", 1)[-1].split()
    except (FileNotFoundError, ProcessLookupError):
        return True
    if fields[0] in ("Z", "X"):
        return True
    return start_token is not None and fields[19] != start_token


def finish(child: subprocess.Popen[str], tree: WorkerTree) -> None:
    """Kill the whole tree, wait for the leader and release its pidfd and pipe."""
    try:
        tree.kill(rounds=20)
        child.wait(timeout=10.0)
    finally:
        tree.close()
        if child.stdout is not None:
            child.stdout.close()


def ppid(pid: int) -> int:
    """Return the parent PID of ``pid`` from ``/proc``, or -1 when it has gone."""
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as handle:
            return int(handle.read().rsplit(")", 1)[-1].split()[1])
    except (FileNotFoundError, ProcessLookupError):
        return -1


def one_child_leader() -> tuple[subprocess.Popen[str], WorkerTree, int]:
    """Start a leader with one forked child and return the leader, its tree and the child PID."""
    child = leader(
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    time.sleep(60); os._exit(0)\n"
        "print(json.dumps([pid]), flush=True)\n"
        "time.sleep(60)\n"
    )
    tree = tree_for(child)
    (grand,) = reported(child)
    return child, tree, grand
