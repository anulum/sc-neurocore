# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — launcher-held worker process tree custody

"""Real process trees, escapes and orphans must stay in launcher custody.

Cases that make a process a child subreaper or reap its adopted children run
in a separate harness interpreter, so the test runner itself never changes
its process-management role or kills unrelated children.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs_process_protocol import _process_worker_environment
from tests.studio_storage_tree_support import *


def test_kill_stops_leader_children_and_grandchildren() -> None:
    """Every observed descendant generation is signalled and confirmed exited."""
    child = leader(
        "pids = []\n"
        "for _ in range(2):\n"
        "    pid = os.fork()\n"
        "    if pid == 0:\n"
        "        grand = os.fork()\n"
        "        if grand == 0:\n"
        "            time.sleep(60); os._exit(0)\n"
        "        print(json.dumps([os.getpid(), grand]), flush=True)\n"
        "        time.sleep(60); os._exit(0)\n"
        "    pids.append(pid)\n"
        "time.sleep(60)\n"
    )
    tree = tree_for(child)
    try:
        assert child.stdout is not None
        descendants = [pid for _ in range(2) for pid in reported(child)]
        observe_until(tree, descendants)
        assert {member.pid for member in tree.live()} == {child.pid, *descendants}
        assert tree.kill(rounds=5) is True
        assert tree.live() == ()
        child.wait(timeout=10.0)
        assert all(gone(pid) for pid in descendants)
    finally:
        finish(child, tree)


def test_setsid_double_fork_escape_stays_under_subreaper_leader() -> None:
    """A grandchild that leaves the session through a double fork is still caught."""
    child = leader(
        "become_child_subreaper()\n"
        "read, write = os.pipe()\n"
        "if os.fork() == 0:\n"
        "    os.setsid()\n"
        "    grand = os.fork()\n"
        "    if grand == 0:\n"
        "        time.sleep(60); os._exit(0)\n"
        "    os.write(write, str(grand).encode()); os._exit(0)\n"
        "grand = int(os.read(read, 32))\n"
        "time.sleep(0.2)\n"
        "print(json.dumps([grand]), flush=True)\n"
        "time.sleep(60)\n"
    )
    tree = tree_for(child)
    try:
        (escaped,) = reported(child)
        with open(f"/proc/{escaped}/stat", encoding="ascii") as handle:
            fields = handle.read().rsplit(")", 1)[-1].split()
        assert int(fields[1]) == child.pid, "escape was not reparented to the subreaper"
        assert int(fields[3]) != os.getsid(child.pid), "escape did not leave the session"
        observe_until(tree, [escaped])
        assert tree.kill(rounds=5) is True
        child.wait(timeout=10.0)
        assert gone(escaped)
    finally:
        finish(child, tree)


def test_fork_storm_is_stopped_within_round_budget() -> None:
    """Children forked continuously during the stop are observed and killed."""
    child = leader(
        "print(json.dumps([]), flush=True)\n"
        "for _ in range(200):\n"
        "    if os.fork() == 0:\n"
        "        time.sleep(60); os._exit(0)\n"
        "    time.sleep(0.005)\n"
        "time.sleep(60)\n"
    )
    tree = tree_for(child)
    try:
        reported(child)
        time.sleep(0.2)
        assert tree.kill(rounds=50) is True
        child.wait(timeout=10.0)
        tree.observe()
        assert tree.live() == ()
        survivors = [
            entry.name
            for entry in Path("/proc").iterdir()
            if entry.name.isdigit()
            and ppid(int(entry.name)) == child.pid
            and not gone(int(entry.name))
        ]
        assert survivors == []
    finally:
        finish(child, tree)


def test_exited_leader_is_reported_without_live_members() -> None:
    """A leader that finished on its own leaves no live member to stop."""
    child = leader("print(json.dumps([]), flush=True)\n")
    tree = tree_for(child)
    try:
        reported(child)
        child.wait(timeout=10.0)
        tree.observe()
        assert tree.live() == ()
        assert tree.kill(rounds=1) is True
        assert tree.leader.pid == child.pid
    finally:
        finish(child, tree)


def test_kill_rounds_must_be_positive() -> None:
    """A stop budget that could never signal anything is refused."""
    child = leader("print(json.dumps([]), flush=True)\ntime.sleep(60)\n")
    tree = tree_for(child)
    try:
        reported(child)
        for rounds in (0, -1, True):
            with pytest.raises(ValueError, match="rounds must be positive"):
                tree.kill(rounds=rounds)
    finally:
        finish(child, tree)


def test_close_releases_every_pidfd() -> None:
    """Closing the tree closes the leader and descendant pidfds exactly once."""
    child = leader(
        "if os.fork() == 0:\n"
        "    time.sleep(60); os._exit(0)\n"
        "print(json.dumps([]), flush=True)\n"
        "time.sleep(60)\n"
    )
    tree = tree_for(child)
    try:
        reported(child)
        deadline = time.monotonic() + 10.0
        while len(tree.live()) < 2 and time.monotonic() < deadline:
            tree.observe()
            time.sleep(0.02)
        members = tree.live()
        assert len(members) == 2
        tree.kill(rounds=5)
        child.wait(timeout=10.0)
        tree.close()
        for member in members:
            with pytest.raises(OSError):
                os.fstat(member.pidfd)
        tree.close()
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=5.0)
        if child.stdout is not None:
            child.stdout.close()


_OWNED_LEADER = (
    "import os, time\n"
    "read, write = os.pipe()\n"
    "if os.fork() == 0:\n"
    "    owned = os.fork()\n"
    "    if owned == 0:\n"
    "        time.sleep(60); os._exit(0)\n"
    "    os.write(write, str(owned).encode()); time.sleep(1.5); os._exit(0)\n"
    "print(int(os.read(read, 32)), flush=True)\n"
    "time.sleep(60)\n"
)


_HARNESS = (
    "import json, os, subprocess, sys, time\n"
    "from sc_neurocore.studio.platform.storage_worker_tree import (\n"
    "    TrackedProcess, WorkerTree, become_child_subreaper, process_start_token, reap_adopted)\n"
    "become_child_subreaper()\n"
    "def state(pid):\n"
    "    try:\n"
    "        with open(f'/proc/{pid}/stat') as handle:\n"
    "            fields = handle.read().rsplit(')', 1)[-1].split()\n"
    "        return fields[0], int(fields[1])\n"
    "    except FileNotFoundError:\n"
    "        return 'gone', -1\n"
    "read, write = os.pipe()\n"
    "helper = os.fork()\n"
    "if helper == 0:\n"
    "    os.setsid()\n"
    "    grand = os.fork()\n"
    "    if grand == 0:\n"
    "        time.sleep(60); os._exit(0)\n"
    "    os.write(write, str(grand).encode()); os._exit(0)\n"
    "os.waitpid(helper, 0)\n"
    "stray = int(os.read(read, 32))\n"
    "leader = subprocess.Popen([sys.executable, '-c', sys.argv[1]], stdout=subprocess.PIPE,\n"
    "    text=True)\n"
    "owned = int(leader.stdout.readline())\n"
    "tree = WorkerTree(TrackedProcess(leader.pid, os.pidfd_open(leader.pid),\n"
    "    process_start_token(leader.pid)))\n"
    "deadline = time.monotonic() + 10.0\n"
    "while not tree.owns(owned) and time.monotonic() < deadline:\n"
    "    tree.observe(); time.sleep(0.02)\n"
    "while state(owned)[1] != os.getpid() and time.monotonic() < deadline:\n"
    "    time.sleep(0.02)\n"
    "adopted_owned = state(owned)[1] == os.getpid()\n"
    "killed = reap_adopted([tree], leaders={leader.pid})\n"
    "time.sleep(0.3)\n"
    "stray_after_kill = state(stray)[0]\n"
    "reap_adopted([tree], leaders={leader.pid})\n"
    "result = {'killed': list(killed), 'stray': stray, 'stray_after_kill': stray_after_kill,\n"
    "    'stray_after_reap': state(stray)[0], 'owned_adopted': adopted_owned,\n"
    "    'owned_state': state(owned)[0], 'leader_state': state(leader.pid)[0]}\n"
    "result['stopped'] = tree.kill(rounds=5)\n"
    "leader.wait()\n"
    "reap_adopted([tree], leaders=set())\n"
    "result['owned_after_stop'] = state(owned)[0]\n"
    "tree.close()\n"
    "print(json.dumps(result))\n"
)


def test_adopted_orphans_are_attributed_killed_and_reaped() -> None:
    """A subreaper kills unattributed adopted processes and keeps attributed ones."""
    completed = subprocess.run(
        [sys.executable, "-c", _HARNESS, _OWNED_LEADER],
        env=_process_worker_environment(),
        capture_output=True,
        text=True,
        timeout=60.0,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["killed"] == [result["stray"]]
    assert result["stray_after_kill"] in {"Z", "gone"}
    assert result["stray_after_reap"] == "gone"
    assert result["owned_adopted"] is True
    assert result["owned_state"] == "S"
    assert result["leader_state"] == "S"
    assert result["stopped"] is True
    assert result["owned_after_stop"] == "gone"
