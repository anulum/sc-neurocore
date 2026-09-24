# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Managed orphan lifetime

"""Supervisor death must stop admitted compute without a replacement manager."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs_process_state import group_survivors
from sc_neurocore.studio.platform.jobs import StudioJobContext


def native_blocking_descendant_task(
    context: StudioJobContext, payload: Mapping[str, object]
) -> dict[str, object]:
    """Hold the worker GIL in native sleep with a TERM-ignoring child in its group."""
    import ctypes

    descendant = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('ready', flush=True); time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
    )
    assert descendant.stdout is not None
    with descendant.stdout:
        assert descendant.stdout.readline() == b"ready\n"
    context.write_artifact("descendant.pid", str(descendant.pid))
    native_sleep = ctypes.PyDLL(None).sleep
    native_sleep.argtypes = [ctypes.c_uint]
    native_sleep.restype = ctypes.c_uint
    context.write_artifact("worker.pid", str(os.getpid()))
    native_sleep(30)
    return {}


@pytest.mark.parametrize("collect_supervisor", [False, True])
@pytest.mark.parametrize("native_descendant", [False, True])
def test_supervisor_death_stops_worker_without_reconciliation(
    tmp_path: Path, collect_supervisor: bool, native_descendant: bool
) -> None:
    """Exercise the managed entrypoint, real supervisor death and whole-group stop."""
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            """
import sys, threading
from pathlib import Path
from sc_neurocore.studio.platform.jobs import StudioJobManager
m = StudioJobManager(root=Path(sys.argv[1]), allowed_kinds=frozenset({'analysis'}),
    default_timeout_seconds=30.0)
m.submit_process_task(kind='analysis', owner='owner', request_id=None,
    task_path=sys.argv[2], payload={})
threading.Event().wait(60)
""",
            str(tmp_path),
            "tests.test_studio_jobs_orphan_lifetime:native_blocking_descendant_task"
            if native_descendant
            else "tests.test_studio_jobs_cancel_race:peer_cancel_process_task",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    group: int | None = None
    try:
        deadline = time.monotonic() + 5.0
        markers = list(tmp_path.glob("sj_*/worker.pid"))
        while not markers and time.monotonic() < deadline:
            assert child.poll() is None
            time.sleep(0.01)
            markers = list(tmp_path.glob("sj_*/worker.pid"))
        assert len(markers) == 1
        group = int(markers[0].read_text())
        assert os.getpgid(group) == group
        assert group_survivors(group)
        if native_descendant:
            descendant = int(markers[0].with_name("descendant.pid").read_text())
            assert os.getpgid(descendant) == group
            worker_stat = Path(f"/proc/{group}/stat")
            deadline = time.monotonic() + 3.0
            while worker_stat.read_text().rsplit(")", 1)[-1].split()[0] != "S":
                assert time.monotonic() < deadline
                time.sleep(0.01)
        child.kill()
        if collect_supervisor:
            child.wait(timeout=3.0)
        else:
            stat = Path(f"/proc/{child.pid}/stat")
            deadline = time.monotonic() + 3.0
            while stat.read_text().rsplit(")", 1)[-1].split()[0] != "Z":
                assert time.monotonic() < deadline
                time.sleep(0.01)
        deadline = time.monotonic() + 3.0
        while group_survivors(group) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not group_survivors(group), "Managed compute outlived its dead supervisor"
        if not collect_supervisor:
            assert stat.read_text().rsplit(")", 1)[-1].split()[0] == "Z"
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=3.0)
        if group is None:
            markers = list(tmp_path.glob("sj_*/worker.pid"))
            if len(markers) == 1:
                group = int(markers[0].read_text())
        if group is not None:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 3.0
            while group_survivors(group) and time.monotonic() < deadline:
                time.sleep(0.01)
            assert not group_survivors(group), "Test group survived cleanup"
