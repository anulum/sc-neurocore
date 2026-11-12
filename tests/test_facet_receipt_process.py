# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native recorder subprocess lifetime contracts

"""Exercise actual recorder subprocess cleanup on the current operating system."""

from __future__ import annotations

import _thread
import ctypes
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from tools.facet_receipt import run_command


def _require_process_exit(pid: int) -> None:
    """Observe the exact recorded child using the native process interface."""
    if sys.platform == "win32":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        kernel.WaitForSingleObject.restype = ctypes.c_uint32
        kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        kernel.CloseHandle.restype = ctypes.c_int
        handle = kernel.OpenProcess(0x00100000, False, pid)
        if handle:
            try:
                assert kernel.WaitForSingleObject(handle, 0) == 0
            finally:
                assert kernel.CloseHandle(handle)
        else:
            assert ctypes.get_last_error() == 87
    else:
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


@pytest.mark.parametrize("termination", ["interruption", "timeout"])
def test_native_recorder_reaps_the_started_child(tmp_path: Path, termination: str) -> None:
    """Interrupt or time out a real child and require its native exit without platform mocks."""
    started = time.monotonic()
    baseline_status, baseline_counts, baseline_outcome, _ = run_command(
        [sys.executable, "-c", "pass"], timeout=None
    )
    baseline = time.monotonic() - started
    assert baseline_status == 0
    assert baseline_counts["passed"] == 0
    assert baseline_outcome == "skipped"
    cpus = os.cpu_count() or 1
    load = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
    pressure = max(1.0, load / cpus) if load is not None else 1.0
    budget = min(4 * 60 * 60, baseline * (1 + cpus) * pressure)
    interval = baseline / (1 + cpus)
    pid_file = tmp_path / "child.pid"
    completed = tmp_path / "child.completed"
    program = (
        "import os,sys,time; from pathlib import Path; "
        "Path(sys.argv[1]).write_text(str(os.getpid())); "
        "print('started',flush=True); time.sleep(float(sys.argv[3])); "
        "Path(sys.argv[2]).write_text('completed')"
    )
    command = [sys.executable, "-c", program, str(pid_file), str(completed), str(2 * budget)]
    stopped = threading.Event()
    observed = threading.Event()
    deadline = time.monotonic() + budget

    def interrupt_started_child() -> None:
        """Signal the actual calling thread only after its subprocess reports startup."""
        while not stopped.wait(interval) and time.monotonic() < deadline:
            if pid_file.is_file() and pid_file.read_text():
                observed.set()
                _thread.interrupt_main()
                return

    control = threading.Thread(target=interrupt_started_child)
    try:
        if termination == "interruption":
            control.start()
            with pytest.raises(KeyboardInterrupt):
                run_command(command, timeout=budget)
        else:
            status, counts, outcome, tool = run_command(command, timeout=budget)
            assert status == -1
            assert outcome == "timeout"
            assert all(value == 0 for value in counts.values())
            assert "started" in tool["stdout"]
    finally:
        stopped.set()
        if control.ident is not None:
            control.join(timeout=budget)
            assert not control.is_alive()
    assert termination == "timeout" or observed.is_set()
    pid = int(pid_file.read_text())
    _require_process_exit(pid)
    assert not completed.exists()
    (tmp_path / "native-process-control.json").write_text(
        json.dumps(
            {
                "platform": sys.platform,
                "termination": termination,
                "child_pid": pid,
                "command": command,
                "measured_baseline_seconds": baseline,
                "cpu_count": cpus,
                "load_1m": load,
                "budget_seconds": budget,
                "poll_interval_seconds": interval,
                "native_process_exit_verified": True,
                "scientific_receipt_credit": False,
            },
            indent=2,
        )
        + "\n"
    )
