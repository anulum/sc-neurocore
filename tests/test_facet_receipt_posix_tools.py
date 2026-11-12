# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native tool identity through the public process runner

"""Verify Linux tool identity and startup refusal using real operating-system commands."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from textwrap import dedent
from pathlib import Path
from types import FrameType

import pytest

from tests.facet_receipt_budget_support import command_budget as host_command_budget
from tools.facet_receipt import RecordError, run_command


@pytest.fixture
def command_budget() -> float:
    """Measure actual recorder import cost and CPU load for finite caller budgets."""
    assert sys.platform.startswith("linux")
    started = time.monotonic()
    status, counts, outcome, _ = run_command(
        [sys.executable, "-c", "from tools.facet_receipt import run_command"], timeout=None
    )
    baseline = time.monotonic() - started
    assert status == 0 and counts["passed"] == 0 and outcome == "skipped"
    return host_command_budget(baseline)


@pytest.mark.parametrize(("program", "arguments"), [("sleep", ["0"]), ("true", [])])
def test_posix_command_identity_uses_a_successful_version_probe_without_validation_credit(
    program: str, arguments: list[str], command_budget: float
) -> None:
    """Native command metadata agrees with the installed tool, never a failed probe's error."""
    executable = shutil.which(program)
    assert executable is not None
    reference = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        timeout=command_budget,
        check=False,
    )
    lines = (reference.stdout or reference.stderr).strip().splitlines()
    expected = lines[0][:200] if reference.returncode == 0 and lines else ""
    status, counts, outcome, tool = run_command([executable, *arguments], timeout=command_budget)
    assert status == 0 and counts["passed"] == 0 and outcome == "skipped"
    assert tool["name"] == program
    assert tool["version"] == expected
    assert tool["stdout"] == tool["stderr"] == ""


@pytest.mark.parametrize("present", [False, True])
def test_posix_unavailable_or_invalid_native_program_refuses_before_artifacts(
    tmp_path: Path, command_budget: float, present: bool
) -> None:
    """Actual missing-file and executable-format errors remain authored startup refusals."""
    executable = tmp_path / "native-program"
    if present:
        executable.touch()
        executable.chmod(0o700)
    private = tmp_path / "raw"
    with pytest.raises(RecordError, match="validator process could not be started") as error:
        run_command([str(executable)], timeout=command_budget, raw_evidence_dir=private)
    assert str(executable) not in str(error.value)
    assert not private.exists()


def test_posix_exited_validator_preserves_sigchld_interruption_and_reaps_child(
    tmp_path: Path, command_budget: float
) -> None:
    """Preserve a real interruption when the owned validator has already exited."""
    child_pid = tmp_path / "exited-validator.pid"

    def interrupt_exited_validator(signum: int, frame: FrameType | None) -> None:
        """Interrupt only after the actual validator publishes its own process identity."""
        if child_pid.is_file():
            raise KeyboardInterrupt("validator exited during interruption")

    previous = signal.signal(signal.SIGCHLD, interrupt_exited_validator)
    try:
        with pytest.raises(KeyboardInterrupt, match="validator exited during interruption"):
            run_command(
                [
                    sys.executable,
                    "-c",
                    "import os,sys; from pathlib import Path; "
                    "Path(sys.argv[1]).write_text(str(os.getpid()))",
                    str(child_pid),
                ],
                timeout=command_budget,
            )
    finally:
        signal.signal(signal.SIGCHLD, previous)
    pid = int(child_pid.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert signal.getsignal(signal.SIGCHLD) == previous
    status, counts, outcome, _ = run_command([sys.executable, "-c", "pass"], timeout=command_budget)
    assert status == 0 and counts["passed"] == 0 and outcome == "skipped"


def test_posix_clock_resolution_budget_refuses_or_times_out_without_validation_credit(
    tmp_path: Path, command_budget: float
) -> None:
    """Exercise the smallest timer budget in an isolated caller and reap its killed children."""
    private = tmp_path / "raw"
    report = tmp_path / "clock-budget.json"
    caller = dedent(
        """\
        import json, os, sys, time
        from pathlib import Path
        from tools.facet_receipt import RecordError, run_command

        budget = float(sys.argv[1])
        resolution = time.get_clock_info("monotonic").resolution
        outcomes = []
        reaped = []
        for _ in range(len(os.sched_getaffinity(0))):
            try:
                status, counts, outcome, _ = run_command(
                    [sys.executable, "-c", "import sys,time;time.sleep(float(sys.argv[1]))", str(budget)],
                    timeout=resolution,
                    raw_evidence_dir=Path(sys.argv[3]),
                )
            except RecordError as error:
                if str(error) != "validator cleanup exceeded its process budget":
                    raise
                outcomes.append("cleanup-budget-refusal")
            else:
                if status != -1 or outcome != "timeout" or any(counts.values()):
                    raise RuntimeError((status, counts, outcome))
                outcomes.append("timeout")
            deadline = time.monotonic() + budget
            while True:
                try:
                    pid, status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    break
                if pid:
                    reaped.append(os.waitstatus_to_exitcode(status))
                elif time.monotonic() >= deadline:
                    raise RuntimeError("the isolated caller could not reap its killed child")
                else:
                    time.sleep(resolution)
        Path(sys.argv[2]).write_text(json.dumps({"outcomes": outcomes, "reaped": reaped}))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", caller, str(command_budget), str(report), str(private)],
        capture_output=True,
        text=True,
        timeout=command_budget,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(report.read_text())
    assert len(payload["outcomes"]) == len(os.sched_getaffinity(0))
    assert set(payload["outcomes"]) <= {"cleanup-budget-refusal", "timeout"}
    assert all(code == -signal.SIGKILL for code in payload["reaped"])
    assert not private.exists()
