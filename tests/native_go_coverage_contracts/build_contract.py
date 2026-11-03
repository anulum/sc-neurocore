# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real native command custody contracts

"""Exercise actual child outcomes, artifact preservation and bounded cleanup."""

from __future__ import annotations

import json
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

import pytest

from tools.go_mat_coverage_build import NativeGo


@pytest.mark.parametrize("status", [0, 3])
def test_command_preserves_actual_child_status_and_stdio(
    native_builder: NativeGo, status: int
) -> None:
    """A real Python child supplies its own observable terminal and output bytes."""
    actual = native_builder.command(
        "real-child",
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import sys;print('real stdout');print('real stderr',file=sys.stderr);"
            "sys.exit(int(sys.argv[1]))",
            str(status),
        ],
        native_builder.job,
    )
    directory = native_builder.job / "commands/real-child"
    receipt = json.loads((directory / "receipt.json").read_text())
    assert actual == status == receipt["returncode"]
    assert receipt["pid"] > 0 and receipt["process_reaped"] is True
    assert receipt["timed_out"] is False and receipt["operation_error"] is None
    assert (directory / "stdout.log").read_bytes() == b"real stdout\n"
    assert (directory / "stderr.log").read_bytes() == b"real stderr\n"
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with pytest.raises(FileExistsError):
        native_builder.command("real-child", [sys.executable, "--version"], native_builder.job)
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}


@pytest.mark.parametrize("status", [0, 4])
def test_require_judges_actual_success_without_losing_refusal(
    native_builder: NativeGo, status: int
) -> None:
    """The checked command path keeps the real refusal receipt before raising."""
    args = [sys.executable, "-I", "-B", "-c", "import sys;sys.exit(int(sys.argv[1]))", str(status)]
    if status:
        with pytest.raises(RuntimeError, match="actual status 4"):
            native_builder.require("checked-child", args, native_builder.job)
    else:
        native_builder.require("checked-child", args, native_builder.job)
    receipt = json.loads((native_builder.job / "commands/checked-child/receipt.json").read_text())
    assert receipt["returncode"] == status and receipt["process_reaped"] is True


def test_missing_executable_has_no_invented_child_status(native_builder: NativeGo) -> None:
    """An actual spawn failure preserves null PID/status and its real OS exception."""
    with pytest.raises(FileNotFoundError):
        native_builder.command(
            "absent-child", [str(native_builder.job / "absent")], native_builder.job
        )
    receipt = json.loads((native_builder.job / "commands/absent-child/receipt.json").read_text())
    assert receipt["pid"] is None and receipt["returncode"] is None
    assert receipt["process_reaped"] is False and "FileNotFoundError" in receipt["operation_error"]
    with pytest.raises(RuntimeError, match="No actual native process terminal"):
        native_builder.retained_terminal_status(None)


def test_output_receipt_collision_refuses_before_launch(native_builder: NativeGo) -> None:
    """A declared native output cannot enter the command namespace through a relative path."""
    output = native_builder.job / "artifacts/../commands/collision/receipt.json"
    with pytest.raises(ValueError, match="collides"):
        native_builder.command(
            "collision", [str(native_builder.go), "version"], native_builder.job, outputs=(output,)
        )
    assert not (native_builder.job / "commands/collision").exists()


def test_bad_label_and_expired_deadline_launch_nothing(native_builder: NativeGo) -> None:
    """Invalid custody names and exhausted time never become native command results."""
    with pytest.raises(ValueError, match="label"):
        native_builder.command(
            "../foreign", [str(native_builder.go), "version"], native_builder.job
        )
    native_builder.deadline = time.monotonic()
    with pytest.raises(TimeoutError, match="exhausted"):
        native_builder.command("expired", [str(native_builder.go), "version"], native_builder.job)
    assert list((native_builder.job / "commands").iterdir()) == []


def test_real_deadline_stops_and_reaps_its_actual_child(native_builder: NativeGo) -> None:
    """An actual blocked child receives the existing admitted stop budget and is reaped."""
    remaining = native_builder.deadline - time.monotonic()
    assert remaining > 0
    native_builder.deadline = time.monotonic() + min(native_builder.stop_seconds, remaining / 2)
    with pytest.raises(TimeoutError, match="exceeded"):
        native_builder.command(
            "blocked-child",
            [sys.executable, "-I", "-B", "-c", "import signal;signal.pause()"],
            native_builder.job,
        )
    receipt = json.loads((native_builder.job / "commands/blocked-child/receipt.json").read_text())
    assert receipt["pid"] > 0 and receipt["returncode"] < 0
    assert receipt["timed_out"] is True and receipt["process_reaped"] is True
    assert not Path("/proc", str(receipt["pid"])).exists()


def test_real_termination_ignoring_child_is_killed_and_reaped(native_builder: NativeGo) -> None:
    """A ready finite owned child proves the admitted stop deadline escalates to actual SIGKILL."""
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            "import signal;signal.signal(signal.SIGTERM,signal.SIG_IGN);"
            "print('ready',flush=True);signal.pause()",
        ],
        cwd=native_builder.job,
        env=native_builder.environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        assert process.stdout is not None
        remaining = native_builder.deadline - time.monotonic()
        assert remaining > native_builder.stop_seconds
        ready = select.select(
            [process.stdout], [], [], min(remaining, native_builder.stop_seconds)
        )[0]
        assert ready and process.stdout.readline() == b"ready\n"
        with pytest.raises(RuntimeError, match="No actual native process terminal"):
            native_builder.retained_terminal_status(process)
    finally:
        native_builder.stop_child_group(process)
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
    assert process.returncode == -signal.SIGKILL
    assert not Path("/proc", str(process.pid)).exists()


def test_absent_counter_directories_refuse_before_covdata_launch(native_builder: NativeGo) -> None:
    """The real producer rejects an empty merge set without creating native command custody."""
    with pytest.raises(ValueError, match="No actual native counter directories"):
        native_builder.profiles([], "empty")
    assert list((native_builder.job / "commands").iterdir()) == []


@pytest.mark.parametrize(
    "fault", ["jobs", "deadline", "nonfinite-deadline", "stop", "nonfinite-stop"]
)
def test_invalid_producer_budgets_refuse_before_output_creation(
    native_builder: NativeGo,
    tmp_path: Path,
    fault: str,
) -> None:
    """The producer never interprets invalid profile inputs as an execution grant."""
    jobs = 0 if fault == "jobs" else native_builder.jobs
    deadline = (
        time.monotonic()
        if fault == "deadline"
        else float("nan")
        if fault == "nonfinite-deadline"
        else native_builder.deadline
    )
    stop = (
        0.0
        if fault == "stop"
        else float("inf")
        if fault == "nonfinite-stop"
        else native_builder.stop_seconds
    )
    output = tmp_path / "invalid"
    output.mkdir()
    with pytest.raises(ValueError):
        NativeGo(
            output,
            native_builder.repository,
            native_builder.go,
            deadline,
            jobs,
            native_builder.environment,
            stop,
            native_builder.python_coverage_config,
        )
    assert list(output.iterdir()) == []
