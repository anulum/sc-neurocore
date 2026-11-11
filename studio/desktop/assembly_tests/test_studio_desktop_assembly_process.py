# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Desktop assembly command and process lifetime contracts

"""Exercise real command output, refusals and descendant cleanup during assembly."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.studio_desktop_assembly_process import run_assembly_command


def test_command_preserves_arguments_and_binary_output(
    tmp_path: Path, assembly_seconds: float
) -> None:
    """Receipts and the public result retain exact arguments and non-UTF8 output."""
    arguments = [
        sys.executable,
        "-I",
        "-B",
        "-c",
        "import os,sys; os.write(1,b'\\xff'+sys.argv[1].encode()); os.write(2,b'\\xfe')",
        "value with spaces and 'quotes'",
    ]
    receipts = tmp_path / "command"
    result = run_assembly_command(arguments, assembly_seconds, receipts)
    assert result.returncode == 0
    assert result.args == arguments
    assert result.stdout == b"\xffvalue with spaces and 'quotes'"
    assert result.stderr == b"\xfe"
    assert (receipts / "stdout").read_bytes() == result.stdout
    assert (receipts / "stderr").read_bytes() == result.stderr
    assert json.loads((receipts / "command.json").read_text()) == arguments
    record = json.loads((receipts / "receipt.json").read_text())
    assert record["timed_out"] is False
    assert record["cleanup"]["reaped"] is True
    assert record["cleanup"]["returncode"] == 0
    assert record["cleanup"]["survivor_count"] == 0


def test_failed_command_preserves_original_status_and_diagnostic(
    tmp_path: Path,
    assembly_seconds: float,
) -> None:
    """A real failed producer propagates its status and both original byte streams."""
    receipts = tmp_path / "failed"
    with pytest.raises(subprocess.CalledProcessError) as raised:
        run_assembly_command(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                "import os; os.write(1,b'partial'); os.write(2,b'failure'); raise SystemExit(7)",
            ],
            assembly_seconds,
            receipts,
        )
    assert raised.value.returncode == 7
    assert raised.value.stdout == b"partial"
    assert raised.value.stderr == b"failure"
    record = json.loads((receipts / "receipt.json").read_text())
    assert record["timed_out"] is False
    assert record["cleanup"]["returncode"] == 7
    assert record["cleanup"]["reaped"] is True
    assert record["cleanup"]["survivor_count"] == 0


@pytest.mark.parametrize("parent_exits", [False, True])
def test_descendants_are_reaped_on_deadline_and_parent_exit(
    tmp_path: Path, parent_exits: bool, assembly_seconds: float
) -> None:
    """Neither a timeout nor a successful parent leaves a child producer running."""
    receipts = tmp_path / "process-tree"
    child = f"import time; time.sleep({assembly_seconds * 2!r})"
    parent = (
        "import subprocess,sys,time; "
        f"child=subprocess.Popen([sys.executable,'-I','-B','-c',{child!r}]); "
        "print(child.pid,flush=True); "
        + ("raise SystemExit(0)" if parent_exits else f"time.sleep({assembly_seconds * 2!r})")
    )
    arguments = [sys.executable, "-I", "-B", "-c", parent]
    if parent_exits:
        result = run_assembly_command(arguments, assembly_seconds, receipts)
        assert result.returncode == 0
    else:
        with pytest.raises(subprocess.TimeoutExpired) as raised:
            run_assembly_command(arguments, assembly_seconds / 100, receipts)
        assert raised.value.output == (receipts / "stdout").read_bytes()
    child_pid = int((receipts / "stdout").read_text().strip())
    record = json.loads((receipts / "receipt.json").read_text())
    assert record["timed_out"] is not parent_exits
    assert record["cleanup"]["reaped"] is True
    assert record["cleanup"]["survivor_count"] == 0
    status = Path(f"/proc/{child_pid}/stat")
    if status.exists():
        assert status.read_text().split(")", 1)[1].split()[0] in {"Z", "X", "x"}


@pytest.mark.parametrize("deadline", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_invalid_deadline_refuses_before_creating_receipts(tmp_path: Path, deadline: float) -> None:
    """An invalid deadline cannot start work or create a misleading command receipt."""
    receipts = tmp_path / "invalid"
    with pytest.raises(ValueError, match="finite positive deadline"):
        run_assembly_command([sys.executable], deadline, receipts)
    assert not receipts.exists()


def test_empty_command_refuses_before_creating_receipts(
    tmp_path: Path, assembly_seconds: float
) -> None:
    """A missing command is a refused input rather than a successful empty job."""
    receipts = tmp_path / "empty"
    with pytest.raises(ValueError, match="need arguments"):
        run_assembly_command([], assembly_seconds, receipts)
    assert not receipts.exists()


def test_start_failure_retains_command_and_refuses_existing_receipts(
    tmp_path: Path,
    assembly_seconds: float,
) -> None:
    """A missing executable records its exact invocation and cannot overwrite custody."""
    receipts = tmp_path / "missing"
    arguments = [str(tmp_path / "missing-executable")]
    with pytest.raises(FileNotFoundError):
        run_assembly_command(arguments, assembly_seconds, receipts)
    assert json.loads((receipts / "command.json").read_text()) == arguments
    assert not (receipts / "receipt.json").exists()
    with pytest.raises(FileExistsError):
        run_assembly_command([sys.executable], assembly_seconds, receipts)
    assert json.loads((receipts / "command.json").read_text()) == arguments


def test_explicit_child_environment_is_used_without_recording_values(
    tmp_path: Path, assembly_seconds: float
) -> None:
    """The real child receives its declared environment while custody retains only output."""
    receipts = tmp_path / "explicit-environment"
    marker = "value with spaces and 'quotes'"
    result = run_assembly_command(
        [sys.executable, "-I", "-B", "-c", "import os; print(os.environ['STUDIO_CHILD_VALUE'])"],
        assembly_seconds,
        receipts,
        environment={"STUDIO_CHILD_VALUE": marker},
    )
    assert result.stdout == (marker + "\n").encode()
    assert result.stderr == b""
    assert marker not in (receipts / "command.json").read_text()
    assert marker not in (receipts / "receipt.json").read_text()


def test_kernel_refused_cleanup_is_reported_before_independent_reaping(
    tmp_path: Path, assembly_seconds: float
) -> None:
    """A real kernel signal refusal must retain a failed custody record and never claim success."""
    support = Path(
        os.environ.get(
            "SC_NEUROCORE_STUDIO_ASSEMBLY_REFUSAL_SUPPORT",
            str(Path(__file__).resolve().parents[3] / "tests/studio_seccomp_support.py"),
        )
    ).resolve(strict=True)
    receipts = tmp_path / "unreaped-command"
    source = (
        "import errno,json,os,runpy,signal,sys\n"
        "from pathlib import Path\n"
        "from tools.studio_desktop_assembly_process import run_assembly_command\n"
        "support=runpy.run_path(sys.argv[1])\n"
        "support['install_refusals']([support['Refusal']('kill',errno.EPERM)])\n"
        "root=Path(sys.argv[2])\n"
        "try:\n"
        "    run_assembly_command([sys.executable,'-I','-B','-c',"
        "'import time; time.sleep('+sys.argv[3]+')'],0.2,root)\n"
        "except RuntimeError as error:\n"
        "    receipt=json.loads((root/'receipt.json').read_text())\n"
        "    worker=receipt['cleanup']['group_id']\n"
        "    descriptor=os.pidfd_open(worker)\n"
        "    try:\n"
        "        signal.pidfd_send_signal(descriptor,signal.SIGKILL)\n"
        "        waited,status=os.waitpid(worker,0)\n"
        "    finally:\n"
        "        os.close(descriptor)\n"
        "    print(json.dumps({'error':str(error),'waited':waited,"
        "'signal':os.WTERMSIG(status),'receipt':receipt}))\n"
        "else:\n"
        "    raise AssertionError('A kernel-refused cleanup was reported as success.')\n"
    )
    observed = run_assembly_command(
        [
            sys.executable,
            "-B",
            "-c",
            source,
            str(support),
            str(receipts),
            str(assembly_seconds * 2),
        ],
        assembly_seconds,
        tmp_path / "kernel-refusal",
    )
    value = json.loads(observed.stdout)
    assert value["error"] == "The assembly command group could not be fully reaped."
    assert value["receipt"]["timed_out"] is True
    cleanup = value["receipt"]["cleanup"]
    assert cleanup["reaped"] is False
    assert cleanup["outcome"] == "unreaped"
    assert cleanup["returncode"] is None
    assert cleanup["survivor_count"] == 1
    assert value["waited"] == cleanup["group_id"]
    assert value["signal"] == 9
    assert not Path(f"/proc/{value['waited']}").exists()
