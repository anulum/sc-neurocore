# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — direct-spawn worker launcher

"""The real launcher service starts, replays and stops exact worker generations.

The launcher runs as its own process through its command-line entry, because
it makes itself a child subreaper and kills unattributed adopted processes.
These same-identity runs establish function only: an unprivileged launcher
starts workers under its own UID and never qualifies isolation. Distinct-UID
refusal and success belong to the disposable three-identity proof.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
from pydantic import ValidationError

from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.storage_launcher_client import (
    exchange_launcher_request,
    new_launcher_request,
)
from sc_neurocore.studio.platform.storage_launcher_protocol import (
    LauncherOperation,
)
from sc_neurocore.studio.platform.storage_transport import write_frame
from sc_neurocore.studio.platform.storage_launcher_configuration import LauncherConfiguration
from sc_neurocore.studio.platform.storage_worker_spawn import (
    BOOTSTRAP_MODULE,
    WorkerInterpreterRefused,
    check_worker_interpreter,
    compute_identity_processes,
)
from sc_neurocore.studio.platform.storage_worker_grant import (
    GRANT_ENDPOINT_NAME,
    ExpectedWorker,
    WorkerGrantEndpoint,
)
from tests.studio_storage_launcher_support import *


@pytest.fixture
def base() -> Iterator[Path]:
    """Short private base directory so every socket path fits the Unix limit."""
    with launcher_base() as path:
        yield path


@pytest.fixture
def launcher(base: Path) -> Iterator[Launcher]:
    """Yield a running same-identity launcher and stop it with SIGTERM afterwards."""
    running = start(base)
    try:
        yield running
    finally:
        shutdown(running)


def test_launched_generation_runs_the_named_task_after_grant(launcher: Launcher) -> None:
    """Launch, grant and completion of one exact generation through the real service."""
    directory = prepare(launcher.spool_root, JOB_A, GEN_A)
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    registered: list[str] = []
    try:
        with WorkerGrantEndpoint(descriptor, directory, GRANT_ENDPOINT_NAME) as endpoint:
            started = send(launcher, "launch", JOB_A, GEN_A)
            assert started.state == "running" and started.pid is not None
            assert started.start_token is not None
            endpoint.grant(
                ExpectedWorker(uid=os.getuid(), pid=started.pid, start_token=started.start_token),
                deadline=time.monotonic() + 60.0,
                max_refusals=1,
                register=registered.append,
            )
    finally:
        os.close(descriptor)
    finished = await_state(launcher, JOB_A, GEN_A, "stopped")
    assert (finished.state, finished.exit_status) == ("stopped", 1)
    assert (finished.pid, finished.start_token) == (started.pid, started.start_token)
    assert registered == [
        f"{supervisor_identity().split(':')[0]}:{started.pid}:{started.start_token}"
    ]
    evidence = json.loads((directory / JOB_A / ".studio_process_result.json").read_text())
    # The worker's result file is private custody and keeps the fault's own text.
    assert evidence["error"] == "invalid_analysis_payload"


def test_retry_and_status_resolve_a_lost_launch_reply(launcher: Launcher) -> None:
    """Retrying the same generation returns the recorded worker, never a second one."""
    prepare(launcher.spool_root, JOB_A, GEN_A)
    first = send(launcher, "launch", JOB_A, GEN_A)
    retried = send(launcher, "launch", JOB_A, GEN_A)
    status = send(launcher, "status", JOB_A, GEN_A)
    assert first.state == "running"
    for response in (retried, status):
        assert (response.pid, response.start_token) == (first.pid, first.start_token)
    other = send(launcher, "launch", JOB_A, GEN_B)
    assert (other.state, other.reason) == ("refused", "conflict")
    stopped = send(launcher, "stop", JOB_A, GEN_A)
    assert (stopped.state, stopped.pid, stopped.exit_status) == ("stopped", first.pid, -9)
    repeated = send(launcher, "stop", JOB_A, GEN_A)
    assert (repeated.state, repeated.exit_status) == ("stopped", -9)


def test_unknown_generation_is_absent(launcher: Launcher) -> None:
    """Status and stop for a generation the launcher never started report absence."""
    operations: tuple[LauncherOperation, ...] = ("status", "stop")
    for operation in operations:
        response = send(launcher, operation, JOB_B, GEN_B)
        assert response.state == "absent" and response.pid is None


def test_missing_spool_refuses_launch(launcher: Launcher) -> None:
    """No API-prepared generation directory means no worker is started."""
    response = send(launcher, "launch", JOB_B, GEN_B)
    assert (response.state, response.reason) == ("refused", "spool")
    assert send(launcher, "status", JOB_B, GEN_B).state == "absent"


def test_stop_kills_a_live_worker_tree(launcher: Launcher) -> None:
    """A worker waiting for its grant is stopped and confirmed gone."""
    prepare(launcher.spool_root, JOB_A, GEN_A)
    started = send(launcher, "launch", JOB_A, GEN_A)
    assert started.state == "running" and started.pid is not None
    stopped = send(launcher, "stop", JOB_A, GEN_A)
    assert (stopped.state, stopped.pid) == ("stopped", started.pid)
    assert not Path(f"/proc/{started.pid}").exists()
    assert send(launcher, "status", JOB_A, GEN_A).state == "stopped"


def test_worker_capacity_is_enforced(base: Path) -> None:
    """With one worker slot a second live generation is refused."""
    running = start(base, max_workers=1)
    try:
        prepare(running.spool_root, JOB_A, GEN_A)
        prepare(running.spool_root, JOB_B, GEN_B)
        assert send(running, "launch", JOB_A, GEN_A).state == "running"
        refused = send(running, "launch", JOB_B, GEN_B)
        assert (refused.state, refused.reason) == ("refused", "capacity")
        assert send(running, "stop", JOB_A, GEN_A).state == "stopped"
        admitted = send(running, "launch", JOB_B, GEN_B)
        assert admitted.state == "running"
        assert send(running, "stop", JOB_B, GEN_B).state == "stopped"
    finally:
        shutdown(running)


def test_record_budget_retires_only_stopped_generations(base: Path) -> None:
    """The oldest stopped record makes room; a live one never does."""
    running = start(base, max_records=1)
    try:
        prepare(running.spool_root, JOB_A, GEN_A)
        prepare(running.spool_root, JOB_B, GEN_B)
        assert send(running, "launch", JOB_A, GEN_A).state == "running"
        refused = send(running, "launch", JOB_B, GEN_B)
        assert (refused.state, refused.reason) == ("refused", "capacity")
        assert send(running, "stop", JOB_A, GEN_A).state == "stopped"
        assert send(running, "launch", JOB_B, GEN_B).state == "running"
        assert send(running, "status", JOB_A, GEN_A).state == "absent"
        assert send(running, "stop", JOB_B, GEN_B).state == "stopped"
    finally:
        shutdown(running)


def test_a_launcher_whose_interpreter_cannot_run_refuses_to_start(base: Path) -> None:
    """A worker interpreter that cannot be run is refused when the launcher starts, by name.

    Parameters
    ----------
    base : Path
        The launcher's private base folder.
    """
    missing = base / "missing-python"
    with pytest.raises(AssertionError) as refused:
        start(base, python_executable=str(missing))
    assert f"worker interpreter {missing} cannot be run: No such file or directory" in str(
        refused.value
    )
    assert not (base / "sock" / "launcher.sock").exists()


def test_an_interpreter_that_cannot_import_the_bootstrap_refuses_the_start(base: Path) -> None:
    """An interpreter whose environment lacks the bootstrap's packages is refused with its own error.

    This is the failure that once surfaced only as a grant that never came: the
    worker runs with ``-s``, so packages in the user site are not seen.

    Parameters
    ----------
    base : Path
        The launcher's private base folder.
    """
    interpreter = base / "no-packages-python"
    interpreter.write_text(
        "#!/bin/sh\n"
        "echo 'Traceback (most recent call last):' >&2\n"
        "echo \"ModuleNotFoundError: No module named 'pydantic'\" >&2\n"
        "exit 1\n",
        encoding="utf-8",
    )
    interpreter.chmod(0o755)
    with pytest.raises(AssertionError) as refused:
        start(base, python_executable=str(interpreter))
    assert (
        f"worker interpreter {interpreter} cannot import the worker bootstrap with -s: "
        "ModuleNotFoundError: No module named 'pydantic'"
    ) in str(refused.value)


def _interpreter(base: Path, name: str, script: str) -> LauncherConfiguration:
    """Return the launcher configuration of a base with a scripted stand-in interpreter.

    Parameters
    ----------
    base : Path
        The launcher's private base folder.
    name : str
        The stand-in's file name.
    script : str
        Its shell script body.

    Returns
    -------
    LauncherConfiguration
        The default configuration with that interpreter.
    """
    interpreter = base / name
    interpreter.write_text("#!/bin/sh\n" + script, encoding="utf-8")
    interpreter.chmod(0o755)
    return LauncherConfiguration.model_validate_json(
        json.dumps(configuration(base, python_executable=str(interpreter))), strict=True
    )


def test_an_interpreter_that_hangs_is_refused_at_its_deadline(base: Path) -> None:
    """The start check does not wait for an interpreter that never finishes its import.

    Parameters
    ----------
    base : Path
        The launcher's private base folder.
    """
    config = _interpreter(base, "hanging-python", "exec sleep 60\n")
    started = time.monotonic()
    with pytest.raises(WorkerInterpreterRefused) as refused:
        check_worker_interpreter(config, privileged=False, timeout=0.5)
    # Far below the stand-in's 60 s and the module's own 60 s default, and far
    # above 0.5 s, so a loaded host cannot make it fail.
    assert time.monotonic() - started < 10.0
    assert str(refused.value) == (
        f"worker interpreter {config.python_executable} did not import the worker bootstrap "
        "within 0.5 s"
    )


def _gone(pid: int) -> bool:
    """Return whether a process no longer runs: absent, or a zombie awaiting its parent.

    Parameters
    ----------
    pid : int
        The process.

    Returns
    -------
    bool
        True when it no longer runs.
    """
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except FileNotFoundError:
        return True
    return stat.rsplit(")", 1)[1].split()[0] == "Z"


def test_what_a_hanging_interpreter_started_does_not_outlive_the_refusal(base: Path) -> None:
    """At the deadline the check's whole session is killed, not only the interpreter.

    Parameters
    ----------
    base : Path
        The launcher's private base folder.
    """
    child = base / "child.pid"
    config = _interpreter(
        base, "forking-python", f'sleep 30 &\necho $! > "{child}"\nexec sleep 60\n'
    )
    # Seconds, not the half second of the hang test: the stand-in must write the
    # child's pid before the deadline, also on a loaded host.
    with pytest.raises(WorkerInterpreterRefused):
        check_worker_interpreter(config, privileged=False, timeout=5.0)
    pid = int(child.read_text(encoding="utf-8"))
    deadline = time.monotonic() + 10.0
    while not _gone(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _gone(pid)


def test_error_output_that_is_not_utf8_is_refused_in_words(base: Path) -> None:
    """An executable that writes bytes no codec reads is refused, its bytes replaced.

    Parameters
    ----------
    base : Path
        The launcher's private base folder.
    """
    config = _interpreter(
        base, "binary-python", "printf '\\377\\376 not an interpreter\\n' >&2\nexit 2\n"
    )
    with pytest.raises(WorkerInterpreterRefused) as refused:
        check_worker_interpreter(config, privileged=False)
    assert str(refused.value) == (
        f"worker interpreter {config.python_executable} cannot import the worker bootstrap "
        "with -s: �� not an interpreter"
    )


def test_an_interpreter_that_fails_silently_is_refused_with_its_status(base: Path) -> None:
    """With no error output, the refusal names the exit status.

    Parameters
    ----------
    base : Path
        The launcher's private base folder.
    """
    config = _interpreter(base, "silent-python", "exit 3\n")
    with pytest.raises(WorkerInterpreterRefused) as refused:
        check_worker_interpreter(config, privileged=False)
    assert str(refused.value) == (
        f"worker interpreter {config.python_executable} cannot import the worker bootstrap "
        "with -s: exit status 3"
    )


def test_the_project_interpreter_passes_the_check(base: Path) -> None:
    """The interpreter that runs these tests imports the bootstrap with -s and the checkout's src.

    Parameters
    ----------
    base : Path
        The launcher's private base folder.
    """
    check_worker_interpreter(
        LauncherConfiguration.model_validate_json(json.dumps(configuration(base)), strict=True),
        privileged=False,
    )


#: Variables a POSIX shell may add to the environment it passes on.
_SHELL_ADDED = frozenset({"PWD", "OLDPWD", "SHLVL", "_"})


def test_the_check_runs_the_interpreter_as_a_worker_runs_it(base: Path) -> None:
    """The check's flags, working directory and environment are the worker's, not the launcher's.

    A stand-in records what it was given, then runs the real interpreter. An
    import that passes only with the launcher's environment, its working
    directory or the user site would let the check pass for an interpreter
    every worker then fails with.

    Parameters
    ----------
    base : Path
        The launcher's private base folder.
    """
    record = base / "record.json"
    recorder = (
        "import json, os, sys; json.dump({'args': sys.argv[2:], 'cwd': os.getcwd(), "
        "'env': dict(os.environ)}, open(sys.argv[1], 'w', encoding='utf-8'))"
    )
    config = _interpreter(
        base,
        "recording-python",
        f'"{sys.executable}" -I -c "{recorder}" "{record}" "$@"\nexec "{sys.executable}" "$@"\n',
    )
    check_worker_interpreter(config, privileged=False)
    seen = json.loads(record.read_text(encoding="utf-8"))
    assert seen["args"] == ["-s", "-B", "-c", f"import {BOOTSTRAP_MODULE}"]
    assert seen["cwd"] == "/"
    environment = {key: value for key, value in seen["env"].items() if key not in _SHELL_ADDED}
    assert environment == {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": os.pathsep.join(str(path) for path in config.python_path),
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": "C.UTF-8",
    }


def test_unstartable_bootstrap_is_reported_unavailable(base: Path) -> None:
    """A spawn failure after a good start is a refusal, not an invented worker identity.

    The interpreter passes the start check, then disappears before the launch. It
    is a script that runs this interpreter, not a link to it: a link outside the
    virtual environment would start the base interpreter without its packages,
    which the start check refuses.

    Parameters
    ----------
    base : Path
        The launcher's private base folder.
    """
    interpreter = base / "vanishing-python"
    interpreter.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    interpreter.chmod(0o755)
    running = start(base, python_executable=str(interpreter))
    try:
        interpreter.unlink()
        prepare(running.spool_root, JOB_A, GEN_A)
        refused = send(running, "launch", JOB_A, GEN_A)
        assert (refused.state, refused.reason, refused.pid) == ("refused", "unavailable", None)
    finally:
        shutdown(running)


def test_compute_identity_scan_reports_live_processes_of_that_identity() -> None:
    """The start-up scan finds this identity's processes and nobody else's."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        mine = compute_identity_processes(os.getuid())
        assert {os.getpid(), child.pid} <= set(mine)
        foreign = compute_identity_processes(os.getuid() + 1_000_000)
        assert not {os.getpid(), child.pid} & set(foreign)
    finally:
        child.kill()
        child.wait(timeout=10.0)


def test_non_api_peer_and_malformed_frames_get_no_reply(base: Path) -> None:
    """Refused peers and malformed requests are closed; the service keeps serving."""
    running = start(base, api_uid=os.getuid() + 1)
    try:
        request = new_launcher_request("status", job_id=JOB_A, generation=GEN_A)
        with pytest.raises((EOFError, ConnectionError, OSError)):
            exchange_launcher_request(
                running.socket_path,
                request,
                launcher_uid=os.getuid(),
                deadline=time.monotonic() + 10.0,
            )
    finally:
        shutdown(running)
    served = start(base)
    try:
        for payload in (b"{}", b"not json"):
            channel = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            with channel:
                channel.connect(str(served.socket_path))
                write_frame(channel, payload, max_bytes=1024, deadline=time.monotonic() + 5.0)
                channel.settimeout(10.0)
                assert channel.recv(16) == b""
        assert send(served, "status", JOB_A, GEN_A).state == "absent"
    finally:
        shutdown(served)


def test_client_refuses_endpoint_not_owned_by_the_launcher(launcher: Launcher) -> None:
    """The API sends nothing to an endpoint owned by another identity."""
    request = new_launcher_request("status", job_id=JOB_A, generation=GEN_A)
    with pytest.raises(PermissionError):
        exchange_launcher_request(
            launcher.socket_path,
            request,
            launcher_uid=os.getuid() + 1,
            deadline=time.monotonic() + 10.0,
        )
    with pytest.raises(TimeoutError, match="already expired"):
        exchange_launcher_request(
            launcher.socket_path, request, launcher_uid=os.getuid(), deadline=time.monotonic()
        )
    with pytest.raises(ValidationError):
        new_launcher_request("status", job_id="sj_bad", generation=GEN_A)
