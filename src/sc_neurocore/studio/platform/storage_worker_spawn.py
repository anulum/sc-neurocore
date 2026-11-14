# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — fixed worker bootstrap spawn

"""Start the fixed worker bootstrap under the compute identity, and check the host.

The argument vector and environment are built only from operator
configuration and the validated job generation. A privileged launcher drops
supplementary groups and switches real and effective user and group IDs to
the compute identity before the bootstrap executes; the bootstrap then forbids
privilege gain and applies the configured ceilings itself.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
import signal
import subprocess  # nosec B404 - fixed argument vector, no shell.

from sc_neurocore.refusals import AuthoredRefusal
from sc_neurocore.studio.platform.storage_launcher_configuration import LauncherConfiguration

BOOTSTRAP_MODULE = "sc_neurocore.studio.platform.storage_worker_bootstrap"
#: How long the start check waits for the interpreter to import the bootstrap.
INTERPRETER_CHECK_SECONDS = 60.0


class WorkerInterpreterRefused(AuthoredRefusal):
    """The configured worker interpreter cannot run the bootstrap; the text names it and why."""


def generation_spool_ready(root: Path, job_id: str, generation: str, owner: int) -> bool:
    """Return whether the API-prepared job and generation directories exist.

    Parameters
    ----------
    root : Path
        Configured compute spool root.
    job_id, generation : str
        Validated identifiers from the launcher request.
    owner : int
        Configured API identity that must own both directories.

    Returns
    -------
    bool
        ``True`` only when both components are real directories owned by
        ``owner``; symbolic links and missing entries yield ``False``.
    """
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    opened: list[int] = []
    try:
        opened.append(os.open(root, flags))
        for name in (job_id, generation):
            opened.append(os.open(name, flags, dir_fd=opened[-1]))
            if os.fstat(opened[-1]).st_uid != owner:
                return False
        return True
    except OSError:
        return False
    finally:
        for descriptor in reversed(opened):
            os.close(descriptor)


def compute_identity_processes(uid: int) -> tuple[int, ...]:
    """Return PIDs of processes whose ``/proc`` entry is owned by ``uid``.

    A process that exits during the scan is skipped.
    """
    found: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid == uid:
                found.append(int(entry.name))
        except OSError:
            continue
    return tuple(found)


def spawn_worker_bootstrap(
    config: LauncherConfiguration, *, job_id: str, generation: str, privileged: bool
) -> subprocess.Popen[bytes]:
    """Start one fixed bootstrap for an exact job generation.

    Parameters
    ----------
    config : LauncherConfiguration
        Operator configuration supplying interpreter, import roots, spool,
        API identity, compute identity and ceilings.
    job_id, generation : str
        Validated identifiers of the admitted job generation.
    privileged : bool
        Whether this launcher switches to the configured compute identity.

    Returns
    -------
    subprocess.Popen[bytes]
        The bootstrap, leading its own session, with no inherited descriptors
        and standard streams connected to ``/dev/null``.

    Raises
    ------
    OSError
        The interpreter cannot be executed or the identity switch fails.
    """
    return subprocess.Popen(
        _bootstrap_command(config, job_id=job_id, generation=generation),
        env=_bootstrap_environment(config),
        cwd="/",
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
        user=config.worker_uid if privileged else None,
        group=config.worker_gid if privileged else None,
        extra_groups=[] if privileged else None,
    )


def _bootstrap_command(config: LauncherConfiguration, *, job_id: str, generation: str) -> list[str]:
    """Return the fixed argument vector of one bootstrap.

    Parameters
    ----------
    config : LauncherConfiguration
        Operator configuration supplying interpreter, spool, API identity and ceilings.
    job_id, generation : str
        Validated identifiers of the admitted job generation.

    Returns
    -------
    list[str]
        The interpreter with ``-s -B -m`` and the bootstrap, then each option and its value.
    """
    options: tuple[tuple[str, object], ...] = (
        ("--spool-root", config.spool_root),
        ("--job", job_id),
        ("--generation", generation),
        ("--server-uid", config.api_uid),
        ("--launcher-pid", os.getpid()),
        ("--max-memory-bytes", config.max_memory_bytes),
        ("--max-cpu-seconds", config.max_cpu_seconds),
        ("--max-open-files", config.max_open_files),
        ("--max-file-bytes", config.max_file_bytes),
        ("--max-processes", config.max_processes),
    )
    command = [str(config.python_executable), "-s", "-B", "-m", BOOTSTRAP_MODULE]
    for option, value in options:
        command += [option, str(value)]
    return command


def _bootstrap_environment(config: LauncherConfiguration) -> dict[str, str]:
    """Return the fixed environment the worker bootstrap runs in.

    Parameters
    ----------
    config : LauncherConfiguration
        Operator configuration supplying the import roots and any event input.

    Returns
    -------
    dict[str, str]
        A system PATH, the configured import roots, no bytecode, a UTF-8 locale and
        the event input's own variables.
    """
    environment = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": os.pathsep.join(str(path) for path in config.python_path),
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": "C.UTF-8",
    }
    if config.event_input is not None:
        environment.update(config.event_input.environment())
    return environment


def check_worker_interpreter(
    config: LauncherConfiguration,
    *,
    privileged: bool,
    timeout: float = INTERPRETER_CHECK_SECONDS,
) -> None:
    """Refuse a worker interpreter that cannot import the bootstrap as a worker will run it.

    The bootstrap runs with ``-s``, so packages in the user site are not seen,
    and with only the configured import roots. An interpreter that cannot import
    it would let every launch fail before the worker asks for its grant. The
    check imports the bootstrap once, with the worker's flags, environment and,
    for a privileged launcher, its identity, so the launcher can refuse to start
    and say why.

    Parameters
    ----------
    config : LauncherConfiguration
        Operator configuration supplying the interpreter, import roots and identity.
    privileged : bool
        Whether this launcher switches to the configured compute identity.
    timeout : float
        Seconds to wait for the import.

    Raises
    ------
    WorkerInterpreterRefused
        The interpreter cannot be run, fails the import (the last line of its error
        output, or its exit status, is named), or does not finish within ``timeout``.
    """
    interpreter = config.python_executable
    try:
        status, error_output = _import_bootstrap(config, privileged=privileged, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise WorkerInterpreterRefused(
            f"worker interpreter {interpreter} did not import the worker bootstrap "
            f"within {timeout:g} s"
        ) from None
    except OSError as exc:
        raise WorkerInterpreterRefused(
            f"worker interpreter {interpreter} cannot be run: {exc.strerror or exc}"
        ) from None
    if status != 0:
        lines = [line.strip() for line in error_output.splitlines() if line.strip()]
        reason = lines[-1] if lines else f"exit status {status}"
        raise WorkerInterpreterRefused(
            f"worker interpreter {interpreter} cannot import the worker bootstrap with -s: {reason}"
        )


def _import_bootstrap(
    config: LauncherConfiguration, *, privileged: bool, timeout: float
) -> tuple[int, str]:
    """Import the bootstrap as a worker would, in a session of its own.

    Error output is decoded with replacement characters, and at the deadline the
    whole session is killed, so nothing the interpreter started outlives it.

    Parameters
    ----------
    config : LauncherConfiguration
        Operator configuration supplying the interpreter, import roots and identity.
    privileged : bool
        Whether to run as the configured compute identity.
    timeout : float
        Seconds to wait for the import.

    Returns
    -------
    tuple[int, str]
        The exit status and the error output.

    Raises
    ------
    OSError
        The interpreter cannot be executed or the identity switch fails.
    subprocess.TimeoutExpired
        The import did not finish in time; the session has been killed.
    """
    with subprocess.Popen(
        [str(config.python_executable), "-s", "-B", "-c", f"import {BOOTSTRAP_MODULE}"],
        env=_bootstrap_environment(config),
        cwd="/",
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        errors="replace",
        close_fds=True,
        start_new_session=True,
        user=config.worker_uid if privileged else None,
        group=config.worker_gid if privileged else None,
        extra_groups=[] if privileged else None,
    ) as process:
        try:
            _, error_output = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_session(process)
            raise
    return process.returncode, error_output


def _kill_session(process: subprocess.Popen[str]) -> None:
    """Kill every process of the session a process leads.

    Only a leader not yet waited for is passed: until it is reaped, its process
    group id cannot be reused by another process.

    Parameters
    ----------
    process : subprocess.Popen[str]
        The session leader.
    """
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
