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

import os
from pathlib import Path
import subprocess  # nosec B404 - fixed argument vector, no shell.

from sc_neurocore.studio.platform.storage_launcher_configuration import LauncherConfiguration

BOOTSTRAP_MODULE = "sc_neurocore.studio.platform.storage_worker_bootstrap"


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
    command = [
        str(config.python_executable),
        "-s",
        "-B",
        "-m",
        BOOTSTRAP_MODULE,
        "--spool-root",
        str(config.spool_root),
        "--job",
        job_id,
        "--generation",
        generation,
        "--server-uid",
        str(config.api_uid),
        "--launcher-pid",
        str(os.getpid()),
        "--max-memory-bytes",
        str(config.max_memory_bytes),
        "--max-cpu-seconds",
        str(config.max_cpu_seconds),
        "--max-open-files",
        str(config.max_open_files),
        "--max-file-bytes",
        str(config.max_file_bytes),
        "--max-processes",
        str(config.max_processes),
    ]
    environment = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": os.pathsep.join(str(path) for path in config.python_path),
        "PYTHONDONTWRITEBYTECODE": "1",
        "LANG": "C.UTF-8",
    }
    return subprocess.Popen(  # nosec B603 - fixed argument vector, no shell.
        command,
        env=environment,
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
