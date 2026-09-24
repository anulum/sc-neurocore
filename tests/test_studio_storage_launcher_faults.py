# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — worker launcher under real kernel refusals

"""A launcher facing kernel refusals never claims custody it does not hold.

Each case runs the real launcher in a child interpreter; at the marked point
a seccomp filter makes the kernel refuse one system call, as a hardened
service manager, a security module or a kernel without pidfd support does.
Privileged start-up checks need a real root launcher and distinct compute
identities; they are qualified by the three-identity proof, not here.
"""

from __future__ import annotations

from collections.abc import Iterator
import errno
import json
from pathlib import Path

import pytest

from tests.studio_seccomp_support import SECCOMP_AVAILABLE, Refusal, run_refused
from tests.studio_storage_launcher_support import *

pytestmark = pytest.mark.skipif(not SECCOMP_AVAILABLE, reason="seccomp filters need Linux x86_64")

HARNESS = (
    "import os, signal\n"
    "from pathlib import Path\n"
    "from sc_neurocore.studio.platform.storage_worker_launcher import WorkerLauncher\n"
    "from sc_neurocore.studio.platform.storage_launcher_configuration import (\n"
    "    load_launcher_configuration)\n"
    "from sc_neurocore.studio.platform.storage_launcher_client import new_launcher_request\n"
    "from sc_neurocore.studio.platform.storage_worker_grant import (\n"
    "    GRANT_ENDPOINT_NAME, WorkerGrantEndpoint)\n"
    "fault, config_path, job, generation = sys.argv[1:5]\n"
    "config = load_launcher_configuration(Path(config_path))\n"
    "service = WorkerLauncher(config)\n"
    "outcome = {}\n"
    "def request(operation):\n"
    "    reply = service.handle(new_launcher_request(operation, job_id=job, generation=generation))\n"
    "    return [reply.state, reply.reason]\n"
    "if fault == 'mode-change':\n"
    "    install_refusals(REFUSALS)\n"
    "    try:\n"
    "        service.start()\n"
    "    except PermissionError as refused:\n"
    "        outcome['start'] = refused.errno\n"
    "    outcome['left'] = sorted(os.listdir(config.socket_path.parent))\n"
    "else:\n"
    "    service.start()\n"
    "    if fault == 'double-start':\n"
    "        try:\n"
    "            service.start()\n"
    "        except RuntimeError as refused:\n"
    "            outcome['start'] = str(refused)\n"
    "    if fault == 'pidfd':\n"
    "        install_refusals(REFUSALS)\n"
    "        outcome['launch'] = request('launch')\n"
    "    if fault == 'undeliverable-stop':\n"
    "        # An API endpoint that never grants keeps the worker waiting, alive.\n"
    "        spool = config.spool_root / job / generation\n"
    "        held = os.open(spool, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)\n"
    "        endpoint = WorkerGrantEndpoint(held, spool, GRANT_ENDPOINT_NAME)\n"
    "        endpoint.open()\n"
    "        outcome['launch'] = request('launch')\n"
    "        leader = service.handle(\n"
    "            new_launcher_request('status', job_id=job, generation=generation)).pid\n"
    "        install_refusals(REFUSALS)\n"
    "        outcome['stop'] = request('stop')\n"
    "        outcome['again'] = request('stop')\n"
    "        os.kill(leader, signal.SIGKILL)\n"
    "        outcome['after_exit'] = request('stop')\n"
    "        endpoint.close()\n"
    "        os.close(held)\n"
    "    children = []\n"
    "    for name in filter(str.isdigit, os.listdir('/proc')):\n"
    "        try:\n"
    "            with open(f'/proc/{name}/stat') as handle:\n"
    "                fields = handle.read().rsplit(')', 1)[-1].split()\n"
    "        except (FileNotFoundError, ProcessLookupError):\n"
    "            continue\n"
    "        if fields[1] == str(os.getpid()) and fields[0] != 'Z':\n"
    "            children.append(name)\n"
    "    outcome['children'] = children\n"
    "    service.stop()\n"
    "print(json.dumps(outcome))\n"
)


def harness(base: Path, fault: str, refusals: list[Refusal]) -> dict[str, object]:
    config_path = base / "launcher.json"
    config_path.write_text(json.dumps(configuration(base)))
    prepare(base / "spool", JOB_A, GEN_A)
    return run_refused(HARNESS, refusals, arguments=(fault, str(config_path), JOB_A, GEN_A))


@pytest.fixture
def base() -> Iterator[Path]:
    """Short private base directory so every socket path fits the Unix limit."""
    with launcher_base() as path:
        yield path


def test_second_start_is_refused(base: Path) -> None:
    """One launcher object owns at most one endpoint."""
    result = harness(base, "double-start", [])
    assert result == {"start": "worker launcher is already started", "children": []}


def test_refused_mode_change_leaves_no_launcher_socket(base: Path) -> None:
    """A kernel-refused mode change after bind closes and unlinks the endpoint."""
    result = harness(
        base,
        "mode-change",
        [Refusal("fchmodat", errno.EPERM), Refusal("fchmodat2", errno.EPERM)],
    )
    assert result == {"start": errno.EPERM, "left": []}


def test_worker_without_pidfd_custody_is_killed_and_unavailable(base: Path) -> None:
    """On a kernel without pidfds the spawned worker is killed, never reported running."""
    result = harness(base, "pidfd", [Refusal("pidfd_open", errno.ENOSYS)])
    assert result == {"launch": ["refused", "unavailable"], "children": []}


def test_undeliverable_stop_reports_survivors_until_the_tree_is_gone(base: Path) -> None:
    """A stop the kernel will not deliver keeps custody and says so, every time.

    Once the generation really exits, the same stop is confirmed.
    """
    result = harness(base, "undeliverable-stop", [Refusal("pidfd_send_signal", errno.EPERM)])
    assert result == {
        "launch": ["running", None],
        "stop": ["refused", "survivors"],
        "again": ["refused", "survivors"],
        "after_exit": ["stopped", None],
        "children": [],
    }
