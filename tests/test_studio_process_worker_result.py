# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — process worker result publication

"""A worker killed while writing its result leaves no result, never a torn one.

The supervisor reads the result after stopping the worker. The writer runs in
a real child interpreter; the kernel holds its first write to the partial file
and the child is killed there with SIGKILL.
"""

from __future__ import annotations

import json
from pathlib import Path
import signal

import pytest

from tests.studio_seccomp_support import SECCOMP_AVAILABLE, run_child

WRITER = (
    "import os, signal, sys\n"
    "from pathlib import Path\n"
    "from sc_neurocore.studio.platform.process_worker import _write_result\n"
    "from tests.studio_syscall_support import finish, hold_system_calls\n"
    "target, kill = Path(sys.argv[1]), sys.argv[2] == 'kill'\n"
    "def decide(call):\n"
    "    if kill and call.descriptor_path(0).endswith('.partial'):\n"
    "        os.kill(os.getpid(), signal.SIGKILL)\n"
    "hold_system_calls(['write'], decide)\n"
    "_write_result(target, status='completed', result={'x': 1}, error=None, context=None)\n"
    "finish({'written': True})\n"
)


@pytest.mark.skipif(not SECCOMP_AVAILABLE, reason="held system calls need Linux x86_64")
def test_a_worker_killed_while_writing_publishes_no_result(tmp_path: Path) -> None:
    """SIGKILL during the write leaves only the partial file; the result is absent."""
    target = tmp_path / ".studio_process_result.json"
    run_child(WRITER, arguments=(str(target), "kill"), expected_returncode=-signal.SIGKILL)
    assert not target.exists()
    assert (tmp_path / ".studio_process_result.json.partial").exists()


@pytest.mark.skipif(not SECCOMP_AVAILABLE, reason="held system calls need Linux x86_64")
def test_a_completed_write_replaces_the_partial_file(tmp_path: Path) -> None:
    """A finished write publishes the whole result under its final name only."""
    target = tmp_path / ".studio_process_result.json"
    assert run_child(WRITER, arguments=(str(target), "keep")) == {"written": True}
    assert json.loads(target.read_text()) == {
        "artifacts": [],
        "error": None,
        "result": {"x": 1},
        "status": "completed",
    }
    assert sorted(path.name for path in tmp_path.iterdir()) == [target.name]
