# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Actual process-state observation under resource exhaustion

"""Distinguish real process exit from an unavailable kernel observation."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time

from sc_neurocore.studio.platform.jobs_process_state import process_exited
from tests.studio_seccomp_support import run_child


def test_descriptor_exhaustion_does_not_confirm_a_live_process_exited() -> None:
    """Keep the actual caller live when its own descriptor limit prevents inspection."""
    result = run_child(
        "import json, os, resource\n"
        "from sc_neurocore.studio.platform.jobs_process_state import process_exited\n"
        "pid = os.getpid()\n"
        "before = process_exited(pid)\n"
        "limits = resource.getrlimit(resource.RLIMIT_NOFILE)\n"
        "try:\n"
        "    resource.setrlimit(resource.RLIMIT_NOFILE, (0, limits[1]))\n"
        "    exhausted = process_exited(pid)\n"
        "finally:\n"
        "    resource.setrlimit(resource.RLIMIT_NOFILE, limits)\n"
        "print(json.dumps({'before': before, 'exhausted': exhausted,\n"
        "    'still_executing': True,\n"
        "    'limits_restored': resource.getrlimit(resource.RLIMIT_NOFILE) == limits}))\n"
    )
    assert result == {
        "before": False,
        "exhausted": False,
        "still_executing": True,
        "limits_restored": True,
    }


def test_a_real_zombie_and_a_reaped_process_are_confirmed_exited() -> None:
    """Observe an actual child before and after its parent collects the exit status."""
    child = subprocess.Popen([sys.executable, "-B", "-c", "pass"])
    try:
        status = Path(f"/proc/{child.pid}/stat")
        deadline = time.monotonic() + 30
        while status.read_text().rsplit(")", 1)[-1].split()[0] != "Z":
            assert time.monotonic() < deadline, "The actual child did not finish."
            time.sleep(0.01)
        assert process_exited(child.pid) is True
        assert child.wait(timeout=5) == 0
        assert not status.exists()
        assert process_exited(child.pid) is True
        assert process_exited(os.getpid()) is False
    finally:
        child.kill()
        child.wait(timeout=5)
