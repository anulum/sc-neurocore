# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — child-interpreter purge scenarios

"""Shared prologue for purge scenarios run in a child with held system calls.

The child owns a real manager over ``sys.argv[1]``, finishes one job and then
installs its own held calls with :func:`tests.studio_syscall_support.hold_system_calls`.
The parent test inspects the same root afterwards through a fresh manager.
"""

from __future__ import annotations

from pathlib import Path

from sc_neurocore.studio.platform.jobs import StudioJobManager

PURGE_PROLOGUE = (
    "import errno, json, os, signal, sys\n"
    "from pathlib import Path\n"
    "from sc_neurocore.studio.platform.jobs import StudioJobManager\n"
    "from sc_neurocore.studio.platform.jobs_models import StudioJobRejected\n"
    "from tests.studio_syscall_support import finish, hold_system_calls\n"
    "root = Path(sys.argv[1])\n"
    "def open_manager():\n"
    "    return StudioJobManager(root=root, allowed_kinds=frozenset({'analysis'}),\n"
    "        default_timeout_seconds=3.0)\n"
    "manager = open_manager()\n"
    "def finished_job(proof=None):\n"
    "    def task(context):\n"
    "        if proof is not None:\n"
    "            context.write_artifact('proof.txt', proof)\n"
    "        return {}\n"
    "    job = manager.submit(kind='analysis', owner='owner', request_id=None, task=task)\n"
    "    assert manager.wait(job.job_id, 10.0).status == 'completed'\n"
    "    assert manager._done_events[job.job_id].wait(5.0)\n"
    "    return job.job_id\n"
)


def reopened(root: Path) -> StudioJobManager:
    """Open a fresh manager over a root a child interpreter used."""
    return StudioJobManager(
        root=root, allowed_kinds=frozenset({"analysis"}), default_timeout_seconds=3.0
    )
