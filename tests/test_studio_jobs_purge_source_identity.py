# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Purge source substitution custody

"""A renamed path is not evidence that the authorised directory was moved."""

from pathlib import Path

import pytest

from tests.studio_purge_child_support import PURGE_PROLOGUE
from tests.studio_seccomp_support import SECCOMP_AVAILABLE, run_child


@pytest.mark.skipif(not SECCOMP_AVAILABLE, reason="held system calls need Linux x86_64")
@pytest.mark.parametrize("restoring", [False, True])
def test_source_substitution_does_not_commit_purge_or_recovery(
    tmp_path: Path, restoring: bool
) -> None:
    """Preserve record and intent when the source changes immediately before the native move.

    The kernel holds the purge (or recovery) rename while its source directory
    is replaced for real by a foreign one.
    """
    root = tmp_path / "jobs"
    retained = tmp_path / "retained-owned"
    result = run_child(
        PURGE_PROLOGUE + "import subprocess\n"
        "from sc_neurocore.studio.platform import jobs_purge_paths\n"
        "job = finished_job('owned evidence')\n"
        "restoring, retained = sys.argv[2] == 'True', Path(sys.argv[3])\n"
        "records = [r.job_id for r in manager.list_records()]\n"
        "history = [dict(t) for t in manager.transitions(job)]\n"
        "original, stage = root / job, root / ('.purge-' + job)\n"
        "if restoring:\n"
        "    departed = subprocess.run([sys.executable, '-c', 'from sc_neurocore.studio.platform.'\n"
        "        'jobs_ledger_supervisor import supervisor_identity; print(supervisor_identity())'],\n"
        "        capture_output=True, text=True, check=True).stdout.strip()\n"
        "    identity = original.stat()\n"
        "    with manager._ledger.transaction() as connection:\n"
        "        connection.execute(\"INSERT INTO job_purges VALUES(?,?,?,?, 'prepared')\",\n"
        "            (job, departed, identity.st_dev, identity.st_ino))\n"
        "    assert jobs_purge_paths.move_without_replace(original, stage)\n"
        "source, target = (stage, original) if restoring else (original, stage)\n"
        "substituted = []\n"
        "def decide(call):\n"
        "    if (call.name == 'renameat2' and not substituted and call.text(1) == str(source)\n"
        "            and call.text(3) == str(target)):\n"
        "        substituted.append(True)\n"
        "        source.rename(retained)\n"
        "        source.mkdir()\n"
        "        (source / 'foreign.txt').write_text('foreign evidence')\n"
        "hold_system_calls(['renameat2'], decide)\n"
        "rejected = False\n"
        "if restoring:\n"
        "    manager.reconcile()\n"
        "else:\n"
        "    try:\n"
        "        manager.purge_terminal_record(job)\n"
        "    except StudioJobRejected:\n"
        "        rejected = True\n"
        "finish({'target': str(target), 'substituted': substituted, 'rejected': rejected,\n"
        "    'records': [r.job_id for r in manager.list_records()] == records,\n"
        "    'history': [dict(t) for t in manager.transitions(job)] == history,\n"
        "    'pending': manager.status().pending_purge_count})\n",
        arguments=(str(root), str(restoring), str(retained)),
    )
    assert result == {
        "target": result["target"],
        "substituted": [True],
        "rejected": not restoring,
        "records": True,
        "history": True,
        "pending": 1,
    }
    assert (retained / "proof.txt").read_text() == "owned evidence"
    assert (Path(str(result["target"])) / "foreign.txt").read_text() == "foreign evidence"
