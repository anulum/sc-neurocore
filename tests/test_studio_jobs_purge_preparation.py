# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Prepared purge directory custody

"""Refuse actual directory substitution after the purge intent becomes visible."""

from pathlib import Path

import pytest

from tests.studio_purge_child_support import PURGE_PROLOGUE
from tests.studio_seccomp_support import run_child


_OBSERVER_SOURCE = (
    "import json, sqlite3, sys\n"
    "connection = sqlite3.connect('file:' + sys.argv[1] + '?mode=ro',\n"
    "    uri=True, isolation_level=None, timeout=0.0)\n"
    "print(json.dumps({'ready': True}), flush=True)\n"
    "for line in sys.stdin:\n"
    "    try:\n"
    "        row = connection.execute('SELECT state FROM job_purges WHERE job_id=?',\n"
    "            (sys.argv[2],)).fetchone()\n"
    "        result = {'phase': None if row is None else row[0]}\n"
    "    except sqlite3.OperationalError as error:\n"
    "        result = {'error': str(error)}\n"
    "    print(json.dumps(result), flush=True)\n"
    "connection.close()\n"
)


@pytest.mark.parametrize("replacement", ["directory", "file", "missing", "stage"])
def test_prepared_purge_retains_record_when_its_directory_changes(
    tmp_path: Path, replacement: str
) -> None:
    """Hold real SQLite locking while replacing the recorded directory for real."""
    root = tmp_path / "jobs"
    retained = tmp_path / "original-directory"
    result = run_child(
        PURGE_PROLOGUE + "import select, subprocess\n"
        "job = finished_job('original evidence')\n"
        "original, retained = root / job, Path(sys.argv[2])\n"
        "shape = sys.argv[3]\n"
        "before = manager.record(job).to_public_dict()\n"
        "history = manager.transitions(job)\n"
        "observer = subprocess.Popen([sys.executable, '-u', '-c', sys.argv[4],\n"
        "    str(manager.ledger_path), job], stdin=subprocess.PIPE,\n"
        "    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)\n"
        "assert select.select([observer.stdout], [], [], 10.0)[0]\n"
        "assert json.loads(observer.stdout.readline()) == {'ready': True}\n"
        "replaced, observation_errors = [], []\n"
        "def decide(call):\n"
        "    if replaced or call.name != 'fcntl':\n"
        "        return None\n"
        "    if call.descriptor_path(0) not in {str(manager.ledger_path),\n"
        "            str(manager.ledger_path) + '-shm'}:\n"
        "        return None\n"
        "    observer.stdin.write('query\\n')\n"
        "    observer.stdin.flush()\n"
        "    if not select.select([observer.stdout], [], [], 1.0)[0]:\n"
        "        observation_errors.append('observer response timed out')\n"
        "        return None\n"
        "    reply = json.loads(observer.stdout.readline())\n"
        "    if 'error' in reply:\n"
        "        observation_errors.append(reply['error'])\n"
        "    if reply.get('phase') == 'prepared':\n"
        "        if shape == 'stage':\n"
        "            stage = root / ('.purge-' + job)\n"
        "            stage.mkdir()\n"
        "            (stage / 'foreign.txt').write_text('foreign evidence')\n"
        "        else:\n"
        "            original.rename(retained)\n"
        "        if shape == 'directory':\n"
        "            original.mkdir()\n"
        "            (original / 'foreign.txt').write_text('foreign evidence')\n"
        "        elif shape == 'file':\n"
        "            original.write_text('foreign evidence')\n"
        "        replaced.append(shape)\n"
        "    return None\n"
        "hold_system_calls(['fcntl'], decide)\n"
        "try:\n"
        "    manager.purge_terminal_record(job)\n"
        "    refusal = None\n"
        "except StudioJobRejected as error:\n"
        "    refusal = str(error)\n"
        "finally:\n"
        "    observer.stdin.close()\n"
        "    try:\n"
        "        observer.wait(timeout=5.0)\n"
        "    except subprocess.TimeoutExpired:\n"
        "        observer.kill()\n"
        "        observer.wait(timeout=5.0)\n"
        "assert observer.returncode == 0, observer.stderr.read()\n"
        "finish({'job': job, 'replaced': replaced, 'refusal': refusal,\n"
        "    'record_preserved': manager.record(job).to_public_dict() == before,\n"
        "    'history_preserved': manager.transitions(job) == history,\n"
        "    'pending': manager.status().pending_purge_count,\n"
        "    'observation_errors': observation_errors})\n",
        arguments=(str(root), str(retained), replacement, _OBSERVER_SOURCE),
    )
    assert result["replaced"] == [replacement]
    reason = {
        "directory": "Studio job purge directory identity changed.",
        "file": "Studio job purge target is not a directory.",
        "missing": "Studio job purge directory disappeared.",
        "stage": "Studio job has a pending purge requiring recovery.",
    }[replacement]
    assert result["refusal"] == reason
    assert result["record_preserved"] is True and result["history_preserved"] is True
    assert result["pending"] == 1
    original = root / str(result["job"])
    if replacement == "stage":
        assert (original / "proof.txt").read_text() == "original evidence"
        assert (
            root / (".purge-" + str(result["job"])) / "foreign.txt"
        ).read_text() == "foreign evidence"
        assert not retained.exists()
        return
    assert (retained / "proof.txt").read_text() == "original evidence"
    if replacement == "directory":
        assert (original / "foreign.txt").read_text() == "foreign evidence"
    elif replacement == "file":
        assert original.read_text() == "foreign evidence"
    else:
        assert not original.exists()
