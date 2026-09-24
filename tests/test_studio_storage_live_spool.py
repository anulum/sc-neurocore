# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — live worker directories of the API generation

"""The API reads a worker's live files and delivers control through its spool.

The spool is staged for real; the worker side is a real child interpreter
using the worker's own ``StudioJobContext`` to append events, consume the
command and read control seeds. Hostile entries are real links and pipes.
"""

from __future__ import annotations

from collections.abc import Iterator
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.jobs_models import (
    StudioJobArtifactUnavailable,
    StudioJobRejected,
)
from sc_neurocore.studio.platform.storage_live_spool import LiveSpools
from sc_neurocore.studio.platform.storage_spool_staging import StagedGeneration, stage_generation
from sc_neurocore.studio.platform.storage_worker_bootstrap import WorkerDescriptor
from tests.studio_seccomp_support import REPOSITORY

JOB = "sj_" + "4" * 16

WORKER = (
    "import json, sys, threading\n"
    "from pathlib import Path\n"
    "from sc_neurocore.studio.platform.jobs_context import StudioJobContext\n"
    "context = StudioJobContext(job_id=sys.argv[2], work_dir=Path(sys.argv[1]),\n"
    "    cancel_event=threading.Event(), max_artifact_bytes=1 << 20)\n"
    "command = context.poll_control_command()\n"
    "seed = context.read_control_seed('nested/weights.bin')\n"
    "context.append_artifact_event('events.jsonl', {'seen': command, 'seed': seed.hex()})\n"
    "print(json.dumps({'again': context.poll_control_command()}))\n"
)


@pytest.fixture
def staged(tmp_path: Path) -> Iterator[StagedGeneration]:
    spool = tmp_path / "spool"
    spool.mkdir()
    descriptor = WorkerDescriptor(
        version="studio.worker.descriptor.v1",
        job_id=JOB,
        generation="c" * 32,
        task_name="analysis.run",
        authorized_route="/api/analysis/jobs",
        supervisor=supervisor_identity(),
        max_artifact_bytes=1024,
    )
    with stage_generation(spool, descriptor, payload=b"{}", seeds={}, group=os.getgid()) as held:
        yield held


@pytest.fixture
def live() -> Iterator[LiveSpools]:
    spools = LiveSpools(retain=1, max_seed_bytes=64)
    try:
        yield spools
    finally:
        spools.close()


def test_a_real_worker_consumes_delivered_control_and_its_output_is_read_live(
    staged: StagedGeneration, live: LiveSpools
) -> None:
    """Command and seed arrive whole; appended events are read by offset."""
    live.attach(JOB, staged.work)
    live.deliver(JOB, b'{"action": "pause"}', {"nested/weights.bin": b"\x01\x02"})
    worker = subprocess.run(
        [sys.executable, "-c", WORKER, str(staged.path / JOB), JOB],
        capture_output=True,
        check=True,
        cwd=REPOSITORY,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY / "src")},
        text=True,
        timeout=120,
    )
    assert json.loads(worker.stdout.splitlines()[-1]) == {"again": None}
    # A second delivery reuses the staged control directories it created.
    live.deliver(JOB, b'{"action": "resume"}', {"nested/weights.bin": b"\x03"})
    control = staged.path / JOB
    assert (control / ".studio_control" / "command.json").read_bytes() == b'{"action": "resume"}'
    # Modes are explicit, whatever the API's umask: the compute group reads them.
    nested = control / ".studio_control_seed" / "nested"
    assert stat.S_IMODE(nested.stat().st_mode) == 0o2750
    for published in (nested / "weights.bin", control / ".studio_control" / "command.json"):
        assert stat.S_IMODE(published.stat().st_mode) == 0o640
    assert (control / ".studio_control_seed" / "nested" / "weights.bin").read_bytes() == b"\x03"
    first, offset = live.read(JOB, "events.jsonl", offset=0, max_bytes=8)
    rest, end = live.read(JOB, "events.jsonl", offset=offset, max_bytes=1 << 16)
    event = json.loads(first + rest)
    assert (offset, end) == (8, len(first + rest))
    assert event["seen"] == {"action": "pause"} and event["seed"] == "0102"
    assert live.read(JOB, "events.jsonl", offset=end, max_bytes=8) == (b"", end)
    assert live.read(JOB, "absent.jsonl", offset=3, max_bytes=8) == (b"", 3)


def test_finished_directories_are_kept_within_the_bound(
    staged: StagedGeneration, live: LiveSpools
) -> None:
    """A retired job stays readable until a later one displaces it."""
    (staged.path / JOB / "events.jsonl").write_bytes(b"done\n")
    live.attach(JOB, staged.work)
    live.attach(JOB, staged.work)
    live.retire(JOB)
    assert live.read(JOB, "events.jsonl", offset=0, max_bytes=64) == (b"done\n", 5)
    with pytest.raises(StudioJobRejected, match="unavailable"):
        live.deliver(JOB, b"{}", {})
    other = "sj_" + "5" * 16
    live.attach(other, staged.work)
    live.retire(other)
    assert live.read(JOB, "events.jsonl", offset=0, max_bytes=64) == (b"", 0)


@pytest.mark.parametrize("entry", ["link", "pipe", "linked-directory", "escape"])
def test_hostile_live_entries_are_unavailable(
    staged: StagedGeneration, live: LiveSpools, tmp_path: Path, entry: str
) -> None:
    """Links, pipes and escaping paths planted by the worker are never read."""
    work = staged.path / JOB
    secret = tmp_path / "secret"
    secret.write_bytes(b"api secret")
    path = "events.jsonl"
    if entry == "link":
        (work / "events.jsonl").symlink_to(secret)
    elif entry == "pipe":
        os.mkfifo(work / "events.jsonl")
    elif entry == "linked-directory":
        (work / "logs").symlink_to(tmp_path, target_is_directory=True)
        path = "logs/secret"
    else:
        path = "../../secret"
    live.attach(JOB, staged.work)
    with pytest.raises(StudioJobArtifactUnavailable):
        live.read(JOB, path, offset=0, max_bytes=64)


def test_delivery_bounds_and_ownership_are_enforced(
    staged: StagedGeneration, live: LiveSpools, tmp_path: Path
) -> None:
    """Oversized or escaping deliveries are refused; a replaced control directory too."""
    live.attach(JOB, staged.work)
    with pytest.raises(StudioJobRejected, match="command"):
        live.deliver(JOB, b"x" * (1024 * 1024 + 1), {})
    with pytest.raises(StudioJobRejected, match="seed input"):
        live.deliver(JOB, b"{}", {"big.bin": b"x" * 65})
    with pytest.raises(StudioJobRejected, match="escapes"):
        live.deliver(JOB, b"{}", {"../x": b""})
    control = staged.path / JOB / ".studio_control"
    control.rmdir()
    control.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(OSError):
        live.deliver(JOB, b"{}", {})
    assert not (tmp_path / "command.json").exists()
    with pytest.raises(ValueError):
        live.read(JOB, "events.jsonl", offset=-1, max_bytes=1)
    with pytest.raises(ValueError):
        live.read(JOB, "events.jsonl", offset=0, max_bytes=0)
    with pytest.raises(ValueError):
        LiveSpools(retain=-1, max_seed_bytes=1)
    with pytest.raises(KeyError):
        live.retire("sj_" + "6" * 16)
