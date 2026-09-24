# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bounded artifact read memory tests

"""Oversized files must fail by contract without first allocating their contents."""

import subprocess
import sys
from pathlib import Path

import pytest


_DRIVER = r"""
import os
import resource
import sys
import threading
from pathlib import Path
from sc_neurocore.studio.platform.jobs import StudioJobContext, StudioJobManager
from sc_neurocore.studio.platform.jobs_models import (
    STUDIO_SEED_INPUT_DIR, STUDIO_CONTROL_SEED_DIR, StudioJobArtifactUnavailable,
)

root, mode = Path(sys.argv[1]), sys.argv[2]
root.mkdir()
context = StudioJobContext(job_id="sj_fixture", work_dir=root,
    cancel_event=threading.Event(), max_artifact_bytes=1024)
manager = None
if mode == "declared":
    manager = StudioJobManager(root=root / "ledger", allowed_kinds=frozenset({"evidence"}),
        default_timeout_seconds=5.0, max_artifact_bytes=1024)
    def task(ctx):
        ctx.write_artifact("payload.bin", b"original")
        return {}
    job = manager.submit(kind="evidence", owner="studio-evidence", request_id=None, task=task)
    assert manager.wait(job.job_id, timeout_seconds=6.0).status == "completed"
    path = root / "ledger" / job.job_id / "payload.bin"
    read = lambda: manager.read_artifact(job.job_id, "payload.bin")
    expected = StudioJobArtifactUnavailable
elif mode == "publish":
    path = root / "payload.bin"
    read = lambda: context.publish_existing_artifact("payload.bin")
    expected = ValueError
else:
    directory = STUDIO_SEED_INPUT_DIR if mode == "seed" else STUDIO_CONTROL_SEED_DIR
    path = root / directory / "payload.bin"
    read = (lambda: context.read_seed_input("payload.bin")) if mode == "seed" else (
        lambda: context.read_control_seed("payload.bin"))
    expected = ValueError
path.parent.mkdir(parents=True, exist_ok=True)
if mode != "declared":
    path.write_bytes(b"x" * 1024)
    boundary = read()
    if mode == "publish":
        assert boundary.size_bytes == 1024
    else:
        assert boundary == b"x" * 1024
else:
    assert read().payload == b"original"
original_artifacts = context.artifacts
with path.open("wb") as output:
    output.truncate(64 * 1024 * 1024)
# Use current virtual size, after imports/threads, and leave enough headroom for
# small reads/errors but not for materializing the 64MiB sparse fixture.
virtual = int(Path("/proc/self/statm").read_text().split()[0]) * os.sysconf("SC_PAGE_SIZE")
_, hard = resource.getrlimit(resource.RLIMIT_AS)
limit = virtual + 16 * 1024 * 1024
if hard != resource.RLIM_INFINITY:
    limit = min(limit, hard)
resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
try:
    read()
except expected as exc:
    assert "size limit" in str(exc) or "integrity check" in str(exc), str(exc)
else:
    raise AssertionError("oversized file accepted")
assert context.artifacts == original_artifacts
if manager is not None:
    manager._ledger.close()
print("bounded rejection")
"""


@pytest.mark.parametrize("mode", ["seed", "control", "publish", "declared"])
def test_oversized_read_rejects_before_exhausting_memory(tmp_path: Path, mode: str) -> None:
    """Real public readers reject a sparse64MiB file with only16MiB headroom."""
    result = subprocess.run(
        [sys.executable, "-c", _DRIVER, str(tmp_path / "case"), mode],
        capture_output=True,
        text=True,
        timeout=25,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "bounded rejection"
