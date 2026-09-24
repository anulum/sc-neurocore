# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — API staging of a launch generation's compute spool

"""The API stages exactly the spool the bootstrap reads, and adopts nothing.

Every case writes a real spool directory. Group ownership uses a real
supplementary group of this account when one exists, so ``fchown`` really
changes the group; the staged descriptor is read back by the bootstrap's own
reader, the consumer that runs under the compute identity.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat

import pytest

from sc_neurocore.studio.platform.jobs_ledger_supervisor import supervisor_identity
from sc_neurocore.studio.platform.storage_spool_staging import (
    CONTROL_SEED_MODE,
    DIRECTORY_MODE,
    FILE_MODE,
    WORK_DIRECTORY_MODE,
    stage_generation,
)
from sc_neurocore.studio.platform.storage_worker_bootstrap import (
    DESCRIPTOR_MAX_BYTES,
    WorkerDescriptor,
    read_worker_descriptor,
)

JOB = "sj_" + "8" * 16
GENERATION = "9" * 32
SEEDS = {"inputs/deep/data.bin": b"\x00\x01", "top.txt": b"top", "empty.bin": b""}


def _group() -> int:
    """A supplementary group other than the primary one, when the account has one."""
    others = [group for group in os.getgroups() if group != os.getegid()]
    return others[0] if others else os.getegid()


def _descriptor(generation: str = GENERATION, **changes: object) -> WorkerDescriptor:
    fields: dict[str, object] = {
        "version": "studio.worker.descriptor.v1",
        "job_id": JOB,
        "generation": generation,
        "task_name": "analysis.run",
        "authorized_route": "/api/analysis/jobs",
        "supervisor": supervisor_identity(),
        "max_artifact_bytes": 1024,
    }
    fields.update(changes)
    return WorkerDescriptor.model_validate(fields, strict=True)


def _open_descriptors() -> int:
    return len(os.listdir("/proc/self/fd"))


@pytest.fixture
def spool(tmp_path: Path) -> Path:
    root = tmp_path / "spool"
    root.mkdir()
    return root


def test_staged_generation_is_what_the_bootstrap_reads(spool: Path) -> None:
    """Inputs, worker directory and seeds carry the exact owner, group and modes."""
    group = _group()
    before = _open_descriptors()
    with stage_generation(
        spool, _descriptor(), payload=b'{"x": 1}', seeds=SEEDS, group=group
    ) as staged:
        assert staged.path == spool / JOB / GENERATION
        held, named = os.fstat(staged.work), (staged.path / JOB).stat()
        assert (held.st_dev, held.st_ino) == (named.st_dev, named.st_ino)
        assert _open_descriptors() == before + 2
    assert _open_descriptors() == before
    expected = {
        spool / JOB: DIRECTORY_MODE,
        staged.path: DIRECTORY_MODE,
        staged.path / "input": DIRECTORY_MODE,
        staged.path / JOB: WORK_DIRECTORY_MODE,
        staged.path / JOB / ".studio_seed": DIRECTORY_MODE,
        staged.path / JOB / ".studio_control": WORK_DIRECTORY_MODE,
        staged.path / JOB / ".studio_control_seed": CONTROL_SEED_MODE,
        staged.path / JOB / ".studio_seed" / "inputs" / "deep": DIRECTORY_MODE,
        staged.path / "input" / "descriptor.json": FILE_MODE,
        staged.path / "input" / "payload.json": FILE_MODE,
        staged.path / JOB / ".studio_seed" / "top.txt": FILE_MODE,
    }
    for path, mode in expected.items():
        metadata = path.stat(follow_symlinks=False)
        assert (metadata.st_uid, metadata.st_gid) == (os.geteuid(), group), path
        assert stat.S_IMODE(metadata.st_mode) == mode, path
    for name, payload in SEEDS.items():
        assert (staged.path / JOB / ".studio_seed" / name).read_bytes() == payload
    assert (staged.path / "input" / "payload.json").read_bytes() == b'{"x": 1}'
    assert (
        read_worker_descriptor(spool, job_id=JOB, generation=GENERATION, server_uid=os.getuid())
        == _descriptor()
    )


def test_a_later_generation_reuses_only_an_identical_job_directory(spool: Path) -> None:
    """A second generation of the job stages beside the first; a changed job directory refuses."""
    group = _group()
    stage_generation(spool, _descriptor(), payload=b"{}", seeds={}, group=group).close()
    second = stage_generation(spool, _descriptor("a" * 32), payload=b"{}", seeds={}, group=group)
    second.close()
    assert sorted(path.name for path in (second.path / JOB).iterdir()) == [
        ".studio_control",
        ".studio_control_seed",
    ]
    with pytest.raises(FileExistsError):
        stage_generation(spool, _descriptor(), payload=b"{}", seeds={}, group=group)
    (spool / JOB).chmod(0o770)
    before = _open_descriptors()
    with pytest.raises(PermissionError, match="unexpected ownership or mode"):
        stage_generation(spool, _descriptor("b" * 32), payload=b"{}", seeds={}, group=group)
    assert _open_descriptors() == before
    assert not (spool / JOB / ("b" * 32)).exists()


@pytest.mark.parametrize(
    "seeds",
    [
        {"../escape": b"x"},
        {"a//b": b"x"},
        {"a/./b": b"x"},
        {"line\nbreak": b"x"},
        {"/absolute": b"x"},
        {"a": b"x", "a/b": b"y"},
    ],
    ids=["traversal", "empty-part", "dot-part", "unprintable", "absolute", "file-as-directory"],
)
def test_seed_paths_stay_canonical_inside_the_seed_directory(
    spool: Path, seeds: dict[str, bytes]
) -> None:
    """Escaping, non-canonical or colliding seed paths refuse without leaking descriptors."""
    before = _open_descriptors()
    with pytest.raises((ValueError, NotADirectoryError)):
        stage_generation(spool, _descriptor(), payload=b"{}", seeds=seeds, group=_group())
    assert _open_descriptors() == before
    assert not (spool.parent / "escape").exists()


def test_linked_or_relative_roots_and_foreign_groups_are_refused(
    spool: Path, tmp_path: Path
) -> None:
    """The root is never followed through a link; a group the API is not in refuses."""
    link = tmp_path / "link"
    link.symlink_to(spool, target_is_directory=True)
    with pytest.raises(OSError):
        stage_generation(link, _descriptor(), payload=b"{}", seeds={}, group=_group())
    with pytest.raises(ValueError, match="absolute"):
        stage_generation(Path("spool"), _descriptor(), payload=b"{}", seeds={}, group=_group())
    foreign = next(group for group in range(1, 1 << 16) if group not in os.getgroups())
    before = _open_descriptors()
    with pytest.raises(PermissionError):
        stage_generation(spool, _descriptor(), payload=b"{}", seeds={}, group=foreign)
    assert _open_descriptors() == before
    assert list(spool.iterdir()) == []


def test_largest_descriptor_fits_the_bootstrap_limit() -> None:
    """Field bounds keep every staged descriptor below the bootstrap's read limit."""
    widest = _descriptor(
        task_name="t" * 128,
        authorized_route="r" * 256,
        supervisor="h" * 253 + ":" + "9" * 10 + ":" + "9" * 20,
        max_artifact_bytes=1 << 40,
    )
    encoded = json.dumps(widest.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    assert len(encoded) < DESCRIPTOR_MAX_BYTES // 2
