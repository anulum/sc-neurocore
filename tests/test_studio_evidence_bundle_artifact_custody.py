# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evidence bundle source artifact custody

"""Bind exported bytes to the source record, not merely to reader assertions."""

import json
import threading
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.evidence_bundle import write_studio_evidence_bundle
from sc_neurocore.studio.platform.jobs import (
    StudioJobArtifactPayload,
    StudioJobContext,
    StudioJobManager,
    StudioJobRecord,
)


@pytest.fixture
def source(tmp_path: Path) -> Iterator[tuple[StudioJobManager, StudioJobRecord]]:
    """Publish actual bytes through a completed ledger-backed job."""
    manager = StudioJobManager(
        root=tmp_path / "source",
        default_timeout_seconds=5.0,
        allowed_kinds=frozenset({"evidence"}),
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        context.write_artifact("output.bin", b"source bytes")
        return {}

    job = manager.submit(kind="evidence", owner="studio-evidence", request_id=None, task=task)
    record = manager.wait(job.job_id, timeout_seconds=6.0)
    assert record.status == "completed"
    try:
        yield manager, record
    finally:
        manager._ledger.close()


def _context(tmp_path: Path) -> StudioJobContext:
    return StudioJobContext(
        job_id="sj_bundle",
        work_dir=tmp_path / "bundle",
        cancel_event=threading.Event(),
        max_artifact_bytes=100_000,
    )


def test_bundle_preserves_declared_source_bytes(
    tmp_path: Path, source: tuple[StudioJobManager, StudioJobRecord]
) -> None:
    """Real reader and writer preserve the source declaration and payload."""
    manager, record = source
    result = write_studio_evidence_bundle(
        _context(tmp_path), job_records=(record,), artifact_reader=manager.read_artifact
    )
    base = tmp_path / "bundle" / "evidence" / "jobs" / record.job_id
    assert (base / "artifacts/output.bin").read_bytes() == b"source bytes"
    saved = json.loads((base / "record.json").read_text())
    assert saved == record.to_public_dict()
    entries = result.manifest["entries"]
    assert isinstance(entries, list)
    artifact_entries = [e for e in entries if isinstance(e, dict) and e["type"] == "job_artifact"]
    assert len(artifact_entries) == 1
    assert artifact_entries[0]["sha256"] == record.artifacts[0].sha256


@pytest.mark.parametrize("field", ["sha256", "size_bytes", "relative_path"])
def test_bundle_rejects_reader_metadata_disagreement(
    tmp_path: Path, source: tuple[StudioJobManager, StudioJobRecord], field: str
) -> None:
    """A reader returning a different declaration cannot certify this record."""
    manager, record = source
    original = manager.read_artifact(record.job_id, "output.bin")
    artifact = original.artifact
    if field == "sha256":
        artifact = replace(artifact, sha256="0" * 64)
    elif field == "size_bytes":
        artifact = replace(artifact, size_bytes=artifact.size_bytes + 1)
    else:
        artifact = replace(artifact, relative_path="other.bin")

    def reader(job_id: str, path: str) -> StudioJobArtifactPayload:
        assert (job_id, path) == (record.job_id, "output.bin")
        return StudioJobArtifactPayload(artifact=artifact, payload=original.payload)

    with pytest.raises(ValueError, match="source record"):
        write_studio_evidence_bundle(
            _context(tmp_path), job_records=(record,), artifact_reader=reader
        )
    assert not (tmp_path / "bundle/evidence/manifest.json").exists()


def test_bundle_rejects_source_snapshot_disagreement_with_real_reader(
    tmp_path: Path, source: tuple[StudioJobManager, StudioJobRecord]
) -> None:
    """Ledger-verified bytes cannot be exported under a different captured hash."""
    manager, record = source
    snapshot = replace(record, artifacts=(replace(record.artifacts[0], sha256="0" * 64),))
    with pytest.raises(ValueError, match="source record"):
        write_studio_evidence_bundle(
            _context(tmp_path), job_records=(snapshot,), artifact_reader=manager.read_artifact
        )
    assert manager.record(record.job_id) == record
    assert manager.read_artifact(record.job_id, "output.bin").payload == b"source bytes"
    assert not (tmp_path / "bundle/evidence/manifest.json").exists()


@pytest.mark.parametrize("data", [b"SOURCE BYTES", b"truncated"])
def test_bundle_rejects_reader_bytes_disagreement(
    tmp_path: Path, source: tuple[StudioJobManager, StudioJobRecord], data: bytes
) -> None:
    """Matching reader metadata cannot bless changed or truncated payload bytes."""
    manager, record = source
    original = manager.read_artifact(record.job_id, "output.bin")

    def reader(job_id: str, path: str) -> StudioJobArtifactPayload:
        assert (job_id, path) == (record.job_id, "output.bin")
        return StudioJobArtifactPayload(artifact=original.artifact, payload=data)

    with pytest.raises(ValueError, match="source record"):
        write_studio_evidence_bundle(
            _context(tmp_path), job_records=(record,), artifact_reader=reader
        )
    assert not (tmp_path / "bundle/evidence/manifest.json").exists()
