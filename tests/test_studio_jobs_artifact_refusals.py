# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public artifact reader refusal contracts

"""Exercise retained manifest and filesystem faults through actual job readers."""

import json
import os
import sqlite3
import time
from pathlib import Path

import pytest

from sc_neurocore.refusals import AuthoredRefusal
from sc_neurocore.studio.platform.jobs import StudioJobContext, StudioJobManager
from sc_neurocore.studio.platform.jobs_models import StudioJobArtifactRefused
from sc_neurocore.studio.platform.storage_artifact_client import artifact_request, exchange_artifact
from sc_neurocore.studio.platform.storage_record_protocol import StorageRequester
from tests.studio_storage_generation_support import Authority


@pytest.mark.parametrize("reader", ["declared", "live"])
def test_public_reader_refuses_replaced_artifact_symlink(tmp_path: Path, reader: str) -> None:
    """A real replaced artifact cannot reveal bytes outside its job directory."""
    root = tmp_path / "jobs"
    retained = tmp_path / "retained.txt"
    retained.write_bytes(b"retained outside the job")
    manager = StudioJobManager(
        root=root, allowed_kinds=frozenset({"analysis"}), default_timeout_seconds=5.0
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        """Create a genuine manifest before substituting the underlying file."""
        context.write_artifact("report.txt", b"data")
        return {}

    try:
        submitted = manager.submit(kind="analysis", owner="operator", request_id=None, task=task)
        record = manager.wait(submitted.job_id, 5.0)
        assert record.status == "completed" and len(record.artifacts) == 1
        artifact = root / record.job_id / "report.txt"
        artifact.unlink()
        artifact.symlink_to(retained)
        with pytest.raises(StudioJobArtifactRefused) as caught:
            if reader == "declared":
                manager.read_artifact(record.job_id, "report.txt")
            else:
                manager.read_live_artifact_bytes(record.job_id, "report.txt", offset=0)
        assert isinstance(caught.value, AuthoredRefusal)
        assert str(caught.value) == "Studio job artifact path escapes the job directory."
        assert caught.value.__cause__ is not None
        assert retained.read_bytes() == b"retained outside the job"
        assert manager.record(record.job_id).status == "completed"
        assert manager.status().active_count == 0
    finally:
        manager._ledger.close()


@pytest.mark.parametrize("fault", ["missing-file", "negative-size"])
def test_public_reader_retains_job_after_manifest_refusal(tmp_path: Path, fault: str) -> None:
    """Missing bytes or an invalid retained size refuse reads without deleting custody."""
    manager = StudioJobManager(
        root=tmp_path, allowed_kinds=frozenset({"analysis"}), default_timeout_seconds=5.0
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        """Seal one real artifact through the production writer."""
        context.write_artifact("report.txt", b"data")
        return {}

    try:
        submitted = manager.submit(kind="analysis", owner="operator", request_id=None, task=task)
        record = manager.wait(submitted.job_id, 5.0)
        assert record.status == "completed"
        if fault == "missing-file":
            (tmp_path / record.job_id / "report.txt").unlink()
            expected = "Studio job artifact is unavailable."
        else:
            manifest = [artifact.to_public_dict() for artifact in record.artifacts]
            manifest[0]["size_bytes"] = -1
            with sqlite3.connect(manager.ledger_path) as connection:
                connection.execute(
                    "UPDATE jobs SET artifacts=? WHERE job_id=?",
                    (json.dumps(manifest), record.job_id),
                )
            expected = "Studio job artifact integrity check failed."
        with pytest.raises(StudioJobArtifactRefused) as caught:
            manager.read_artifact(record.job_id, "report.txt")
        assert isinstance(caught.value, AuthoredRefusal) and str(caught.value) == expected
        restored = manager.record(record.job_id)
        assert restored.status == "completed" and restored.artifacts
        assert manager.status().active_count == 0
    finally:
        manager._ledger.close()


def test_real_storage_reader_refuses_missing_sealed_bytes(tmp_path: Path) -> None:
    """The actual authority returns exact sealed bytes or an authored unavailable reason."""
    manager = StudioJobManager(
        root=tmp_path, allowed_kinds=frozenset({"analysis"}), default_timeout_seconds=5.0
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        """Produce the real file and declaration consumed by the authority."""
        context.write_artifact("report.txt", b"data")
        return {}

    authority = Authority(manager._ledger)
    try:
        submitted = manager.submit(kind="analysis", owner="operator", request_id=None, task=task)
        record = manager.wait(submitted.job_id, 5.0)
        assert record.status == "completed"
        requester = StorageRequester(principal_id="operator", roles=("studio.admin",))
        for available in (True, False):
            if not available:
                (tmp_path / record.job_id / "report.txt").unlink()
            request = artifact_request(
                "default",
                record.job_id,
                "report.txt",
                route="/api/studio/jobs/{job_id}/artifacts/{artifact_path:path}",
                requester=requester,
            )
            if available:
                payload = exchange_artifact(
                    authority.connect(),
                    request,
                    expected_service_uid=os.getuid(),
                    max_bytes=4096,
                    deadline=time.monotonic() + 10.0,
                )
                assert payload.payload == b"data" and payload.artifact == record.artifacts[0]
            else:
                with pytest.raises(StudioJobArtifactRefused) as caught:
                    exchange_artifact(
                        authority.connect(),
                        request,
                        expected_service_uid=os.getuid(),
                        max_bytes=4096,
                        deadline=time.monotonic() + 10.0,
                    )
                assert isinstance(caught.value, AuthoredRefusal)
                assert str(caught.value) == "Studio job artifact is unavailable."
        assert manager.record(record.job_id) == record and manager.status().active_count == 0
    finally:
        authority.join()
        manager._ledger.close()
