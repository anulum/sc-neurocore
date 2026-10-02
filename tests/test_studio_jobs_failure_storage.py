# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real storage error projection exchanges

"""Retain qualified errors through real storage handlers, framing and record pickle."""

import json
import os
import pickle
import time
from pathlib import Path

import pytest

from sc_neurocore.refusals import AuthoredRefusal
from sc_neurocore.studio.platform.jobs import StudioJobContext, StudioJobManager
from sc_neurocore.studio.platform.jobs_failures import GENERIC_JOB_FAILURE
from sc_neurocore.studio.platform.jobs_models import (
    StudioJobArtifactUnavailable,
    StudioJobRejected,
)
from sc_neurocore.studio.platform.jobs_snapshot import decode_job_snapshot
from sc_neurocore.studio.platform.storage_cancel_client import cancel_request, exchange_cancel
from sc_neurocore.studio.platform.storage_live_spool import LiveSpools
from sc_neurocore.studio.platform.storage_purge_client import exchange_purge, purge_request
from sc_neurocore.studio.platform.storage_query_client import QueryReader
from sc_neurocore.studio.platform.storage_record_client import read_storage_record
from sc_neurocore.studio.platform.storage_record_protocol import (
    StorageRecordRequest,
    StorageRequester,
)
from tests.studio_storage_generation_support import Authority
from tests.studio_seccomp_support import run_child

ADMIN = StorageRequester(principal_id="operator", roles=("studio.admin",))
FRAME = 512


@pytest.mark.parametrize("authored", [False, True])
def test_qualified_errors_survive_all_real_storage_views(tmp_path: Path, authored: bool) -> None:
    """A real failure survives chunked read, list, stop, pickle and terminal purge."""
    manager = StudioJobManager(
        root=tmp_path,
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=5.0,
        max_artifact_bytes=4,
        max_concurrent_jobs=2,
        max_queued_jobs=0,
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        """Exercise either the real source-owned byte limit or a real OS error."""
        context.write_artifact(
            "report.txt" if authored else "x" * 300, b"12345" if authored else b"1234"
        )
        return {}

    submitted = manager.submit(kind="analysis", owner="operator", request_id=None, task=task)
    local = manager.wait(submitted.job_id, 5.0)
    public = (
        "Studio job artifact exceeds configured size limit." if authored else GENERIC_JOB_FAILURE
    )
    assert local.status == "failed" and local.public_error == public
    assert local.error is not None
    if not authored:
        assert str(tmp_path) in local.error
    restored = pickle.loads(pickle.dumps(local))
    assert restored.error == local.error and restored.public_error == public
    # A generic decoder deliberately does not trust arbitrary public text.
    decoded = decode_job_snapshot(local.to_public_dict())
    assert decoded.public_error == GENERIC_JOB_FAILURE
    assert len(json.dumps(local.to_public_dict()).encode()) > FRAME
    authority = Authority(manager._ledger, frame_max_bytes=FRAME)
    try:
        remote = read_storage_record(
            authority.connect(),
            request=StorageRecordRequest(
                schema_version="studio.storage.record.v3",
                operation="record",
                request_id=None,
                job_id=local.job_id,
                workspace="default",
                requester=ADMIN,
            ),
            expected_service_uid=os.getuid(),
            max_bytes=FRAME,
            deadline=time.monotonic() + 10.0,
        )
        reader = QueryReader(
            authority.connect,
            workspace="default",
            storage_uid=os.getuid(),
            max_bytes=FRAME,
            timeout_seconds=10.0,
        )
        records = reader.records(ADMIN)
        cancelled = exchange_cancel(
            authority.connect(),
            cancel_request("default", local.job_id, requester=ADMIN),
            expected_service_uid=os.getuid(),
            max_bytes=FRAME,
            deadline=time.monotonic() + 10.0,
        )
        assert manager.record(local.job_id).error == local.error
        purged = exchange_purge(
            authority.connect(),
            purge_request("default", local.job_id, requester=ADMIN),
            expected_service_uid=os.getuid(),
            max_bytes=FRAME,
            deadline=time.monotonic() + 10.0,
        )
        for viewed in (remote, *records, cancelled, purged):
            assert viewed.job_id == local.job_id and viewed.status == "failed"
            assert viewed.error == viewed.public_error == public
            assert viewed.to_public_dict()["error"] == public
            assert str(tmp_path) not in json.dumps(viewed.to_public_dict())
        with pytest.raises(KeyError):
            manager.record(local.job_id)
        assert manager.status().active_count == 0 and manager.unreaped_workers == ()
    finally:
        authority.join()
        manager._ledger.close()


def test_remote_purge_preserves_pending_stage_and_refusal(tmp_path: Path) -> None:
    """An actual retained stage refuses remote purge without deleting job or bytes."""
    manager = StudioJobManager(
        root=tmp_path, allowed_kinds=frozenset({"analysis"}), default_timeout_seconds=5.0
    )

    def task(context: StudioJobContext) -> dict[str, object]:
        """Create actual terminal job custody before introducing a stage conflict."""
        context.write_artifact("report.txt", b"data")
        return {}

    authority = Authority(manager._ledger)
    try:
        submitted = manager.submit(kind="analysis", owner="operator", request_id=None, task=task)
        record = manager.wait(submitted.job_id, 5.0)
        assert record.status == "completed"
        retained = tmp_path / (".purge-" + record.job_id) / "retained.bin"
        retained.parent.mkdir()
        retained.write_bytes(b"retained stage")
        with pytest.raises(StudioJobRejected) as caught:
            exchange_purge(
                authority.connect(),
                purge_request("default", record.job_id, requester=ADMIN),
                expected_service_uid=os.getuid(),
                max_bytes=FRAME,
                deadline=time.monotonic() + 10.0,
            )
        assert str(caught.value) == "Studio job has a pending purge requiring recovery."
        assert not isinstance(caught.value, AuthoredRefusal)
        assert manager.record(record.job_id) == record
        assert retained.read_bytes() == b"retained stage"
        assert (tmp_path / record.job_id / "report.txt").read_bytes() == b"data"
        assert manager.status().active_count == 0 and manager.unreaped_workers == ()
    finally:
        authority.join()
        manager._ledger.close()


@pytest.mark.parametrize("entry", ["escape", "link", "pipe", "directory"])
def test_live_spool_refuses_actual_worker_paths(tmp_path: Path, entry: str) -> None:
    """A real held-directory reader refuses unsafe entries and preserves private bytes.

    This exercises the public live-spool component under one identity; it does
    not claim a three-role launch or acceptance of staging permissions.
    """
    job_id = "sj_" + "8" * 16
    work = tmp_path / job_id
    work.mkdir()
    private = tmp_path / "private.bin"
    private.write_bytes(b"private authority bytes")
    target = work / "events.jsonl"
    path = "events.jsonl"
    if entry == "escape":
        path = "../private.bin"
    elif entry == "link":
        target.symlink_to(private)
    elif entry == "pipe":
        os.mkfifo(target)
    else:
        target.mkdir()
    live = LiveSpools(retain=1, max_seed_bytes=4)
    descriptor = os.open(work, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        live.attach(job_id, descriptor)
        with pytest.raises(StudioJobArtifactUnavailable) as caught:
            live.read(job_id, path, offset=0, max_bytes=64)
        assert isinstance(caught.value, AuthoredRefusal)
        expected = (
            "spool path escapes its directory"
            if entry == "escape"
            else "Studio live artifact is unavailable."
        )
        assert str(caught.value) == expected and str(tmp_path) not in str(caught.value)
        assert private.read_bytes() == b"private authority bytes"
        (work / "ordinary.txt").write_bytes(b"healthy")
        assert live.read(job_id, "ordinary.txt", offset=0, max_bytes=64) == (b"healthy", 7)
        assert live.read(job_id, "absent.txt", offset=3, max_bytes=64) == (b"", 3)
    finally:
        os.close(descriptor)
        live.close()


@pytest.mark.parametrize("boundary", ["command", "seed", "path", "missing"])
def test_live_spool_delivery_refusals_precede_any_write(tmp_path: Path, boundary: str) -> None:
    """Public delivery refuses bounds and unknown custody without publishing any file."""
    from sc_neurocore.studio.platform.jobs_models import STUDIO_CONTROL_COMMAND_MAX_BYTES

    live = LiveSpools(retain=1, max_seed_bytes=4)
    command = b"x" * (STUDIO_CONTROL_COMMAND_MAX_BYTES + 1) if boundary == "command" else b"{}"
    seeds = {"large.bin": b"12345"} if boundary == "seed" else {}
    if boundary == "path":
        seeds = {"../private.bin": b"x"}
    try:
        with pytest.raises(StudioJobRejected) as caught:
            live.deliver("sj_" + "8" * 16, command, seeds)
        assert isinstance(caught.value, AuthoredRefusal)
        expected = {
            "command": "Studio job control command exceeds configured size limit.",
            "seed": "Studio job seed input exceeds configured size limit.",
            "path": "spool path escapes its directory",
            "missing": "Studio job work directory is unavailable.",
        }[boundary]
        assert str(caught.value) == expected and str(tmp_path) not in str(caught.value)
        assert list(tmp_path.iterdir()) == []
    finally:
        live.close()


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("line\nbreak", "spool path must be printable text"),
        ("nested//seed.bin", "spool path must be canonical"),
        ("./seed.bin", "spool path must be canonical"),
    ],
)
def test_live_reads_and_control_delivery_require_canonical_text(path: str, message: str) -> None:
    """Both public spool consumers reject unsafe text before accessing any directory."""
    live = LiveSpools(retain=1, max_seed_bytes=4)
    try:
        with pytest.raises(StudioJobArtifactUnavailable) as read_error:
            live.read("sj_" + "8" * 16, path, offset=0, max_bytes=64)
        with pytest.raises(StudioJobRejected) as delivery_error:
            live.deliver("sj_" + "8" * 16, b"{}", {path: b"x"})
        for caught in (read_error, delivery_error):
            assert isinstance(caught.value, AuthoredRefusal) and str(caught.value) == message
    finally:
        live.close()


def test_untyped_live_reader_refuses_nontext_path() -> None:
    """An actual untyped Python caller reaches the public nontext boundary."""
    result = run_child(
        "import json\n"
        "from sc_neurocore.refusals import AuthoredRefusal\n"
        "from sc_neurocore.studio.platform.jobs_models import StudioJobArtifactUnavailable\n"
        "from sc_neurocore.studio.platform.storage_live_spool import LiveSpools\n"
        "live = LiveSpools(retain=1, max_seed_bytes=4)\n"
        "try:\n"
        "    live.read('sj_' + '8' * 16, 7, offset=0, max_bytes=64)\n"
        "except StudioJobArtifactUnavailable as error:\n"
        "    print(json.dumps({'authored': isinstance(error, AuthoredRefusal),\n"
        "        'message': str(error)}))\n"
        "finally:\n"
        "    live.close()\n"
    )
    assert result == {"authored": True, "message": "spool path must be printable text"}
