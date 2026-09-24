# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — API reading of a stopped worker's spool

"""A worker's spool is untrusted input: only stable regular files are sent.

Every case builds a real spool directory with real files, links, pipes and
directories. A file changing while it is read is produced by the kernel
holding the API's ``read`` while the file is really appended to.
"""

from __future__ import annotations

from collections.abc import Iterator
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import pytest
from pydantic import ValidationError

from sc_neurocore.studio.platform.storage_finish_client import (
    RESULT_NAME,
    exchange_finish,
    spool_finish_request,
)
from sc_neurocore.studio.platform.storage_finish_protocol import (
    FinishOutcome,
    StorageFinishRequest,
)
from tests.studio_seccomp_support import SECCOMP_AVAILABLE, run_child

JOB = "sj_" + "6" * 16
FILES = {"reports/summary.json": b'{"ok": true}', "weights.bin": b"\x00\x01"}


def _manifest(files: dict[str, bytes]) -> list[dict[str, object]]:
    return [
        {"relative_path": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in files.items()
    ]


def _write(root: Path, result: object, files: dict[str, bytes]) -> None:
    for name, data in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(data)
    if result is not None:
        (root / RESULT_NAME).write_text(json.dumps(result))


@pytest.fixture
def spool(tmp_path: Path) -> Iterator[tuple[Path, int]]:
    root = tmp_path / "spool"
    root.mkdir()
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        yield root, descriptor
    finally:
        os.close(descriptor)


def _read(
    descriptor: int,
    outcome: FinishOutcome | None = None,
    limit: int = 4096,
    *,
    exit_status: int | None = 0,
    error: str | None = None,
    entries: int = 16,
) -> tuple[StorageFinishRequest, tuple[bytes, ...]]:
    return spool_finish_request(
        descriptor,
        workspace="default",
        job_id=JOB,
        outcome=outcome,
        exit_status=exit_status,
        frame_max_bytes=limit,
        max_artifact_bytes=entries * limit,
        max_artifact_entries=entries,
        error=error,
    )


def test_completed_worker_output_becomes_a_verified_request(spool: tuple[Path, int]) -> None:
    """Declared artefacts are re-read and re-hashed; the result is carried."""
    root, descriptor = spool
    _write(root, {"status": "completed", "result": {"a": 1}, "artifacts": _manifest(FILES)}, FILES)
    request, payloads = _read(descriptor)
    assert (request.outcome, request.result, request.error) == ("completed", {"a": 1}, None)
    assert [artifact.relative_path for artifact in request.artifacts] == list(FILES)
    assert payloads == tuple(FILES.values())


_COMPLETED = {"status": "completed", "result": [1]}
_EXITED = "Studio process worker exited with {}."


@pytest.mark.parametrize(
    "result,outcome,exit_status,error,expected",
    [
        (None, None, 1, None, ("failed", _EXITED.format(1))),
        ({"status": "failed", "error": "ValueError"}, None, 1, None, ("failed", "ValueError")),
        ({"status": "failed", "error": "x" * 2000}, None, 1, None, ("failed", _EXITED.format(1))),
        (_COMPLETED, None, 3, None, ("failed", _EXITED.format(3))),
        (None, None, None, None, ("failed", "Studio process worker has no recorded exit status.")),
        (_COMPLETED, "cancelled", -9, None, ("cancelled", None)),
        (_COMPLETED, "cancelled", -9, "not reaped", ("cancelled", "not reaped")),
        (None, "timed_out", -9, None, ("timed_out", "Studio job exceeded its timeout.")),
        (None, "failed", None, "launch refused", ("failed", "launch refused")),
        (_COMPLETED, None, 0, None, ("completed", None)),
    ],
    ids=[
        "no-result",
        "worker-error",
        "oversized-error",
        "completed-nonzero-exit",
        "never-ran",
        "cancelled",
        "cancelled-with-error",
        "timed-out",
        "api-error",
        "non-object-result",
    ],
)
def test_outcome_and_error_follow_the_worker_or_the_api(
    spool: tuple[Path, int],
    result: object,
    outcome: FinishOutcome | None,
    exit_status: int | None,
    error: str | None,
    expected: tuple[str, str | None],
) -> None:
    """The API's verdict wins; errors follow the embedded supervisor's wording."""
    root, descriptor = spool
    _write(root, result, {})
    request, _ = _read(descriptor, outcome, exit_status=exit_status, error=error)
    assert (request.outcome, request.error) == expected
    if request.outcome == "completed":
        assert request.result == {}


@pytest.mark.parametrize(
    "result,error",
    [
        ([1], ValueError),
        ({"artifacts": {"a": 1}}, ValueError),
        (
            {"artifacts": [{"relative_path": "../x", "size_bytes": 0, "sha256": "0" * 64}]},
            ValidationError,
        ),
    ],
    ids=["result-not-object", "manifest-not-list", "escaping-path"],
)
def test_malformed_worker_results_are_refused(
    spool: tuple[Path, int], result: object, error: type[Exception]
) -> None:
    """Shapes the contract does not allow never become a request."""
    root, descriptor = spool
    _write(root, result, {})
    with pytest.raises(error):
        _read(descriptor)


@pytest.mark.parametrize(
    "damage", ["digest", "grown", "symlink", "linked-directory", "pipe", "oversize"]
)
def test_untrusted_artefact_entries_are_refused(
    spool: tuple[Path, int], tmp_path: Path, damage: str
) -> None:
    """Changed bytes, links, pipes and oversized files are never read into a request."""
    root, descriptor = spool
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "summary.json").write_bytes(FILES["reports/summary.json"])
    declared = dict(FILES)
    files = dict(FILES)
    if damage == "digest":
        files["weights.bin"] = b"\x09\x09"
    elif damage == "grown":
        files["weights.bin"] = b"\x00\x01" + b"\x02" * 4096
    _write(root, {"status": "completed", "artifacts": _manifest(declared)}, files)
    if damage == "symlink":
        (root / "weights.bin").unlink()
        (root / "weights.bin").symlink_to(outside / "summary.json")
    elif damage == "linked-directory":
        (root / "reports" / "summary.json").unlink()
        (root / "reports").rmdir()
        (root / "reports").symlink_to(outside, target_is_directory=True)
    elif damage == "pipe":
        (root / "weights.bin").unlink()
        os.mkfifo(root / "weights.bin")
    limit = 4 if damage == "oversize" else 4096
    with pytest.raises((ValueError, OSError)):
        _read(descriptor, limit=limit)


@pytest.mark.parametrize(
    "limits,match",
    [
        ({"max_artifact_entries": 1}, "entry limit"),
        ({"frame_max_bytes": 512}, "frame limit"),
        ({"max_artifact_bytes": 13}, "aggregate limit"),
    ],
)
def test_declared_budgets_refuse_before_any_artefact_is_read(
    spool: tuple[Path, int], limits: dict[str, int], match: str
) -> None:
    """A worker declaring more than the authority accepts costs the API no reads.

    None of the declared files exists, so only a check made before reading
    can produce the budget refusal.
    """
    root, descriptor = spool
    declared = [
        *_manifest(FILES),
        {"relative_path": "big.bin", "size_bytes": 1000, "sha256": "0" * 64},
    ]
    _write(root, {"status": "completed", "artifacts": declared}, {})
    budgets = {"frame_max_bytes": 4096, "max_artifact_bytes": 4096, "max_artifact_entries": 16}
    budgets.update(limits)
    with pytest.raises(ValueError, match=match):
        spool_finish_request(
            descriptor,
            workspace="default",
            job_id=JOB,
            outcome=None,
            exit_status=0,
            frame_max_bytes=budgets["frame_max_bytes"],
            max_artifact_bytes=budgets["max_artifact_bytes"],
            max_artifact_entries=budgets["max_artifact_entries"],
        )


@pytest.mark.skipif(not SECCOMP_AVAILABLE, reason="held system calls need Linux x86_64")
def test_artefact_changing_while_read_is_refused(spool: tuple[Path, int]) -> None:
    """A file the worker keeps writing is not certified by hashing it once."""
    root, _ = spool
    _write(root, {"status": "completed", "artifacts": _manifest(FILES)}, FILES)
    result = run_child(
        "import os, sys\n"
        "from sc_neurocore.studio.platform.storage_finish_client import spool_finish_request\n"
        "from tests.studio_syscall_support import finish, hold_system_calls\n"
        "root = sys.argv[1]\n"
        "target = os.path.join(root, 'weights.bin')\n"
        "grown = []\n"
        "def decide(call):\n"
        "    if call.name == 'read' and not grown and call.descriptor_path(0) == target:\n"
        "        with open(target, 'ab') as handle:\n"
        "            handle.write(b'more')\n"
        "        grown.append(True)\n"
        "held = os.open(root, os.O_RDONLY | os.O_DIRECTORY)\n"
        "hold_system_calls(['read'], decide)\n"
        "try:\n"
        "    spool_finish_request(held, workspace='default', job_id='sj_' + '6' * 16,\n"
        "        outcome=None, exit_status=0, frame_max_bytes=4096,\n"
        "        max_artifact_bytes=65536, max_artifact_entries=16)\n"
        "    refused = None\n"
        "except ValueError as error:\n"
        "    refused = str(error)\n"
        "finish({'grown': grown, 'refused': refused})\n",
        arguments=(str(root),),
    )
    assert result == {"grown": [True], "refused": "spool entry changed while it was read"}


def test_payloads_must_match_the_manifest(spool: tuple[Path, int]) -> None:
    """The exchange refuses before sending when bytes and manifest disagree."""
    root, descriptor = spool
    _write(root, {"status": "completed", "artifacts": _manifest(FILES)}, FILES)
    request, _ = _read(descriptor)
    service, client = socket.socketpair()
    with service, pytest.raises(ValueError, match="do not match the manifest"):
        exchange_finish(
            client,
            request,
            [],
            expected_service_uid=os.getuid(),
            max_bytes=4096,
            deadline=time.monotonic() + 5,
        )
