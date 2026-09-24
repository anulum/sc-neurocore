# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sealed artefact reads from the storage authority

"""Completed artefacts are served only from the authority's sealed, verified copy.

Artefacts are sealed by the real finish exchange of a started job whose real
worker stopped; reads go over real sockets through the service's dispatch.
Damage is done to the real sealed files.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from sc_neurocore.studio.platform.jobs_ledger import StudioJobLedger
from sc_neurocore.studio.platform.jobs_models import (
    StudioJobArtifactPayload,
    StudioJobArtifactUnavailable,
)
from sc_neurocore.studio.platform.storage_artifact_client import artifact_request, exchange_artifact
from sc_neurocore.studio.platform.storage_artifact_protocol import ArtifactRoute
from sc_neurocore.studio.platform.storage_record_protocol import StorageRequester
from tests.studio_storage_finish_support import FILES, finish, request, started, stop
from tests.studio_storage_generation_support import FRAME, Authority
from tests.studio_storage_supervision_support import *

ADMIN = StorageRequester(principal_id="operator", roles=("studio.admin",))
DOWNLOAD: ArtifactRoute = "/api/studio/jobs/{job_id}/artifacts/{artifact_path:path}"
RESTORE: ArtifactRoute = "/api/studio/training/weight-restore"


@pytest.fixture
def sealed(ledger: StudioJobLedger) -> Path:
    """A completed job whose artefacts the authority sealed."""
    stop(started(ledger))
    assert finish(ledger, request(FILES), list(FILES.values())).reply == "sealed"
    return ledger.path.parent / JOB


def _read(
    authority: Authority,
    path: str,
    *,
    route: ArtifactRoute = DOWNLOAD,
    requester: StorageRequester | None = ADMIN,
    job_id: str = JOB,
) -> StudioJobArtifactPayload:
    return exchange_artifact(
        authority.connect(),
        artifact_request("default", job_id, path, route=route, requester=requester),
        expected_service_uid=os.getuid(),
        max_bytes=authority.services.frame_max_bytes,
        deadline=time.monotonic() + 10,
    )


@pytest.mark.parametrize("route", [DOWNLOAD, RESTORE])
def test_every_declared_artefact_is_served_exactly(
    ledger: StudioJobLedger, sealed: Path, route: ArtifactRoute
) -> None:
    """Bytes, size and digest match the declaration, the empty artefact included."""
    authority = Authority(ledger)
    for name, payload in FILES.items():
        served = _read(authority, name, route=route)
        assert (served.artifact.relative_path, served.payload) == (name, payload)
        assert served.artifact.size_bytes == len(payload)
    authority.join()
    assert {event.route for event in authority.audit.events} == {route}


def test_another_workspace_is_never_served(ledger: StudioJobLedger, sealed: Path) -> None:
    """A read naming another workspace is refused before any lookup."""
    authority = Authority(ledger)
    with pytest.raises(EOFError):
        exchange_artifact(
            authority.connect(),
            artifact_request("elsewhere", JOB, "weights.bin", route=DOWNLOAD, requester=ADMIN),
            expected_service_uid=os.getuid(),
            max_bytes=FRAME,
            deadline=time.monotonic() + 10,
        )
    authority.join(expected=(ValueError,))


def test_policy_and_declarations_bound_what_is_served(
    ledger: StudioJobLedger, sealed: Path
) -> None:
    """Non-admins are denied; unknown jobs, undeclared and escaping paths are not found."""
    authority = Authority(ledger)
    for requester in (None, StorageRequester(principal_id="viewer", roles=())):
        with pytest.raises(PermissionError):
            _read(authority, "weights.bin", requester=requester)
    for path, job_id in (
        ("absent.bin", JOB),
        ("../weights.bin", JOB),
        ("weights.bin", "sj_" + "0" * 16),
    ):
        with pytest.raises(KeyError):
            _read(authority, path, job_id=job_id)
    authority.join()


@pytest.mark.parametrize("damage", ["changed", "grown", "removed", "linked", "pipe"])
def test_untrusted_sealed_bytes_are_unavailable(
    ledger: StudioJobLedger, sealed: Path, tmp_path: Path, damage: str
) -> None:
    """Changed, resized, missing, linked or non-regular sealed files are never served."""
    target = sealed / "reports" / "summary.json"
    if damage in ("changed", "grown"):
        target.chmod(0o600)
        target.write_bytes(b'{"ok": nope}' if damage == "changed" else b'{"ok": true} ')
    elif damage == "removed":
        target.unlink()
    elif damage == "pipe":
        target.unlink()
        os.mkfifo(target)
    else:
        outside = tmp_path / "outside.json"
        outside.write_bytes(FILES["reports/summary.json"])
        target.unlink()
        target.symlink_to(outside)
    authority = Authority(ledger)
    with pytest.raises(StudioJobArtifactUnavailable):
        _read(authority, "reports/summary.json")
    assert _read(authority, "weights.bin").payload == FILES["weights.bin"]
    authority.join()


def test_an_artefact_above_the_frame_is_unavailable(ledger: StudioJobLedger) -> None:
    """A service configured with a smaller frame than the seal refuses, never truncates."""
    stop(started(ledger))
    large = {"large.bin": bytes(range(256)) * 12}
    assert finish(ledger, request(large), list(large.values())).reply == "sealed"
    authority = Authority(ledger, frame_max_bytes=2048)
    with pytest.raises(StudioJobArtifactUnavailable):
        _read(authority, "large.bin")
    authority.join()
    assert _read(Authority(ledger), "large.bin").payload == large["large.bin"]
