# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — preflight of the isolated Studio API

"""The preflight reads this process and host as they are and names every failure.

The passing case configures this test process as the API identity with a
real spool it owns; failing cases use real missing or open directories and
another identity.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI
import pytest

from sc_neurocore.studio.api.runtime import build_studio_api_context
from sc_neurocore.studio.platform.api_process_lock import api_lock_path
from sc_neurocore.studio.platform.settings import StudioRuntimeSettings
from sc_neurocore.studio.platform.storage_isolated_jobs import IsolatedJobManager
from sc_neurocore.studio.platform.storage_configuration import StorageBoundaryConfiguration
from sc_neurocore.studio.platform.storage_launcher_client_settings import LauncherClientSettings
from sc_neurocore.studio.platform.storage_preflight import (
    isolated_preflight,
    require_isolated_preflight,
)


def _boundary(tmp_path: Path, *, api_uid: int) -> StorageBoundaryConfiguration:
    uids = sorted({api_uid, os.getuid() + 11, os.getuid() + 12} - {api_uid})
    return StorageBoundaryConfiguration(
        storage_uid=uids[0],
        api_uid=api_uid,
        worker_uid=uids[1],
        authority_root=tmp_path / "authority",
        spool_root=tmp_path / "spool",
        socket_path=tmp_path / "endpoint" / "storage.sock",
        workspace="default",
        frame_max_bytes=8192,
        max_metadata_bytes=4096,
        max_seed_bytes=8192,
        max_seed_entries=16,
        max_manifest_bytes=1024,
        max_artifact_bytes=65536,
        max_artifact_entries=16,
        transfer_timeout_seconds=2.0,
        max_connections=2,
    )


def _launcher(tmp_path: Path, *, accept: bool = True) -> LauncherClientSettings:
    return LauncherClientSettings(
        socket_path=tmp_path / "launcher" / "launcher.sock",
        launcher_uid=os.getuid() + 13,
        worker_gid=os.getgid(),
        grant_timeout_seconds=3.0,
        heartbeat_seconds=5.0,
        poll_seconds=0.1,
        attempts=3,
        live_retain=8,
        accept_direct_backend_limits=accept,
    )


def _settings(tmp_path: Path, *, api_uid: int) -> StudioRuntimeSettings:
    return StudioRuntimeSettings(
        storage_mode="isolated",
        storage_boundary=_boundary(tmp_path, api_uid=api_uid),
        storage_launcher=_launcher(tmp_path),
        enforce_route_policies=True,
        allow_header_principal=False,
        identity_file_path=str(tmp_path / "identity.json"),
        audit_log_path=str(tmp_path / "audit.jsonl"),
    )


def test_a_prepared_api_process_passes_every_check(tmp_path: Path) -> None:
    """The API identity, policy, audit, link protection, spool and endpoint all hold."""
    (tmp_path / "spool").mkdir()
    (tmp_path / "spool").chmod(0o750)
    (tmp_path / "launcher").mkdir()
    settings = _settings(tmp_path, api_uid=os.getuid())
    assert isolated_preflight(settings) == ()
    require_isolated_preflight(settings)


def test_every_unmet_requirement_is_named(tmp_path: Path) -> None:
    """Missing configuration and a permissive HTTP profile each fail by name."""
    failures = isolated_preflight(StudioRuntimeSettings(storage_mode="isolated"))
    assert [failure.check for failure in failures] == [
        "boundary",
        "launcher",
        "policies",
        "header_identity",
        "identity_store",
        "audit",
    ]
    with pytest.raises(RuntimeError, match="boundary: no storage boundary is configured"):
        require_isolated_preflight(StudioRuntimeSettings(storage_mode="isolated"))


@pytest.mark.parametrize("spool", ["missing", "open", "not-traversable"])
def test_identity_spool_and_endpoint_are_checked_against_the_host(
    tmp_path: Path, spool: str
) -> None:
    """Another API identity, an unusable spool and a missing endpoint directory fail."""
    if spool == "open":
        (tmp_path / "spool").mkdir()
        (tmp_path / "spool").chmod(0o757)
    elif spool == "not-traversable":
        (tmp_path / "spool").mkdir()
        (tmp_path / "spool").chmod(0o700)
    failures = isolated_preflight(_settings(tmp_path, api_uid=os.getuid() + 1))
    assert [failure.check for failure in failures] == ["identity", "spool", "launcher_endpoint"]


def test_an_api_outside_the_compute_group_fails(tmp_path: Path) -> None:
    """The API must belong to the group whose members read worker output."""
    (tmp_path / "spool").mkdir()
    (tmp_path / "spool").chmod(0o750)
    (tmp_path / "launcher").mkdir()
    settings = _settings(tmp_path, api_uid=os.getuid())
    foreign = next(gid for gid in range(1, 1 << 16) if gid not in {os.getegid(), *os.getgroups()})
    outside = replace(
        settings,
        storage_launcher=_launcher(tmp_path).model_copy(update={"worker_gid": foreign}),
    )
    assert [failure.check for failure in isolated_preflight(outside)] == ["compute_group"]


@pytest.mark.parametrize(("hardlinks", "symlinks"), [(None, "1"), ("0", "2")])
def test_unprotected_or_unreadable_link_settings_fail_by_name(
    tmp_path: Path, hardlinks: str | None, symlinks: str
) -> None:
    """A link-protection value that is off, relaxed or unreadable names its setting."""
    (tmp_path / "spool").mkdir()
    (tmp_path / "spool").chmod(0o750)
    (tmp_path / "launcher").mkdir()
    fs = tmp_path / "fs"
    fs.mkdir()
    if hardlinks is not None:
        (fs / "protected_hardlinks").write_text(hardlinks + "\n", encoding="ascii")
    (fs / "protected_symlinks").write_text(symlinks + "\n", encoding="ascii")
    failures = isolated_preflight(_settings(tmp_path, api_uid=os.getuid()), sysctl_root=fs)
    expected = [("protected_hardlinks", f"fs.protected_hardlinks is {hardlinks or 'unreadable'}")]
    if symlinks != "1":
        expected.append(("protected_symlinks", f"fs.protected_symlinks is {symlinks}"))
    assert [(failure.check, failure.detail) for failure in failures] == expected


def test_unaccepted_direct_backend_limits_refuse_an_otherwise_ready_api(tmp_path: Path) -> None:
    """Without the operator's explicit acceptance the direct backend refuses by name."""
    (tmp_path / "spool").mkdir()
    (tmp_path / "spool").chmod(0o750)
    (tmp_path / "launcher").mkdir()
    settings = replace(
        _settings(tmp_path, api_uid=os.getuid()),
        storage_launcher=_launcher(tmp_path, accept=False),
    )
    assert [failure.check for failure in isolated_preflight(settings)] == ["launcher_backend"]
    with pytest.raises(RuntimeError, match="launcher_backend: the direct launcher backend"):
        require_isolated_preflight(settings)


def test_a_passing_preflight_builds_the_isolated_facade_and_no_local_job_state(
    tmp_path: Path,
) -> None:
    """The API context holds the storage-backed facade; no job root or ledger appears.

    The one file the API process adds is its lock beside the identity store,
    which marks the serving process and holds no job state.
    """
    (tmp_path / "spool").mkdir()
    (tmp_path / "spool").chmod(0o750)
    (tmp_path / "launcher").mkdir()
    identity = tmp_path / "identity.json"
    identity.write_text(
        json.dumps(
            {
                "schema_version": "sc-neurocore.studio.identity.v1",
                "service_accounts": [
                    {
                        "principal_id": "svc-admin",
                        "roles": ["studio.admin"],
                        "token_sha256": hashlib.sha256(b"admin-token").hexdigest(),
                    }
                ],
            }
        )
    )
    settings = replace(_settings(tmp_path, api_uid=os.getuid()), identity_file_path=str(identity))
    before = sorted(tmp_path.iterdir())
    app = FastAPI()
    context = build_studio_api_context(app, settings)
    assert isinstance(context.studio_job_manager, IsolatedJobManager)
    assert app.state.studio_job_manager is context.studio_job_manager
    assert sorted(tmp_path.iterdir()) == sorted([*before, api_lock_path(identity)])
