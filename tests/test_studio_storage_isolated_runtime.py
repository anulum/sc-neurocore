# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — isolated jobs runtime from explicit configuration

"""The isolated facade is built only from validated explicit configuration.

The built facade uses the production storage connection, which refuses to
act from a process that is not the configured API identity: this test
process is not, so the refusal is the real behaviour of the wiring.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

import pytest
from pydantic import ValidationError

from sc_neurocore.studio.platform.policy_models import Principal
from sc_neurocore.studio.platform.storage_configuration import StorageBoundaryConfiguration
from sc_neurocore.studio.platform.storage_isolated_runtime import build_isolated_job_manager
from sc_neurocore.studio.platform.settings import (
    StudioRuntimeSettings,
    build_default_studio_runtime_settings,
)
from sc_neurocore.studio.platform.storage_launcher_client_settings import (
    LauncherClientSettings,
    parse_launcher_client,
)
from sc_neurocore.studio.platform.storage_requester import delegated

LAUNCHER: dict[str, object] = {
    "socket_path": "/run/sc-neurocore/launcher.sock",
    "launcher_uid": 4001,
    "worker_gid": 4002,
    "grant_timeout_seconds": 3.0,
    "heartbeat_seconds": 5.0,
    "poll_seconds": 0.1,
    "attempts": 3,
    "live_retain": 32,
    "accept_direct_backend_limits": True,
}


def _boundary(tmp_path: Path) -> StorageBoundaryConfiguration:
    return StorageBoundaryConfiguration(
        storage_uid=os.getuid() + 3,
        api_uid=os.getuid() + 1,
        worker_uid=os.getuid() + 2,
        authority_root=tmp_path / "authority",
        spool_root=tmp_path / "spool",
        socket_path=tmp_path / "endpoint" / "storage.sock",
        workspace="default",
        frame_max_bytes=65536,
        max_metadata_bytes=4096,
        max_seed_bytes=1 << 20,
        max_seed_entries=8,
        max_manifest_bytes=4096,
        max_artifact_bytes=1 << 20,
        max_artifact_entries=64,
        transfer_timeout_seconds=5.0,
        max_connections=4,
    )


def test_absent_launcher_settings_mean_none_and_valid_ones_decode() -> None:
    """Only explicit JSON configures the launcher client; a root launcher is valid."""
    assert parse_launcher_client(None) is None
    root = parse_launcher_client(json.dumps({**LAUNCHER, "launcher_uid": 0}))
    assert root is not None and root.launcher_uid == 0
    settings = parse_launcher_client(json.dumps(LAUNCHER))
    assert settings is not None
    assert settings.socket_path == Path("/run/sc-neurocore/launcher.sock")


@pytest.mark.parametrize(
    "raw",
    [
        json.dumps({**LAUNCHER, "socket_path": "relative.sock"}),
        json.dumps({**LAUNCHER, "socket_path": "/run/../run/x.sock"}),
        json.dumps({**LAUNCHER, "attempts": 0}),
        json.dumps({**LAUNCHER, "launcher_uid": -1}),
        json.dumps({**LAUNCHER, "poll_seconds": float("inf")}).replace("Infinity", "1e999"),
        json.dumps({**LAUNCHER, "extra": 1}),
        json.dumps(
            {key: value for key, value in LAUNCHER.items() if key != "accept_direct_backend_limits"}
        ),
        json.dumps({**LAUNCHER, "accept_direct_backend_limits": 1}),
        '{"attempts": 1, "attempts": 2}',
        "[" * 100000,
    ],
    ids=[
        "relative",
        "noncanonical",
        "no-attempts",
        "negative-uid",
        "infinite",
        "extra",
        "unstated-backend-limits",
        "non-boolean-acceptance",
        "duplicate",
        "deep",
    ],
)
def test_invalid_launcher_settings_are_refused(raw: str) -> None:
    """Relative or non-canonical endpoints, bad bounds and ambiguous JSON refuse."""
    with pytest.raises((ValueError, ValidationError)):
        parse_launcher_client(raw)


def test_the_built_facade_acts_only_as_the_configured_api_identity(tmp_path: Path) -> None:
    """Every exchange goes through the production connection and its identity check."""
    settings = parse_launcher_client(json.dumps(LAUNCHER))
    assert settings is not None
    manager = build_isolated_job_manager(
        _boundary(tmp_path),
        settings,
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=30.0,
        max_artifact_bytes=1 << 20,
    )
    principal = Principal("operator", frozenset({"studio.admin"}))
    with (
        delegated(principal, method="GET", route="/api/studio/jobs", request_id="t"),
        pytest.raises(PermissionError, match="configured Linux API identity"),
    ):
        manager.list_records()


def test_launcher_settings_reach_the_runtime_settings_only_in_isolated_mode() -> None:
    """The environment names the launcher client; embedded mode refuses one."""
    settings = build_default_studio_runtime_settings(
        {
            "SC_NEUROCORE_STUDIO_STORAGE_MODE": "isolated",
            "SC_NEUROCORE_STUDIO_STORAGE_LAUNCHER": json.dumps(LAUNCHER),
        }
    )
    assert settings.storage_launcher == parse_launcher_client(json.dumps(LAUNCHER))
    assert build_default_studio_runtime_settings({}).storage_launcher is None
    with pytest.raises(ValueError, match="requires isolated mode"):
        build_default_studio_runtime_settings(
            {"SC_NEUROCORE_STUDIO_STORAGE_LAUNCHER": json.dumps(LAUNCHER)}
        )
    with pytest.raises(ValueError, match="validated configuration"):
        StudioRuntimeSettings(
            storage_mode="isolated", storage_launcher=cast(LauncherClientSettings, LAUNCHER)
        )
