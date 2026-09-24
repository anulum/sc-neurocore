# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Storage boundary startup validation

"""Validate operator intent through real settings and fail-closed API startup."""

import json
from pathlib import Path
from typing import cast

from fastapi import FastAPI
import pytest

from sc_neurocore.studio.api.runtime import build_studio_api_context
from sc_neurocore.studio.platform.settings import (
    StudioRuntimeSettings,
    build_default_studio_runtime_settings,
)
from sc_neurocore.studio.platform.storage_configuration import StorageBoundaryConfiguration


def _configuration(root: Path) -> dict[str, object]:
    return {
        "storage_uid": 1001,
        "api_uid": 1002,
        "worker_uid": 1003,
        "authority_root": str(root / "authority"),
        "spool_root": str(root / "spool"),
        "socket_path": str(root / "endpoint" / "storage.sock"),
        "workspace": "default",
        "frame_max_bytes": 8192,
        "max_metadata_bytes": 4096,
        "max_seed_bytes": 8192,
        "max_seed_entries": 16,
        "max_manifest_bytes": 1024,
        "max_artifact_bytes": 65536,
        "max_artifact_entries": 16,
        "transfer_timeout_seconds": 2.0,
        "max_connections": 4,
    }


def _settings(configuration: dict[str, object]) -> StudioRuntimeSettings:
    return build_default_studio_runtime_settings(
        {
            "SC_NEUROCORE_STUDIO_STORAGE_MODE": "isolated",
            "SC_NEUROCORE_STUDIO_STORAGE_BOUNDARY": json.dumps(configuration),
        }
    )


def test_boundary_intent_survives_settings_but_does_not_enable_storage(tmp_path: Path) -> None:
    """Valid distinct-role configuration creates no ledger, spool or endpoint."""
    payload = _configuration(tmp_path)
    settings = _settings(payload)
    assert settings.storage_boundary is not None
    assert settings.storage_boundary.model_dump(mode="json") == payload
    with pytest.raises(ValueError, match="frozen"):
        settings.storage_boundary.workspace = "another"
    app = FastAPI()
    with pytest.raises(RuntimeError, match="isolated preflight failed"):
        build_studio_api_context(app, settings)
    assert list(tmp_path.iterdir()) == []
    assert not hasattr(app.state, "studio_job_manager")
    assert build_default_studio_runtime_settings({}).storage_boundary is None


@pytest.mark.parametrize("field", list(_configuration(Path("/nonexistent"))))
def test_boundary_requires_all_explicit_fields(tmp_path: Path, field: str) -> None:
    """No missing deployment identity/path/budget is silently defaulted."""
    payload = _configuration(tmp_path)
    del payload[field]
    with pytest.raises(ValueError):
        _settings(payload)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("storage_uid", 0),
        ("api_uid", -1),
        ("worker_uid", 0xFFFFFFFF),
        ("storage_uid", True),
        ("api_uid", "1002"),
        ("worker_uid", 1002),
        ("authority_root", "/"),
        ("spool_root", "relative"),
        ("socket_path", "/"),
        ("workspace", ""),
        ("workspace", " "),
        ("workspace", 1),
        ("frame_max_bytes", 0),
        ("frame_max_bytes", 0x100000000),
        ("frame_max_bytes", True),
        ("max_metadata_bytes", 0),
        ("max_metadata_bytes", 8193),
        ("max_metadata_bytes", True),
        ("max_seed_bytes", -1),
        ("max_seed_bytes", True),
        ("max_seed_entries", -1),
        ("max_seed_entries", True),
        ("max_manifest_bytes", 0),
        ("max_manifest_bytes", 4097),
        ("max_manifest_bytes", True),
        ("transfer_timeout_seconds", 0),
        ("transfer_timeout_seconds", float("nan")),
        ("transfer_timeout_seconds", float("inf")),
        ("transfer_timeout_seconds", True),
        ("max_connections", 0),
        ("max_connections", True),
        ("extra", "refuse"),
    ],
)
def test_boundary_refuses_invalid_roles_paths_and_limits(
    tmp_path: Path, field: str, value: object
) -> None:
    """Malformed configuration fails before filesystem or runtime mutation."""
    payload = _configuration(tmp_path)
    payload[field] = value
    with pytest.raises(ValueError):
        _settings(payload)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "case",
    ["equal", "storage-parent", "spool-parent", "socket-storage", "socket-spool", "socket-parent"],
)
def test_overlapping_namespaces_are_not_accepted(tmp_path: Path, case: str) -> None:
    """Every root-pair overlap orientation refuses before creating a path."""
    payload = _configuration(tmp_path)
    if case == "equal":
        payload["spool_root"] = payload["authority_root"]
    elif case == "storage-parent":
        payload["spool_root"] = str(tmp_path / "authority" / "spool")
    elif case == "spool-parent":
        payload["authority_root"] = str(tmp_path / "spool" / "authority")
    elif case == "socket-storage":
        payload["socket_path"] = str(tmp_path / "authority" / "storage.sock")
    elif case == "socket-spool":
        payload["socket_path"] = str(tmp_path / "spool" / "storage.sock")
    else:
        payload["socket_path"] = str(tmp_path / "storage.sock")
    with pytest.raises(ValueError, match="overlap"):
        _settings(payload)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("loop", [False, True])
def test_existing_symlink_alias_is_not_boundary_evidence(tmp_path: Path, loop: bool) -> None:
    """Actual linked directories and symlink loops refuse without following writes."""
    link = tmp_path / "alias"
    link.symlink_to(link if loop else tmp_path, target_is_directory=True)
    payload = _configuration(tmp_path)
    payload["authority_root"] = str(link / "authority")
    with pytest.raises(ValueError):
        _settings(payload)
    assert list(tmp_path.iterdir()) == [link]
    assert link.is_symlink()


@pytest.mark.parametrize(
    "payload",
    ["", "null", "[]", "{}{}", '{"storage_uid":1,"storage_uid":2}', "[" * 2000 + "]" * 2000],
)
def test_invalid_json_does_not_become_embedded_configuration(payload: str) -> None:
    """Bad or ambiguous JSON cannot silently reduce the selected isolation."""
    with pytest.raises(ValueError):
        build_default_studio_runtime_settings(
            {
                "SC_NEUROCORE_STUDIO_STORAGE_MODE": "isolated",
                "SC_NEUROCORE_STUDIO_STORAGE_BOUNDARY": payload,
            }
        )


def test_embedded_and_unvalidated_boundaries_refuse(tmp_path: Path) -> None:
    """Programmatic settings cannot ignore or accept an unvalidated boundary."""
    configured = _settings(_configuration(tmp_path)).storage_boundary
    with pytest.raises(ValueError, match="requires isolated"):
        StudioRuntimeSettings(storage_boundary=configured)
    with pytest.raises(ValueError, match="validated configuration"):
        StudioRuntimeSettings(
            storage_mode="isolated", storage_boundary=cast(StorageBoundaryConfiguration, {})
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("nested", ['{"a": {}}', "[[]]", '{"storage_uid": [1]}'])
def test_any_nesting_inside_the_flat_boundary_object_is_refused(nested: str) -> None:
    """The boundary is one flat object; a nested value is refused before parsing."""
    with pytest.raises(ValueError, match="nesting"):
        build_default_studio_runtime_settings(
            {
                "SC_NEUROCORE_STUDIO_STORAGE_MODE": "isolated",
                "SC_NEUROCORE_STUDIO_STORAGE_BOUNDARY": nested,
            }
        )


def test_brackets_and_escaped_quotes_inside_strings_are_not_nesting(tmp_path: Path) -> None:
    """Only structure counts: a workspace name may contain brackets and quotes."""
    payload = _configuration(tmp_path)
    payload["workspace"] = 'lab [{"a"}] \\ ]'

    assert _settings(payload).storage_boundary is not None


def test_native_parser_depth_failure_is_a_configuration_error() -> None:
    """Deep nesting is refused by the boundary's own limit on every interpreter.

    CPython 3.12's JSON decoder raised RecursionError at this depth and 3.14's
    decodes it, so the refusal no longer depends on which one runs.
    """
    depth = 10_000
    with pytest.raises(ValueError, match="nesting"):
        build_default_studio_runtime_settings(
            {
                "SC_NEUROCORE_STUDIO_STORAGE_MODE": "isolated",
                "SC_NEUROCORE_STUDIO_STORAGE_BOUNDARY": "[" * depth + "]" * depth,
            }
        )
