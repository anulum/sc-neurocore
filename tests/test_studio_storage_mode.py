# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Explicit storage mode startup contracts

"""Storage mode selection must not silently downgrade requested isolation."""

from pathlib import Path
import tempfile
from typing import cast

from fastapi import FastAPI
import pytest

from sc_neurocore.studio.api.runtime import build_studio_api_context
from sc_neurocore.studio.platform.settings import (
    StudioRuntimeSettings,
    build_default_studio_runtime_settings,
)
from sc_neurocore.studio.platform.jobs import StudioJobManager
from sc_neurocore.studio.platform.storage_mode import StudioStorageMode


def test_isolated_environment_refuses_before_persistent_effects(tmp_path: Path) -> None:
    """An isolated API that fails its preflight creates no ledger or audit sink."""
    root = tmp_path / "jobs"
    audit = tmp_path / "audit" / "events.jsonl"
    settings = build_default_studio_runtime_settings(
        {
            "SC_NEUROCORE_STUDIO_STORAGE_MODE": "isolated",
            "SC_NEUROCORE_STUDIO_JOB_ROOT": str(root),
            "SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH": str(audit),
        }
    )
    with pytest.raises(RuntimeError, match="isolated preflight failed: boundary"):
        build_studio_api_context(FastAPI(), settings)
    assert not root.exists()
    assert not audit.parent.exists()


@pytest.mark.parametrize("value", ["", " ", "ISOLATED", "remote", "embeddd"])
def test_invalid_mode_never_defaults_to_embedded(value: str) -> None:
    """Malformed operator configuration must not silently reduce isolation."""
    with pytest.raises(ValueError, match="storage mode"):
        build_default_studio_runtime_settings({"SC_NEUROCORE_STUDIO_STORAGE_MODE": value})
    with pytest.raises(ValueError, match="storage mode"):
        StudioRuntimeSettings(storage_mode=cast(StudioStorageMode, value))


def test_direct_isolated_settings_refuse_without_temporary_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The programmatic API enforces the same gate before creating scratch state.

    Temporary files are directed to an empty directory through the standard
    library's ``tempfile.tempdir`` setting; it stays empty.
    """
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    with pytest.raises(RuntimeError, match="isolated preflight failed"):
        build_studio_api_context(FastAPI(), StudioRuntimeSettings(storage_mode="isolated"))
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize("value", [None, "embedded", " embedded "])
def test_embedded_mode_executes_a_real_process_job(tmp_path: Path, value: str | None) -> None:
    """Absent and explicit compatibility selection retain the real worker path."""
    env = {"SC_NEUROCORE_STUDIO_JOB_ROOT": str(tmp_path / "jobs")}
    if value is not None:
        env["SC_NEUROCORE_STUDIO_STORAGE_MODE"] = value
    settings = build_default_studio_runtime_settings(env)
    assert settings.storage_mode == "embedded"
    context = build_studio_api_context(FastAPI(), settings)
    try:
        result = context.run_studio_process_job_sync(
            kind="analysis",
            owner="storage-mode-test",
            task_path="tests.studio_job_tasks:process_echo_task",
            payload={"answer": 42},
        )
        assert result["payload"] == {"answer": 42}
        records = context.studio_job_manager.list_records()
        assert len(records) == 1 and records[0].status == "completed"
        assert result["worker_job_id"] == records[0].job_id
    finally:
        cast(StudioJobManager, context.studio_job_manager)._ledger.close()


def test_an_unknown_mode_bypassing_the_type_is_refused() -> None:
    """A value outside the typed contract never selects either storage."""
    from sc_neurocore.studio.platform.storage_mode import require_available_storage

    with pytest.raises(ValueError, match="embedded or isolated"):
        require_available_storage(cast(StudioStorageMode, "remote"))
