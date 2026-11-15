# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio API runtime builder contracts

"""The Studio API context builder's collaborators and refusals, and no runtime function over 50 lines."""

from __future__ import annotations

import ast
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException

from sc_neurocore.studio.api import runtime
from sc_neurocore.studio.api.runtime import StudioApiContext, build_studio_api_context
from sc_neurocore.studio.platform import (
    InMemoryAuditSink,
    JsonlAuditSink,
    StudioJobManager,
    StudioJobRecord,
    build_default_studio_runtime_settings,
)
from sc_neurocore.studio.platform.jobs_models import StudioJobStatus

MAX_FUNCTION_LINES = 50

EXPOSED_COLLABORATORS = (
    "studio_capabilities",
    "studio_route_policies",
    "studio_audit_sink",
    "studio_identity_authenticator",
    "studio_browser_session_manager",
    "studio_browser_login_throttle",
    "studio_job_manager",
    "studio_policy_gateway",
)


def _function_lengths(path: Path) -> dict[str, int]:
    """Return the line count of every function and method defined in ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: (node.end_lineno or node.lineno) - node.lineno + 1
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _job_manager(context: StudioApiContext) -> StudioJobManager:
    """Return the context's local job manager, which every test here builds."""
    manager = context.studio_job_manager
    assert isinstance(manager, StudioJobManager)
    return manager


@pytest.fixture
def configured_context(tmp_path: Path) -> Iterator[StudioApiContext]:
    """Build the context on a configured job root and close its job manager afterwards.

    Parameters
    ----------
    tmp_path : Path
        The configured job root.

    Yields
    ------
    StudioApiContext
        The context the app was given.
    """
    settings = replace(build_default_studio_runtime_settings(), job_root_path=str(tmp_path))
    context = build_studio_api_context(FastAPI(), settings)
    yield context
    _job_manager(context).close()


def test_every_function_of_the_runtime_module_is_at_most_fifty_lines() -> None:
    """The context builder (116 lines) and the process-job runner (56) are split by responsibility."""
    lengths = _function_lengths(Path(runtime.__file__))
    assert {name: count for name, count in lengths.items() if count > MAX_FUNCTION_LINES} == {}


def test_a_configured_audit_log_path_gives_the_jsonl_sink(tmp_path: Path) -> None:
    """The context records policy decisions to the configured JSONL file.

    Parameters
    ----------
    tmp_path : Path
        Directory for the job root and the audit log.
    """
    settings = replace(
        build_default_studio_runtime_settings(),
        job_root_path=str(tmp_path / "jobs"),
        audit_log_path=str(tmp_path / "audit.jsonl"),
    )
    context = build_studio_api_context(FastAPI(), settings)
    try:
        assert isinstance(context.studio_audit_sink, JsonlAuditSink)
    finally:
        _job_manager(context).close()


def test_without_an_audit_log_path_the_sink_stays_in_memory(
    configured_context: StudioApiContext,
) -> None:
    """Without an audit log path the context keeps its policy decisions in memory.

    Parameters
    ----------
    configured_context : StudioApiContext
        A context built without an audit log path.
    """
    assert isinstance(configured_context.studio_audit_sink, InMemoryAuditSink)


def test_a_configured_job_root_is_the_durable_root(
    configured_context: StudioApiContext, tmp_path: Path
) -> None:
    """A configured job root is the manager's root, and the manager reports it as configured.

    Parameters
    ----------
    configured_context : StudioApiContext
        A context built on ``tmp_path`` as its job root.
    tmp_path : Path
        The configured job root.
    """
    manager = _job_manager(configured_context)
    assert manager.root == tmp_path.resolve()
    assert manager.status().configured is True


def test_without_a_job_root_the_context_gets_a_private_scratch_root() -> None:
    """Without a configured root the job manager runs on this process's scratch root, not configured."""
    context = build_studio_api_context(FastAPI(), build_default_studio_runtime_settings())
    manager = _job_manager(context)
    try:
        assert manager.root.name.startswith("sc-neurocore-studio-jobs-")
        assert manager.status().configured is False
    finally:
        manager.close()


@pytest.mark.parametrize(
    ("status", "code", "detail"),
    [
        ("pending", 503, "studio_job_wait_exceeded"),
        ("running", 503, "studio_job_wait_exceeded"),
        ("cancelling", 503, "studio_job_wait_exceeded"),
        ("timed_out", 504, "studio_job_timed_out"),
        ("failed", 500, "studio_job_failed"),
        ("completed", 500, "studio_job_failed"),
    ],
)
def test_a_process_job_without_a_result_is_refused(
    configured_context: StudioApiContext,
    monkeypatch: pytest.MonkeyPatch,
    status: StudioJobStatus,
    code: int,
    detail: str,
) -> None:
    """A synchronous process job that ends without a result answers with its own HTTP refusal.

    The job manager's submission and wait are replaced by the record a worker
    would leave, so each status reaches the runner without a real worker.

    Parameters
    ----------
    configured_context : StudioApiContext
        The context whose runner is called.
    monkeypatch : MonkeyPatch
        Replaces the manager's submission and wait.
    status : StudioJobStatus
        The status the wait ends with; ``completed`` carries no result.
    code : int
        The expected HTTP status code.
    detail : str
        The expected refusal detail.
    """
    record = StudioJobRecord(
        job_id="runtime-refusal",
        kind="synthesis",
        owner="runtime-test",
        request_id=None,
        status=status,
        execution_model="process",
        created_at_utc="2026-11-14T00:00:00Z",
    )
    manager = _job_manager(configured_context)
    monkeypatch.setattr(manager, "submit_process_task", lambda **_kwargs: record)
    monkeypatch.setattr(manager, "wait", lambda _job_id, timeout_seconds=None: record)
    with pytest.raises(HTTPException) as raised:
        configured_context.run_studio_process_job_sync(
            kind="synthesis",
            owner="runtime-test",
            task_path="tests.studio_job_tasks:process_echo_task",
            payload={},
        )
    assert (raised.value.status_code, raised.value.detail) == (code, detail)


def test_every_collaborator_is_exposed_on_the_app_state(tmp_path: Path) -> None:
    """The routers read the same collaborators from the app state as from the context.

    Parameters
    ----------
    tmp_path : Path
        The configured job root.
    """
    app = FastAPI()
    settings = replace(build_default_studio_runtime_settings(), job_root_path=str(tmp_path))
    context = build_studio_api_context(app, settings)
    try:
        assert app.state.studio_runtime_settings is context.settings
        for name in EXPOSED_COLLABORATORS:
            assert getattr(app.state, name) is getattr(context, name)
    finally:
        _job_manager(context).close()
