# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio API runtime context

"""Own the state shared by Studio middleware and responsibility routers."""

from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

from fastapi import FastAPI, HTTPException

from sc_neurocore.studio.api.analysis_guards import _analysis_budget_from_settings
from sc_neurocore.studio.platform import (
    AnalysisBudget,
    AuditSink,
    CapabilityRegistry,
    InMemoryAuditSink,
    JsonlAuditSink,
    PolicyGateway,
    RoutePolicyRegistry,
    StudioBrowserLoginThrottle,
    StudioBrowserSessionManager,
    StudioIdentityAuthenticator,
    StudioJobManager,
    StudioProcessJobPayload,
    StudioRuntimeSettings,
    build_default_studio_capability_registry,
    build_default_studio_route_policy_registry,
    build_default_studio_runtime_settings,
    load_studio_identity_store,
)
from sc_neurocore.studio.platform.api_process_lock import hold_identity_store
from sc_neurocore.studio.platform.jobs_models import StudioJobRecord
from sc_neurocore.studio.platform.storage_configuration import StorageBoundaryConfiguration
from sc_neurocore.studio.platform.storage_isolated_runtime import build_isolated_job_manager
from sc_neurocore.studio.platform.storage_launcher_client_settings import LauncherClientSettings
from sc_neurocore.studio.platform.storage_mode import require_available_storage
from sc_neurocore.studio.platform.storage_preflight import require_isolated_preflight
from sc_neurocore.studio.platform.studio_job_service import StudioJobService
from sc_neurocore.studio.synthesis import EdaProcessLimits
from sc_neurocore.studio.project_paths import resolve_studio_project_root

DEFAULT_STUDIO_JOB_KINDS = frozenset(
    {"analysis", "compiler", "evidence", "model_scan", "synthesis", "training"}
)


@dataclass(slots=True)
class StudioApiContext:
    """Hold the collaborators shared across Studio route responsibilities."""

    app: FastAPI
    settings: StudioRuntimeSettings
    analysis_budget: AnalysisBudget
    studio_capabilities: CapabilityRegistry
    studio_route_policies: RoutePolicyRegistry
    studio_audit_sink: AuditSink
    studio_identity_authenticator: StudioIdentityAuthenticator | None
    studio_browser_session_manager: StudioBrowserSessionManager
    studio_browser_login_throttle: StudioBrowserLoginThrottle
    studio_job_manager: StudioJobService
    studio_policy_gateway: PolicyGateway
    eda_process_limits: EdaProcessLimits

    def run_studio_process_job_sync(
        self,
        *,
        kind: str,
        owner: str,
        task_path: str,
        payload: StudioProcessJobPayload,
        include_job_receipt: bool = False,
    ) -> dict[str, Any]:
        """Run one importable task through the bounded process worker.

        Parameters
        ----------
        kind, owner, task_path, payload:
            Existing process admission and execution inputs.
        include_job_receipt:
            Attach a path-free ``studio_job_receipt`` from the exact completed
            record. The default preserves existing caller response shapes.

        Returns
        -------
        dict[str, Any]
            Worker result, optionally with job identity and artifact digests.
            The receipt is response metadata; stored result bytes are unchanged.
        """
        submitted = self.studio_job_manager.submit_process_task(
            kind=kind,
            owner=owner,
            request_id=None,
            task_path=task_path,
            payload=payload,
        )
        completed = self.studio_job_manager.wait(
            submitted.job_id,
            timeout_seconds=self.settings.job_default_timeout_seconds + 1.0,
        )
        if completed.status != "completed" or completed.result is None:
            _refuse_incomplete_job(completed.status)
        if include_job_receipt:
            return {**completed.result, "studio_job_receipt": _job_receipt(completed)}
        return cast(dict[str, Any], completed.result)


def _job_receipt(completed: StudioJobRecord) -> dict[str, Any]:
    """Return the path-free receipt of a completed process job.

    Parameters
    ----------
    completed : StudioJobRecord
        The completed job record.

    Returns
    -------
    dict
        Schema version, job identity, kind, status, and the public digest of
        every artifact.
    """
    return {
        "schema_version": "studio.job-receipt.v1",
        "job_id": completed.job_id,
        "kind": completed.kind,
        "status": completed.status,
        "artifacts": [artifact.to_public_dict() for artifact in completed.artifacts],
    }


def _refuse_incomplete_job(status: str) -> NoReturn:
    """Refuse a synchronous process job that did not complete with a result.

    Parameters
    ----------
    status : str
        The job's status when the wait ended.

    Raises
    ------
    HTTPException
        503 while the job is still pending, running or cancelling; 504 when it
        timed out; 500 otherwise, also for a completed job without a result.
    """
    if status in {"pending", "running", "cancelling"}:
        raise HTTPException(503, "studio_job_wait_exceeded")
    if status == "timed_out":
        raise HTTPException(504, "studio_job_timed_out")
    raise HTTPException(500, "studio_job_failed")


def _remove_scratch_job_root(root: Path, owner_pid: int) -> None:
    """Remove a scratch job root, but only in the process that made it.

    A forked child inherits the exit handlers of its parent; it must not remove
    the root the parent still uses. Process workers start through an exec and
    inherit none.

    Parameters
    ----------
    root : Path
        The scratch root.
    owner_pid : int
        The process that made it.
    """
    if os.getpid() == owner_pid:
        shutil.rmtree(root, ignore_errors=True)


def _scratch_job_root() -> Path:
    """Make this process's private scratch job root and remove it when the process exits.

    An unconfigured root is scratch: no later process reconciles it, and its
    records go with the process. One fixed shared path would hand every Studio
    on the host the same ledger, so each process makes its own.

    Returns
    -------
    Path
        The new scratch root, registered for removal at exit.
    """
    root = Path(tempfile.mkdtemp(prefix="sc-neurocore-studio-jobs-"))
    atexit.register(_remove_scratch_job_root, root, os.getpid())
    return root


def _build_audit_sink(settings: StudioRuntimeSettings) -> AuditSink:
    """Return the JSONL audit sink at the configured path, or an in-memory one.

    Parameters
    ----------
    settings : StudioRuntimeSettings
        The validated runtime settings.

    Returns
    -------
    AuditSink
        The sink that records every policy decision.
    """
    if settings.audit_log_path is None:
        return InMemoryAuditSink()
    return JsonlAuditSink(
        Path(settings.audit_log_path),
        rotation_bytes=settings.audit_rotation_bytes,
        retained_files=settings.audit_retained_files,
    )


def _build_identity(settings: StudioRuntimeSettings) -> StudioIdentityAuthenticator | None:
    """Return the identity authenticator and hold its store, or ``None`` without a store.

    Sessions and login throttles live in this process: one process serves the
    store, and another that tries is refused.

    Parameters
    ----------
    settings : StudioRuntimeSettings
        The validated runtime settings.

    Returns
    -------
    StudioIdentityAuthenticator or None
        The authenticator over the loaded store, or ``None`` when no identity
        file is configured.
    """
    if settings.identity_file_path is None:
        return None
    path = Path(settings.identity_file_path)
    authenticator = StudioIdentityAuthenticator(load_studio_identity_store(path))
    hold_identity_store(path)
    return authenticator


def _build_browser_guards(
    settings: StudioRuntimeSettings,
) -> tuple[StudioBrowserSessionManager, StudioBrowserLoginThrottle]:
    """Return the browser session manager and the login throttle the settings bound.

    Parameters
    ----------
    settings : StudioRuntimeSettings
        The validated runtime settings.

    Returns
    -------
    tuple
        The session manager and the login throttle.
    """
    sessions = StudioBrowserSessionManager(ttl_seconds=settings.browser_session_ttl_seconds)
    throttle = StudioBrowserLoginThrottle(
        max_failed_attempts=settings.browser_login_max_failures,
        failure_window_seconds=settings.browser_login_failure_window_seconds,
        cooldown_seconds=settings.browser_login_cooldown_seconds,
        max_buckets=settings.browser_login_max_buckets,
    )
    return sessions, throttle


def _build_job_manager(
    settings: StudioRuntimeSettings,
    isolated: tuple[StorageBoundaryConfiguration, LauncherClientSettings] | None,
) -> StudioJobService:
    """Return the job manager for the storage mode and the configured root.

    In isolated mode every durable job operation goes to the storage authority,
    and this process owns no job root and no ledger. Otherwise a configured
    root is durable: its ledger outlives the process, and recovery reconciles
    what a departed supervisor left behind. An unconfigured root is scratch,
    private to this process; one fixed path under the system temp directory
    would be shared by every Studio on the host, accumulating another run's
    records, and on a multi-user machine owned by whichever user made it first.

    Parameters
    ----------
    settings : StudioRuntimeSettings
        The validated runtime settings.
    isolated : tuple or None
        The storage boundary and launcher settings the isolated preflight
        accepted, or ``None`` outside isolated mode.

    Returns
    -------
    StudioJobService
        The job manager the routers submit to.
    """
    if isolated is not None:
        return build_isolated_job_manager(
            *isolated,
            allowed_kinds=DEFAULT_STUDIO_JOB_KINDS,
            default_timeout_seconds=settings.job_default_timeout_seconds,
            max_artifact_bytes=settings.job_max_artifact_bytes,
        )
    configured = settings.job_root_path is not None
    return StudioJobManager(
        root=Path(settings.job_root_path)
        if settings.job_root_path is not None
        else _scratch_job_root(),
        allowed_kinds=DEFAULT_STUDIO_JOB_KINDS,
        default_timeout_seconds=settings.job_default_timeout_seconds,
        max_artifact_bytes=settings.job_max_artifact_bytes,
        configured=configured,
    )


def _assemble_context(
    app: FastAPI,
    settings: StudioRuntimeSettings,
    isolated: tuple[StorageBoundaryConfiguration, LauncherClientSettings] | None,
) -> StudioApiContext:
    """Build every collaborator, in the order startup has always made them.

    Parameters
    ----------
    app : FastAPI
        The application the context belongs to.
    settings : StudioRuntimeSettings
        The validated runtime settings.
    isolated : tuple or None
        The accepted isolated preflight, or ``None`` outside isolated mode.

    Returns
    -------
    StudioApiContext
        The shared runtime state.
    """
    analysis_budget = _analysis_budget_from_settings(settings)
    capabilities = build_default_studio_capability_registry()
    route_policies = build_default_studio_route_policy_registry()
    audit_sink = _build_audit_sink(settings)
    identity = _build_identity(settings)
    sessions, throttle = _build_browser_guards(settings)
    return StudioApiContext(
        app=app,
        settings=settings,
        analysis_budget=analysis_budget,
        studio_capabilities=capabilities,
        studio_route_policies=route_policies,
        studio_audit_sink=audit_sink,
        studio_identity_authenticator=identity,
        studio_browser_session_manager=sessions,
        studio_browser_login_throttle=throttle,
        studio_job_manager=_build_job_manager(settings, isolated),
        studio_policy_gateway=PolicyGateway(audit_sink=audit_sink),
        eda_process_limits=EdaProcessLimits(
            cpu_seconds=settings.eda_process_cpu_seconds,
            address_space_bytes=settings.eda_process_memory_bytes,
        ),
    )


def _expose_on_app_state(app: FastAPI, context: StudioApiContext) -> None:
    """Put the settings and every shared collaborator on the app state, where routers read them.

    Parameters
    ----------
    app : FastAPI
        The application whose state receives them.
    context : StudioApiContext
        The shared runtime state.
    """
    app.state.studio_runtime_settings = context.settings
    app.state.studio_capabilities = context.studio_capabilities
    app.state.studio_route_policies = context.studio_route_policies
    app.state.studio_audit_sink = context.studio_audit_sink
    app.state.studio_identity_authenticator = context.studio_identity_authenticator
    app.state.studio_browser_session_manager = context.studio_browser_session_manager
    app.state.studio_browser_login_throttle = context.studio_browser_login_throttle
    app.state.studio_job_manager = context.studio_job_manager
    app.state.studio_policy_gateway = context.studio_policy_gateway


def build_studio_api_context(
    app: FastAPI,
    runtime_settings: StudioRuntimeSettings | None = None,
) -> StudioApiContext:
    """Build and expose the collaborators used by Studio routers.

    Parameters
    ----------
    app:
        FastAPI application receiving the collaborators in its state.
    runtime_settings:
        Optional validated settings override.

    Returns
    -------
    StudioApiContext
        Shared mutable runtime state.

    Raises
    ------
    StudioProjectRootRefused
        The process-wide project directory is unusable. Refuse before creating
        runtime collaborators or persistent state.
    """
    settings = runtime_settings or build_default_studio_runtime_settings()
    resolve_studio_project_root()
    require_available_storage(settings.storage_mode)
    # Isolated startup is checked before any collaborator or file exists.
    isolated = require_isolated_preflight(settings) if settings.storage_mode == "isolated" else None
    context = _assemble_context(app, settings, isolated)
    _expose_on_app_state(app, context)
    return context
