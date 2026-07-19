# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio API runtime context

"""Own the state shared by Studio middleware and responsibility routers."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

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
from sc_neurocore.studio.synthesis import EdaProcessLimits


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
    studio_job_manager: StudioJobManager
    studio_policy_gateway: PolicyGateway
    eda_process_limits: EdaProcessLimits

    def run_studio_process_job_sync(
        self,
        *,
        kind: str,
        owner: str,
        task_path: str,
        payload: StudioProcessJobPayload,
    ) -> dict[str, Any]:
        """Run one importable task through the bounded process worker."""
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
        if completed.status == "completed" and completed.result is not None:
            return cast(dict[str, Any], completed.result)
        if completed.status in {"pending", "running", "cancelling"}:
            raise HTTPException(503, "studio_job_wait_exceeded")
        if completed.status == "timed_out":
            raise HTTPException(504, "studio_job_timed_out")
        raise HTTPException(500, "studio_job_failed")


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
    """
    settings = runtime_settings or build_default_studio_runtime_settings()
    analysis_budget = _analysis_budget_from_settings(settings)
    studio_capabilities = build_default_studio_capability_registry()
    studio_route_policies = build_default_studio_route_policy_registry()
    studio_audit_sink: AuditSink = (
        JsonlAuditSink(
            Path(settings.audit_log_path),
            rotation_bytes=settings.audit_rotation_bytes,
            retained_files=settings.audit_retained_files,
        )
        if settings.audit_log_path is not None
        else InMemoryAuditSink()
    )
    studio_identity_authenticator = (
        StudioIdentityAuthenticator(load_studio_identity_store(Path(settings.identity_file_path)))
        if settings.identity_file_path is not None
        else None
    )
    studio_browser_session_manager = StudioBrowserSessionManager(
        ttl_seconds=settings.browser_session_ttl_seconds
    )
    studio_browser_login_throttle = StudioBrowserLoginThrottle(
        max_failed_attempts=settings.browser_login_max_failures,
        failure_window_seconds=settings.browser_login_failure_window_seconds,
        cooldown_seconds=settings.browser_login_cooldown_seconds,
    )
    studio_job_root = (
        Path(settings.job_root_path)
        if settings.job_root_path is not None
        else Path(tempfile.gettempdir()) / "sc-neurocore-studio-jobs"
    )
    studio_job_manager = StudioJobManager(
        root=studio_job_root,
        allowed_kinds=DEFAULT_STUDIO_JOB_KINDS,
        default_timeout_seconds=settings.job_default_timeout_seconds,
        max_artifact_bytes=settings.job_max_artifact_bytes,
        configured=settings.job_root_path is not None,
    )
    studio_policy_gateway = PolicyGateway(audit_sink=studio_audit_sink)
    context = StudioApiContext(
        app=app,
        settings=settings,
        analysis_budget=analysis_budget,
        studio_capabilities=studio_capabilities,
        studio_route_policies=studio_route_policies,
        studio_audit_sink=studio_audit_sink,
        studio_identity_authenticator=studio_identity_authenticator,
        studio_browser_session_manager=studio_browser_session_manager,
        studio_browser_login_throttle=studio_browser_login_throttle,
        studio_job_manager=studio_job_manager,
        studio_policy_gateway=studio_policy_gateway,
        eda_process_limits=EdaProcessLimits(
            cpu_seconds=settings.eda_process_cpu_seconds,
            address_space_bytes=settings.eda_process_memory_bytes,
        ),
    )
    app.state.studio_runtime_settings = settings
    app.state.studio_capabilities = studio_capabilities
    app.state.studio_route_policies = studio_route_policies
    app.state.studio_audit_sink = studio_audit_sink
    app.state.studio_identity_authenticator = studio_identity_authenticator
    app.state.studio_browser_session_manager = studio_browser_session_manager
    app.state.studio_browser_login_throttle = studio_browser_login_throttle
    app.state.studio_job_manager = studio_job_manager
    app.state.studio_policy_gateway = studio_policy_gateway
    return context
