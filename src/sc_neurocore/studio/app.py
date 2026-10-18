# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FastAPI application factory for Visual SNN Design Studio

"""Compose the Visual SNN Studio FastAPI application from responsibility routers."""

from __future__ import annotations

import threading

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from sc_neurocore.studio.api import build_studio_routers
from sc_neurocore.studio.api.common import request_validation_error_handler
from sc_neurocore.studio.api.frontend import mount_studio_api_docs, mount_studio_frontend
from sc_neurocore.studio.api.runtime import (
    DEFAULT_STUDIO_JOB_KINDS,
    build_studio_api_context,
)
from sc_neurocore.studio.api.schemas import (
    AdaptivePrecisionAutoTuneRequest,
    AdaptivePrecisionFormalBundleRequest,
    BenchmarkContributeRequest,
    BenchmarkRunRequest,
    BifurcationRequest,
    CompareRequest,
    CompileRequest,
    DclsEvaluateRequest,
    ExperimentExportRequest,
    FICurveRequest,
    FreqResponseRequest,
    HeatmapRequest,
    ModelSimulateRequest,
    NetworkRequest,
    NullclineRequest,
    PrecisionRequest,
    PresetActionResolveRequest,
    PresetActionsExecuteAllRequest,
    PresetDefaultFlowAttestationVerifyRequest,
    PresetDefaultFlowAttestRequest,
    PresetDefaultFlowGuardedRunRequest,
    PresetDefaultFlowRunFromContractRequest,
    PresetDefaultFlowRunRequest,
    PresetDefaultFlowVerifyRequest,
    SensitivityRequest,
    SimulateRequest,
    StudioAuditQuarantineArchivePurgeRequest,
    StudioAuditQuarantineArchiveRequest,
    StudioAuditQuarantineArchiveRestoreRequest,
    StudioAuditQuarantineArchiveValidateRequest,
    StudioBrowserLoginRequest,
    StudioBrowserUserCreateRequest,
    StudioBrowserUserPasswordRotateRequest,
    StudioBrowserUserUpdateRequest,
    StudioEvidenceBundleRequest,
    StudioIdentityServiceAccountUpdateRequest,
    StudioTrainingWeightAttachRequest,
    StudioTrainingWeightLiveAttachRequest,
    StudioTrainingWeightRestoreRequest,
)
from sc_neurocore.studio.api.security import install_studio_security_middleware
from sc_neurocore.studio.platform import StudioRuntimeSettings

__all__ = [
    "AdaptivePrecisionAutoTuneRequest",
    "AdaptivePrecisionFormalBundleRequest",
    "BenchmarkContributeRequest",
    "BenchmarkRunRequest",
    "BifurcationRequest",
    "ExperimentExportRequest",
    "CompareRequest",
    "CompileRequest",
    "DEFAULT_STUDIO_JOB_KINDS",
    "DclsEvaluateRequest",
    "FICurveRequest",
    "FreqResponseRequest",
    "HeatmapRequest",
    "ModelSimulateRequest",
    "NetworkRequest",
    "NullclineRequest",
    "PrecisionRequest",
    "PresetActionResolveRequest",
    "PresetActionsExecuteAllRequest",
    "PresetDefaultFlowAttestationVerifyRequest",
    "PresetDefaultFlowAttestRequest",
    "PresetDefaultFlowGuardedRunRequest",
    "PresetDefaultFlowRunFromContractRequest",
    "PresetDefaultFlowRunRequest",
    "PresetDefaultFlowVerifyRequest",
    "SensitivityRequest",
    "SimulateRequest",
    "StudioAuditQuarantineArchivePurgeRequest",
    "StudioAuditQuarantineArchiveRequest",
    "StudioAuditQuarantineArchiveRestoreRequest",
    "StudioAuditQuarantineArchiveValidateRequest",
    "StudioBrowserLoginRequest",
    "StudioBrowserUserCreateRequest",
    "StudioBrowserUserPasswordRotateRequest",
    "StudioBrowserUserUpdateRequest",
    "StudioEvidenceBundleRequest",
    "StudioIdentityServiceAccountUpdateRequest",
    "StudioTrainingWeightAttachRequest",
    "StudioTrainingWeightLiveAttachRequest",
    "StudioTrainingWeightRestoreRequest",
    "app",
    "create_app",
]


def create_app(runtime_settings: StudioRuntimeSettings | None = None) -> FastAPI:
    """Create and configure the Visual SNN Studio FastAPI application.

    Parameters
    ----------
    runtime_settings:
        Optional validated runtime settings. Environment-derived settings are
        used when omitted.

    Returns
    -------
    FastAPI
        Application with security middleware and responsibility routers mounted.
    """
    application = FastAPI(
        title="SC-NeuroCore Studio", version="1.0.0", docs_url=None, redoc_url=None
    )
    application.add_exception_handler(RequestValidationError, request_validation_error_handler)
    context = build_studio_api_context(application, runtime_settings)
    settings = context.settings
    if settings.deployment_profile == "development":
        mount_studio_api_docs(application, security_headers=settings.http_security_headers)
    application.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=list(settings.allowed_hosts),
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_allowed_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    install_studio_security_middleware(application, context)
    for router in build_studio_routers(context):
        application.include_router(router)
    mount_studio_frontend(application, app_module_file=__file__)
    return application


app: FastAPI
_default_app: FastAPI | None = None
_default_app_guard = threading.Lock()


def __getattr__(name: str) -> FastAPI:
    """Build the ASGI ``app`` lazily for servers importing the module entry point.

    Factory callers create only the application they requested. Importing the
    factory does not first initialise a second job manager or identity runtime.
    """
    if name != "app":
        raise AttributeError(name)
    global _default_app
    with _default_app_guard:
        if _default_app is None:
            _default_app = create_app()
        return _default_app
