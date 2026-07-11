# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FastAPI application factory for Visual SNN Design Studio

"""Compose the Visual SNN Studio FastAPI application from responsibility routers."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from sc_neurocore.studio.api import build_studio_routers
from sc_neurocore.studio.api.frontend import mount_studio_frontend
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
    CodegenRequest,
    CompareRequest,
    CompileRequest,
    DclsEvaluateRequest,
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
    "CodegenRequest",
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
    application = FastAPI(title="SC-NeuroCore Studio", version="1.0.0")
    context = build_studio_api_context(application, runtime_settings)
    settings = context.settings
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


app = create_app()
