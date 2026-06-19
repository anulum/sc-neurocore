# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio operator status

"""Operator status aggregation for SC-NeuroCore Studio."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sc_neurocore.studio.platform.capabilities import CapabilityHealth, CapabilityStatus
from sc_neurocore.studio.platform.jobs import StudioJobStatusSnapshot
from sc_neurocore.studio.platform.policy import AuditSinkStatus
from sc_neurocore.studio.platform.settings import StudioRuntimeSettings

OPERATOR_STATUS_SCHEMA_VERSION = "studio.operator.status.v1"
OperatorIdentityMode = Literal["service_account", "header_principal", "disabled"]


@dataclass(frozen=True, slots=True)
class StudioOperatorCapabilityStatus:
    """Aggregate health for the Studio capability registry."""

    total_count: int
    healthy_count: int
    degraded_count: int
    unavailable_count: int
    experimental_count: int
    stable_count: int

    def to_public_dict(self) -> dict[str, int]:
        """Return a public capability-health aggregate."""

        return {
            "degraded_count": self.degraded_count,
            "experimental_count": self.experimental_count,
            "healthy_count": self.healthy_count,
            "stable_count": self.stable_count,
            "total_count": self.total_count,
            "unavailable_count": self.unavailable_count,
        }


@dataclass(frozen=True, slots=True)
class StudioOperatorIdentityStatus:
    """Path-free identity posture for Studio operator APIs."""

    configured: bool
    header_principal_allowed: bool
    mode: OperatorIdentityMode

    def to_public_dict(self) -> dict[str, bool | str]:
        """Return public identity posture without service-account material."""

        return {
            "configured": self.configured,
            "header_principal_allowed": self.header_principal_allowed,
            "mode": self.mode,
        }


@dataclass(frozen=True, slots=True)
class StudioOperatorRoutePolicyStatus:
    """Route-policy enforcement posture for Studio operator APIs."""

    enforced: bool

    def to_public_dict(self) -> dict[str, bool]:
        """Return public route-policy posture."""

        return {"enforced": self.enforced}


@dataclass(frozen=True, slots=True)
class StudioOperatorStatus:
    """Path-free aggregate status for the Studio operator control plane."""

    deployment_profile: str
    route_policies: StudioOperatorRoutePolicyStatus
    identity: StudioOperatorIdentityStatus
    audit: AuditSinkStatus
    jobs: StudioJobStatusSnapshot
    capabilities: StudioOperatorCapabilityStatus
    schema_version: str = OPERATOR_STATUS_SCHEMA_VERSION

    def to_public_dict(self) -> dict[str, object]:
        """Return the path-free operator status API payload."""

        return {
            "audit": self.audit.to_public_dict(),
            "capabilities": self.capabilities.to_public_dict(),
            "deployment_profile": self.deployment_profile,
            "identity": self.identity.to_public_dict(),
            "jobs": self.jobs.to_public_dict(),
            "route_policies": self.route_policies.to_public_dict(),
            "schema_version": self.schema_version,
        }


def build_studio_operator_status(
    *,
    settings: StudioRuntimeSettings,
    capabilities: tuple[CapabilityHealth, ...],
    audit_status: AuditSinkStatus,
    job_status: StudioJobStatusSnapshot,
) -> StudioOperatorStatus:
    """Build the aggregate operator status from live Studio platform components."""

    return StudioOperatorStatus(
        deployment_profile=settings.deployment_profile,
        route_policies=StudioOperatorRoutePolicyStatus(enforced=settings.enforce_route_policies),
        identity=_build_identity_status(settings),
        audit=audit_status,
        jobs=job_status,
        capabilities=_build_capability_status(capabilities),
    )


def _build_identity_status(settings: StudioRuntimeSettings) -> StudioOperatorIdentityStatus:
    if settings.identity_file_path is not None:
        mode: OperatorIdentityMode = "service_account"
    elif settings.allow_header_principal:
        mode = "header_principal"
    else:
        mode = "disabled"
    return StudioOperatorIdentityStatus(
        configured=settings.identity_file_path is not None,
        header_principal_allowed=settings.allow_header_principal,
        mode=mode,
    )


def _build_capability_status(
    capabilities: tuple[CapabilityHealth, ...],
) -> StudioOperatorCapabilityStatus:
    return StudioOperatorCapabilityStatus(
        total_count=len(capabilities),
        healthy_count=sum(capability.healthy for capability in capabilities),
        degraded_count=sum(
            capability.status is CapabilityStatus.DEGRADED for capability in capabilities
        ),
        unavailable_count=sum(
            capability.status is CapabilityStatus.UNAVAILABLE for capability in capabilities
        ),
        experimental_count=sum(
            capability.status is CapabilityStatus.EXPERIMENTAL for capability in capabilities
        ),
        stable_count=sum(
            capability.status is CapabilityStatus.STABLE for capability in capabilities
        ),
    )


__all__ = [
    "OPERATOR_STATUS_SCHEMA_VERSION",
    "OperatorIdentityMode",
    "StudioOperatorCapabilityStatus",
    "StudioOperatorIdentityStatus",
    "StudioOperatorRoutePolicyStatus",
    "StudioOperatorStatus",
    "build_studio_operator_status",
]
