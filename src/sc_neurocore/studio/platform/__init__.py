# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio platform contracts

"""Platform contracts for SC-NeuroCore Studio."""

from sc_neurocore.studio.platform.capabilities import (
    CapabilityDescriptor,
    CapabilityHealth,
    CapabilityRegistry,
    CapabilityRequirement,
    CapabilityStatus,
    EvidenceClass,
    build_default_studio_capability_registry,
)
from sc_neurocore.studio.platform.policy import (
    AUDIT_SCHEMA_VERSION,
    AUDIT_EXPORT_SCHEMA_VERSION,
    AuditExport,
    AuditExportValue,
    AuditEvent,
    AuditSink,
    AuditSinkError,
    AuditSinkStatus,
    InMemoryAuditSink,
    JsonlAuditSink,
    PolicyDecision,
    PolicyGateway,
    Principal,
    RoutePolicy,
    RoutePolicyRegistry,
    RouteVisibility,
    build_default_studio_route_policy_registry,
)
from sc_neurocore.studio.platform.identity import (
    IDENTITY_SCHEMA_VERSION,
    StudioIdentityAuthenticator,
    StudioIdentityRecord,
    StudioIdentityResult,
    StudioIdentityStore,
    load_studio_identity_store,
)
from sc_neurocore.studio.platform.settings import (
    DEFAULT_STUDIO_AUDIT_RETAINED_FILES,
    DEFAULT_STUDIO_ALLOWED_HOSTS,
    DEFAULT_STUDIO_CORS_ORIGINS,
    DEFAULT_STUDIO_HTTP_SECURITY_HEADERS,
    DEFAULT_STUDIO_MAX_REQUEST_BODY_BYTES,
    DEFAULT_STUDIO_WEBSOCKET_ALLOWED_ORIGINS,
    StudioRuntimeSettings,
    build_default_studio_runtime_settings,
)

__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "AUDIT_EXPORT_SCHEMA_VERSION",
    "AuditExport",
    "AuditExportValue",
    "AuditEvent",
    "AuditSink",
    "AuditSinkError",
    "AuditSinkStatus",
    "CapabilityDescriptor",
    "CapabilityHealth",
    "CapabilityRegistry",
    "CapabilityRequirement",
    "CapabilityStatus",
    "EvidenceClass",
    "IDENTITY_SCHEMA_VERSION",
    "InMemoryAuditSink",
    "JsonlAuditSink",
    "PolicyDecision",
    "PolicyGateway",
    "Principal",
    "RoutePolicy",
    "RoutePolicyRegistry",
    "RouteVisibility",
    "DEFAULT_STUDIO_AUDIT_RETAINED_FILES",
    "DEFAULT_STUDIO_ALLOWED_HOSTS",
    "DEFAULT_STUDIO_CORS_ORIGINS",
    "DEFAULT_STUDIO_HTTP_SECURITY_HEADERS",
    "DEFAULT_STUDIO_MAX_REQUEST_BODY_BYTES",
    "DEFAULT_STUDIO_WEBSOCKET_ALLOWED_ORIGINS",
    "StudioRuntimeSettings",
    "StudioIdentityAuthenticator",
    "StudioIdentityRecord",
    "StudioIdentityResult",
    "StudioIdentityStore",
    "build_default_studio_capability_registry",
    "build_default_studio_route_policy_registry",
    "build_default_studio_runtime_settings",
    "load_studio_identity_store",
]
