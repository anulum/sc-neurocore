# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio policy compatibility facade

"""Historical facade for Studio authorization and tamper-evident audit policy."""

from __future__ import annotations

from sc_neurocore.studio.platform.policy_audit import JsonlAuditSink
from sc_neurocore.studio.platform.policy_gateway import PolicyGateway, RoutePolicyRegistry
from sc_neurocore.studio.platform.policy_models import (
    AUDIT_EXPORT_SCHEMA_VERSION,
    AUDIT_QUARANTINE_EXPORT_SCHEMA_VERSION,
    AUDIT_SCHEMA_VERSION,
    UTC,
    AuditEvent,
    AuditExport,
    AuditExportValue,
    AuditQuarantineExport,
    AuditSink,
    AuditSinkError,
    AuditSinkStatus,
    InMemoryAuditSink,
    PolicyDecision,
    Principal,
    RoutePolicy,
    RouteVisibility,
)
from sc_neurocore.studio.platform.policy_routes import (
    build_default_studio_route_policy_registry,
)

for _historical_symbol in (
    AuditEvent,
    AuditExport,
    AuditQuarantineExport,
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
):
    _historical_symbol.__module__ = __name__
build_default_studio_route_policy_registry.__module__ = __name__
del _historical_symbol

__all__ = [
    "AUDIT_EXPORT_SCHEMA_VERSION",
    "AUDIT_QUARANTINE_EXPORT_SCHEMA_VERSION",
    "AUDIT_SCHEMA_VERSION",
    "AuditEvent",
    "AuditExport",
    "AuditExportValue",
    "AuditQuarantineExport",
    "AuditSink",
    "AuditSinkError",
    "AuditSinkStatus",
    "InMemoryAuditSink",
    "JsonlAuditSink",
    "PolicyDecision",
    "PolicyGateway",
    "Principal",
    "RoutePolicy",
    "RoutePolicyRegistry",
    "RouteVisibility",
    "UTC",
    "build_default_studio_route_policy_registry",
]
