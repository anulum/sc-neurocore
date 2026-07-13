# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio policy test support

"""Shared typed helpers for Studio policy contract tests."""

from __future__ import annotations

import hashlib
import json
from datetime import timezone
from typing import Any

import pytest

UTC = timezone.utc


def policy_contract() -> dict[str, Any]:
    """Import the historical Studio policy contract or fail the test."""

    try:
        from sc_neurocore.studio.platform.policy import (  # noqa: PLC0415
            AuditEvent,
            AUDIT_SCHEMA_VERSION,
            AUDIT_QUARANTINE_EXPORT_SCHEMA_VERSION,
            AuditSinkError,
            AuditSinkStatus,
            InMemoryAuditSink,
            JsonlAuditSink,
            PolicyGateway,
            Principal,
            RoutePolicyRegistry,
            RoutePolicy,
            RouteVisibility,
            build_default_studio_route_policy_registry,
        )
    except ImportError as exc:
        pytest.fail(f"Studio policy contract is missing: {exc}")
    return {
        "AuditEvent": AuditEvent,
        "AUDIT_SCHEMA_VERSION": AUDIT_SCHEMA_VERSION,
        "AUDIT_QUARANTINE_EXPORT_SCHEMA_VERSION": AUDIT_QUARANTINE_EXPORT_SCHEMA_VERSION,
        "AuditSinkError": AuditSinkError,
        "AuditSinkStatus": AuditSinkStatus,
        "InMemoryAuditSink": InMemoryAuditSink,
        "JsonlAuditSink": JsonlAuditSink,
        "PolicyGateway": PolicyGateway,
        "Principal": Principal,
        "RoutePolicyRegistry": RoutePolicyRegistry,
        "RoutePolicy": RoutePolicy,
        "RouteVisibility": RouteVisibility,
        "build_default_studio_route_policy_registry": build_default_studio_route_policy_registry,
    }


def audit_event_hash(row: dict[str, Any]) -> str:
    """Return the canonical digest expected for one persisted audit row."""

    unsigned_row = dict(row)
    unsigned_row.pop("event_hash", None)
    canonical_row = json.dumps(
        unsigned_row,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical_row).hexdigest()
