# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio policy data contracts

"""Immutable contracts shared by Studio policy and audit components."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timezone
from enum import Enum
from typing import Protocol

AUDIT_SCHEMA_VERSION = "studio.audit.v1"
AUDIT_EXPORT_SCHEMA_VERSION = "studio.audit.export.v1"
AUDIT_QUARANTINE_EXPORT_SCHEMA_VERSION = "studio.audit.quarantine.export.v1"
UTC = timezone.utc


class AuditSinkError(RuntimeError):
    """Raised when a Studio audit sink cannot persist an event."""


@dataclass(frozen=True, slots=True)
class AuditSinkStatus:
    """Operator-safe status for a Studio audit sink."""

    configured: bool
    healthy: bool
    path_configured: bool
    sink_type: str
    integrity_verified: bool | None = None
    last_error: str | None = None
    latest_event_hash: str | None = None
    integrity_error: str | None = None
    quarantined_event_count: int | None = None
    quarantine_reason: str | None = None
    retained_event_count: int | None = None

    def to_public_dict(self) -> dict[str, bool | int | str | None]:
        """Return an operator-safe status dictionary without local paths."""

        return {
            "configured": self.configured,
            "healthy": self.healthy,
            "integrity_error": self.integrity_error,
            "integrity_verified": self.integrity_verified,
            "last_error": self.last_error,
            "latest_event_hash": self.latest_event_hash,
            "path_configured": self.path_configured,
            "quarantine_reason": self.quarantine_reason,
            "quarantined_event_count": self.quarantined_event_count,
            "retained_event_count": self.retained_event_count,
            "sink_type": self.sink_type,
        }


AuditExportValue = bool | int | str | None | list[dict[str, str | None]]


@dataclass(frozen=True, slots=True)
class AuditExport:
    """Operator-safe export of persisted Studio audit rows."""

    configured: bool
    sink_type: str
    event_count: int
    truncated: bool
    events: tuple[dict[str, str | None], ...]
    integrity_verified: bool
    latest_event_hash: str | None
    integrity_error: str | None = None
    quarantined_event_count: int = 0
    quarantine_reason: str | None = None
    retained_event_count: int = 0
    schema_version: str = AUDIT_EXPORT_SCHEMA_VERSION

    def to_public_dict(self) -> dict[str, AuditExportValue]:
        """Return a path-free JSON export payload for admin operators."""

        return {
            "configured": self.configured,
            "event_count": self.event_count,
            "events": [dict(event) for event in self.events],
            "integrity_error": self.integrity_error,
            "integrity_verified": self.integrity_verified,
            "latest_event_hash": self.latest_event_hash,
            "quarantine_reason": self.quarantine_reason,
            "quarantined_event_count": self.quarantined_event_count,
            "retained_event_count": self.retained_event_count,
            "schema_version": self.schema_version,
            "sink_type": self.sink_type,
            "truncated": self.truncated,
        }


@dataclass(frozen=True, slots=True)
class AuditQuarantineExport:
    """Operator-safe export of quarantined retained Studio audit rows."""

    configured: bool
    sink_type: str
    event_count: int
    truncated: bool
    events: tuple[dict[str, str | None], ...]
    retained_event_count: int
    quarantine_reason: str | None
    schema_version: str = AUDIT_QUARANTINE_EXPORT_SCHEMA_VERSION

    def to_public_dict(self) -> dict[str, AuditExportValue]:
        """Return path-free quarantined audit rows for incident handoff."""

        return {
            "configured": self.configured,
            "event_count": self.event_count,
            "events": [dict(event) for event in self.events],
            "quarantine_reason": self.quarantine_reason,
            "retained_event_count": self.retained_event_count,
            "schema_version": self.schema_version,
            "sink_type": self.sink_type,
            "truncated": self.truncated,
        }


class RouteVisibility(str, Enum):
    """Visibility class for a Studio API route."""

    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    ADMIN = "admin"


@dataclass(frozen=True, slots=True)
class Principal:
    """Authenticated Studio caller identity."""

    principal_id: str
    roles: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class RoutePolicy:
    """Authorization policy attached to one Studio route."""

    visibility: RouteVisibility
    required_roles: frozenset[str] = field(default_factory=frozenset)
    audit_action: str | None = None

    def __post_init__(self) -> None:
        """Validate fail-closed policy metadata."""

        if self.visibility is RouteVisibility.PUBLIC:
            return
        if self.audit_action is None or not self.audit_action.strip():
            raise ValueError("Protected Studio route policies require audit_action.")


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Authorization decision returned by the Studio policy gateway."""

    allowed: bool
    reason: str
    status_code: int


@dataclass(frozen=True, slots=True)
class AuditEvent:
    """Audit event emitted for Studio policy decisions."""

    action: str
    route: str
    principal_id: str | None
    decision: str
    reason: str
    request_id: str | None = None
    schema_version: str = AUDIT_SCHEMA_VERSION
    timestamp_utc: str | None = None

    def to_json_dict(self) -> dict[str, str | None]:
        """Return a JSON-serializable representation of the audit event."""

        return {
            "action": self.action,
            "decision": self.decision,
            "principal_id": self.principal_id,
            "reason": self.reason,
            "request_id": self.request_id,
            "route": self.route,
            "schema_version": self.schema_version,
            "timestamp_utc": self.timestamp_utc,
        }


class AuditSink(Protocol):
    """Append-only sink for Studio policy audit events."""

    def record(self, event: AuditEvent) -> None:
        """Record a Studio policy audit event."""

    def status(self) -> AuditSinkStatus:
        """Return operator-safe audit sink status."""


class InMemoryAuditSink:
    """Append-only in-memory audit sink for local Studio policy tests."""

    def __init__(self) -> None:
        """Create an empty process-local audit event buffer."""

        self._events: list[AuditEvent] = []

    @property
    def events(self) -> tuple[AuditEvent, ...]:
        """Return recorded audit events in insertion order."""

        return tuple(self._events)

    def record(self, event: AuditEvent) -> None:
        """Record a Studio policy audit event."""

        self._events.append(event)

    def status(self) -> AuditSinkStatus:
        """Return status for the non-persistent in-memory audit sink."""

        return AuditSinkStatus(
            configured=False,
            healthy=True,
            integrity_verified=None,
            path_configured=False,
            sink_type="memory",
        )


@dataclass(frozen=True, slots=True)
class _AuditIntegrityReport:
    """Path-free integrity status for retained JSONL audit rows."""

    verified: bool
    error: str | None
    latest_event_hash: str | None
    retained_event_count: int = 0
    quarantined_event_count: int = 0
    quarantine_reason: str | None = None
