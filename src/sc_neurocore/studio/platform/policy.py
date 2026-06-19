# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio policy gateway

"""Authorization policy contracts for SC-NeuroCore Studio."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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


class InMemoryAuditSink:
    """Append-only in-memory audit sink for local Studio policy tests."""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    @property
    def events(self) -> tuple[AuditEvent, ...]:
        """Return recorded audit events in insertion order."""

        return tuple(self._events)

    def record(self, event: AuditEvent) -> None:
        """Record a Studio policy audit event."""

        self._events.append(event)


class RoutePolicyRegistry:
    """Registry of Studio route policies keyed by HTTP method and path."""

    def __init__(self) -> None:
        self._policies: dict[tuple[str, str], RoutePolicy] = {}

    def register(self, method: str, path_template: str, policy: RoutePolicy) -> None:
        """Register one Studio route policy."""

        key = self._key(method, path_template)
        if key in self._policies:
            raise ValueError(f"{key[0]} {key[1]} already has a Studio route policy.")
        self._policies[key] = policy

    def policy_for(self, method: str, path_template: str) -> RoutePolicy:
        """Return the policy for one HTTP method and path template."""

        return self._policies[self._key(method, path_template)]

    def missing_policies(self, routes: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
        """Return route signatures that have no registered Studio policy."""

        missing: list[str] = []
        for method, path_template in routes:
            key = self._key(method, path_template)
            if key not in self._policies:
                missing.append(f"{key[0]} {key[1]}")
        return tuple(missing)

    @staticmethod
    def _key(method: str, path_template: str) -> tuple[str, str]:
        return method.upper(), path_template


class PolicyGateway:
    """Fail-closed Studio route authorization gateway."""

    def __init__(self, audit_sink: InMemoryAuditSink) -> None:
        self._audit_sink = audit_sink

    def authorize(
        self,
        policy: RoutePolicy,
        *,
        principal: Principal | None,
        route: str,
    ) -> PolicyDecision:
        """Authorize a caller against a Studio route policy."""

        if policy.visibility is RouteVisibility.PUBLIC:
            return PolicyDecision(allowed=True, reason="public_route", status_code=200)

        if principal is None:
            return self._deny(policy, route, principal, "missing_principal", 401)

        if policy.visibility is RouteVisibility.ADMIN and "studio.admin" not in principal.roles:
            return self._deny(policy, route, principal, "missing_admin_role", 403)

        if not policy.required_roles.issubset(principal.roles):
            return self._deny(policy, route, principal, "missing_required_role", 403)

        decision = PolicyDecision(allowed=True, reason="authorized", status_code=200)
        self._record(policy, route, principal, decision)
        return decision

    def _deny(
        self,
        policy: RoutePolicy,
        route: str,
        principal: Principal | None,
        reason: str,
        status_code: int,
    ) -> PolicyDecision:
        decision = PolicyDecision(allowed=False, reason=reason, status_code=status_code)
        self._record(policy, route, principal, decision)
        return decision

    def _record(
        self,
        policy: RoutePolicy,
        route: str,
        principal: Principal | None,
        decision: PolicyDecision,
    ) -> None:
        self._audit_sink.record(
            AuditEvent(
                action=policy.audit_action or "studio.public",
                route=route,
                principal_id=None if principal is None else principal.principal_id,
                decision="allow" if decision.allowed else "deny",
                reason=decision.reason,
            )
        )


def build_default_studio_route_policy_registry() -> RoutePolicyRegistry:
    """Build route policies for the current Studio platform API surface."""

    registry = RoutePolicyRegistry()
    registry.register(
        "GET",
        "/api/health",
        RoutePolicy(
            visibility=RouteVisibility.PUBLIC,
            audit_action="studio.health.read",
        ),
    )
    registry.register(
        "GET",
        "/api/studio/capabilities",
        RoutePolicy(
            visibility=RouteVisibility.PUBLIC,
            audit_action="studio.capabilities.read",
        ),
    )
    registry.register(
        "GET",
        "/api/studio/capabilities/{capability_id}",
        RoutePolicy(
            visibility=RouteVisibility.PUBLIC,
            audit_action="studio.capabilities.read",
        ),
    )
    return registry
