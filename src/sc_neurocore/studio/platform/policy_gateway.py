# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio authorization gateway

"""Fail-closed route-policy registry and authorization gateway."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from sc_neurocore.studio.platform.policy_models import (
    UTC,
    AuditEvent,
    AuditSink,
    PolicyDecision,
    Principal,
    RoutePolicy,
    RouteVisibility,
)


class RoutePolicyRegistry:
    """Registry of Studio route policies keyed by HTTP method and path."""

    def __init__(self) -> None:
        """Create an empty method-and-path policy registry."""

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

    def policies(self) -> tuple[tuple[str, str, RoutePolicy], ...]:
        """Return registered route policies in stable method/path order."""

        return tuple(
            (method, path_template, policy)
            for (method, path_template), policy in sorted(self._policies.items())
        )

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
        """Normalize one route signature for case-stable lookup."""

        return method.upper(), path_template


class PolicyGateway:
    """Fail-closed Studio route authorization gateway."""

    def __init__(
        self,
        audit_sink: AuditSink,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Bind the required audit sink and optional deterministic clock."""

        self._audit_sink = audit_sink
        self._clock = clock or self._utc_now

    def authorize(
        self,
        policy: RoutePolicy,
        *,
        principal: Principal | None,
        route: str,
        request_id: str | None = None,
        identity_failure_reason: str | None = None,
    ) -> PolicyDecision:
        """Authorize a caller against a Studio route policy."""

        if policy.visibility is RouteVisibility.PUBLIC:
            return PolicyDecision(allowed=True, reason="public_route", status_code=200)

        if principal is None:
            reason = identity_failure_reason or "missing_principal"
            return self._deny(policy, route, principal, reason, 401, request_id)

        if policy.visibility is RouteVisibility.ADMIN and "studio.admin" not in principal.roles:
            return self._deny(policy, route, principal, "missing_admin_role", 403, request_id)

        if not policy.required_roles.issubset(principal.roles):
            return self._deny(policy, route, principal, "missing_required_role", 403, request_id)

        decision = PolicyDecision(allowed=True, reason="authorized", status_code=200)
        self._record(policy, route, principal, decision, request_id)
        return decision

    def _deny(
        self,
        policy: RoutePolicy,
        route: str,
        principal: Principal | None,
        reason: str,
        status_code: int,
        request_id: str | None,
    ) -> PolicyDecision:
        """Record and return a fail-closed authorization decision."""

        decision = PolicyDecision(allowed=False, reason=reason, status_code=status_code)
        self._record(policy, route, principal, decision, request_id)
        return decision

    def _record(
        self,
        policy: RoutePolicy,
        route: str,
        principal: Principal | None,
        decision: PolicyDecision,
        request_id: str | None,
    ) -> None:
        """Persist one path-free authorization decision as audit evidence."""

        self._audit_sink.record(
            AuditEvent(
                action=policy.audit_action or "studio.public",
                route=route,
                principal_id=None if principal is None else principal.principal_id,
                decision="allow" if decision.allowed else "deny",
                reason=decision.reason,
                request_id=request_id,
                timestamp_utc=self._timestamp_utc(),
            )
        )

    def _timestamp_utc(self) -> str:
        """Return the injected clock value as a whole-second UTC timestamp."""

        timestamp = self._clock().astimezone(UTC).replace(microsecond=0)
        return timestamp.isoformat().replace("+00:00", "Z")

    @staticmethod
    def _utc_now() -> datetime:
        """Return the current timezone-aware UTC time."""

        return datetime.now(UTC)
