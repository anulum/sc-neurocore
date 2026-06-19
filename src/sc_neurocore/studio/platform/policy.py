# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio policy gateway

"""Authorization policy contracts for SC-NeuroCore Studio."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Protocol


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
            "timestamp_utc": self.timestamp_utc,
        }


class AuditSink(Protocol):
    """Append-only sink for Studio policy audit events."""

    def record(self, event: AuditEvent) -> None:
        """Record a Studio policy audit event."""


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


class JsonlAuditSink:
    """Append-only JSONL audit sink for Studio policy decisions."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        """Return the configured JSONL audit log path."""

        return self._path

    def record(self, event: AuditEvent) -> None:
        """Append a Studio policy audit event as one JSON object."""

        self._path.parent.mkdir(parents=True, exist_ok=True)
        row = json.dumps(
            event.to_json_dict(),
            separators=(",", ":"),
            sort_keys=True,
        )
        with self._path.open("a", encoding="utf-8") as audit_file:
            audit_file.write(f"{row}\n")


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

    def __init__(
        self,
        audit_sink: AuditSink,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._audit_sink = audit_sink
        self._clock = clock or self._utc_now

    def authorize(
        self,
        policy: RoutePolicy,
        *,
        principal: Principal | None,
        route: str,
        request_id: str | None = None,
    ) -> PolicyDecision:
        """Authorize a caller against a Studio route policy."""

        if policy.visibility is RouteVisibility.PUBLIC:
            return PolicyDecision(allowed=True, reason="public_route", status_code=200)

        if principal is None:
            return self._deny(policy, route, principal, "missing_principal", 401, request_id)

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
        timestamp = self._clock().astimezone(UTC).replace(microsecond=0)
        return timestamp.isoformat().replace("+00:00", "Z")

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(UTC)


def _register_routes(
    registry: RoutePolicyRegistry,
    routes: tuple[tuple[str, str, RouteVisibility, str], ...],
) -> None:
    for method, path_template, visibility, audit_action in routes:
        registry.register(
            method,
            path_template,
            RoutePolicy(visibility=visibility, audit_action=audit_action),
        )


def build_default_studio_route_policy_registry() -> RoutePolicyRegistry:
    """Build route policies for the current Studio platform API surface."""

    registry = RoutePolicyRegistry()
    _register_routes(
        registry,
        (
            ("GET", "/api/health", RouteVisibility.PUBLIC, "studio.health.read"),
            (
                "GET",
                "/api/studio/capabilities",
                RouteVisibility.PUBLIC,
                "studio.capabilities.read",
            ),
            (
                "GET",
                "/api/studio/capabilities/{capability_id}",
                RouteVisibility.PUBLIC,
                "studio.capabilities.read",
            ),
            ("GET", "/api/templates", RouteVisibility.PUBLIC, "studio.templates.read"),
            ("GET", "/api/templates/{name}", RouteVisibility.PUBLIC, "studio.templates.read"),
            ("GET", "/api/models", RouteVisibility.PUBLIC, "studio.models.read"),
            ("GET", "/api/models/{name}", RouteVisibility.PUBLIC, "studio.models.read"),
            ("GET", "/api/models/scan", RouteVisibility.PUBLIC, "studio.models.scan"),
            ("GET", "/api/presets", RouteVisibility.PUBLIC, "studio.presets.read"),
            (
                "GET",
                "/api/presets/{preset_id}",
                RouteVisibility.PUBLIC,
                "studio.presets.read",
            ),
            (
                "GET",
                "/api/presets/actions/catalog",
                RouteVisibility.PUBLIC,
                "studio.presets.actions.catalog",
            ),
            (
                "GET",
                "/api/presets/{preset_id}/actions",
                RouteVisibility.PUBLIC,
                "studio.presets.actions.read",
            ),
            (
                "GET",
                "/api/presets/{preset_id}/default-flow/plan",
                RouteVisibility.PUBLIC,
                "studio.presets.default_flow.plan.read",
            ),
            (
                "GET",
                "/api/presets/{preset_id}/default-flow/contract",
                RouteVisibility.PUBLIC,
                "studio.presets.default_flow.contract.read",
            ),
            ("GET", "/api/cache/stats", RouteVisibility.PUBLIC, "studio.cache.stats.read"),
            (
                "GET",
                "/api/synth/tools-status",
                RouteVisibility.PUBLIC,
                "studio.synth.tools.read",
            ),
            (
                "GET",
                "/api/training/surrogates",
                RouteVisibility.PUBLIC,
                "studio.training.surrogates.read",
            ),
            (
                "GET",
                "/api/training/cell-types",
                RouteVisibility.PUBLIC,
                "studio.training.cell_types.read",
            ),
            ("GET", "/api/graph/models", RouteVisibility.PUBLIC, "studio.graph.models.read"),
            (
                "POST",
                "/api/simulate",
                RouteVisibility.AUTHENTICATED,
                "studio.simulation.run",
            ),
            (
                "POST",
                "/api/models/simulate",
                RouteVisibility.AUTHENTICATED,
                "studio.models.simulate",
            ),
            (
                "POST",
                "/api/multi-simulate",
                RouteVisibility.AUTHENTICATED,
                "studio.models.multi_simulate",
            ),
            (
                "POST",
                "/api/characterize",
                RouteVisibility.AUTHENTICATED,
                "studio.models.characterize",
            ),
            (
                "POST",
                "/api/classify",
                RouteVisibility.AUTHENTICATED,
                "studio.models.classify",
            ),
            ("POST", "/api/fi-curve", RouteVisibility.AUTHENTICATED, "studio.analysis.fi_curve"),
            (
                "POST",
                "/api/bifurcation",
                RouteVisibility.AUTHENTICATED,
                "studio.analysis.bifurcation",
            ),
            (
                "POST",
                "/api/sensitivity",
                RouteVisibility.AUTHENTICATED,
                "studio.analysis.sensitivity",
            ),
            (
                "POST",
                "/api/nullclines",
                RouteVisibility.AUTHENTICATED,
                "studio.analysis.nullclines",
            ),
            ("POST", "/api/precision", RouteVisibility.AUTHENTICATED, "studio.analysis.precision"),
            (
                "POST",
                "/api/freq-response",
                RouteVisibility.AUTHENTICATED,
                "studio.analysis.freq_response",
            ),
            ("POST", "/api/heatmap", RouteVisibility.AUTHENTICATED, "studio.analysis.heatmap"),
            (
                "POST",
                "/api/compare",
                RouteVisibility.AUTHENTICATED,
                "studio.analysis.compare",
            ),
            (
                "POST",
                "/api/import-trace",
                RouteVisibility.AUTHENTICATED,
                "studio.trace.import",
            ),
            ("POST", "/api/codegen", RouteVisibility.AUTHENTICATED, "studio.codegen.generate"),
            ("POST", "/api/compile", RouteVisibility.AUTHENTICATED, "studio.compiler.compile"),
            (
                "POST",
                "/api/adaptive-precision/auto-tune",
                RouteVisibility.AUTHENTICATED,
                "studio.compiler.adaptive_precision.auto_tune",
            ),
            (
                "POST",
                "/api/adaptive-precision/formal-bundle",
                RouteVisibility.AUTHENTICATED,
                "studio.compiler.adaptive_precision.formal_bundle",
            ),
            ("POST", "/api/ir/build", RouteVisibility.AUTHENTICATED, "studio.ir.build"),
            ("POST", "/api/ir/verify", RouteVisibility.AUTHENTICATED, "studio.ir.verify"),
            ("POST", "/api/ir/emit-sv", RouteVisibility.AUTHENTICATED, "studio.ir.emit_sv"),
            (
                "POST",
                "/api/ir/emit-sv-direct",
                RouteVisibility.AUTHENTICATED,
                "studio.ir.emit_sv_direct",
            ),
            ("POST", "/api/ir/cosim", RouteVisibility.AUTHENTICATED, "studio.ir.cosim"),
            (
                "POST",
                "/api/synth/estimate",
                RouteVisibility.AUTHENTICATED,
                "studio.synth.estimate",
            ),
            ("POST", "/api/synth/run", RouteVisibility.ADMIN, "studio.synth.run"),
            (
                "POST",
                "/api/synth/multi-target",
                RouteVisibility.ADMIN,
                "studio.synth.multi_target",
            ),
            ("POST", "/api/synth/pnr", RouteVisibility.ADMIN, "studio.synth.pnr"),
            ("POST", "/api/export/svg", RouteVisibility.AUTHENTICATED, "studio.export.svg"),
            (
                "POST",
                "/api/network/ei",
                RouteVisibility.AUTHENTICATED,
                "studio.network.ei.simulate",
            ),
            (
                "POST",
                "/api/graph/population",
                RouteVisibility.AUTHENTICATED,
                "studio.graph.population.create",
            ),
            (
                "POST",
                "/api/graph/projection",
                RouteVisibility.AUTHENTICATED,
                "studio.graph.projection.create",
            ),
            (
                "POST",
                "/api/graph/validate",
                RouteVisibility.AUTHENTICATED,
                "studio.graph.validate",
            ),
            (
                "POST",
                "/api/graph/simulate",
                RouteVisibility.AUTHENTICATED,
                "studio.graph.simulate",
            ),
            (
                "POST",
                "/api/graph/export-nir",
                RouteVisibility.AUTHENTICATED,
                "studio.graph.export_nir",
            ),
            (
                "POST",
                "/api/graph/import-nir",
                RouteVisibility.AUTHENTICATED,
                "studio.graph.import_nir",
            ),
            (
                "GET",
                "/api/training/jobs",
                RouteVisibility.AUTHENTICATED,
                "studio.training.jobs.read",
            ),
            (
                "GET",
                "/api/training/status/{job_id}",
                RouteVisibility.AUTHENTICATED,
                "studio.training.status.read",
            ),
            (
                "GET",
                "/api/training/stream/{job_id}",
                RouteVisibility.AUTHENTICATED,
                "studio.training.stream",
            ),
            (
                "POST",
                "/api/training/start",
                RouteVisibility.AUTHENTICATED,
                "studio.training.start",
            ),
            (
                "POST",
                "/api/training/stop",
                RouteVisibility.AUTHENTICATED,
                "studio.training.stop",
            ),
            (
                "GET",
                "/api/project/list",
                RouteVisibility.AUTHENTICATED,
                "studio.project.list",
            ),
            (
                "GET",
                "/api/project/load/{name}",
                RouteVisibility.AUTHENTICATED,
                "studio.project.load",
            ),
            (
                "POST",
                "/api/project/save",
                RouteVisibility.AUTHENTICATED,
                "studio.project.save",
            ),
            (
                "DELETE",
                "/api/project/{name}",
                RouteVisibility.AUTHENTICATED,
                "studio.project.delete",
            ),
            (
                "POST",
                "/api/pipeline/run",
                RouteVisibility.AUTHENTICATED,
                "studio.pipeline.run",
            ),
            (
                "POST",
                "/api/presets/{preset_id}/actions/{action_id}/resolve",
                RouteVisibility.AUTHENTICATED,
                "studio.presets.actions.resolve",
            ),
            (
                "POST",
                "/api/presets/{preset_id}/actions/{action_id}/execute",
                RouteVisibility.AUTHENTICATED,
                "studio.presets.actions.execute",
            ),
            (
                "POST",
                "/api/presets/{preset_id}/actions/execute-all",
                RouteVisibility.AUTHENTICATED,
                "studio.presets.actions.execute_all",
            ),
            (
                "POST",
                "/api/presets/{preset_id}/default-flow/run",
                RouteVisibility.AUTHENTICATED,
                "studio.presets.default_flow.run",
            ),
            (
                "POST",
                "/api/presets/{preset_id}/default-flow/verify",
                RouteVisibility.AUTHENTICATED,
                "studio.presets.default_flow.verify",
            ),
            (
                "POST",
                "/api/presets/{preset_id}/default-flow/run-guarded",
                RouteVisibility.AUTHENTICATED,
                "studio.presets.default_flow.run_guarded",
            ),
            (
                "POST",
                "/api/presets/{preset_id}/default-flow/run-from-contract",
                RouteVisibility.AUTHENTICATED,
                "studio.presets.default_flow.run_from_contract",
            ),
            (
                "POST",
                "/api/presets/{preset_id}/default-flow/attest",
                RouteVisibility.AUTHENTICATED,
                "studio.presets.default_flow.attest",
            ),
            (
                "POST",
                "/api/presets/{preset_id}/default-flow/attest/verify",
                RouteVisibility.AUTHENTICATED,
                "studio.presets.default_flow.attest_verify",
            ),
            (
                "WEBSOCKET",
                "/ws/progress",
                RouteVisibility.AUTHENTICATED,
                "studio.websocket.progress",
            ),
        ),
    )
    return registry
