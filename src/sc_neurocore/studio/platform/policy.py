# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio policy gateway

"""Authorization policy contracts for SC-NeuroCore Studio."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Protocol

AUDIT_SCHEMA_VERSION = "studio.audit.v1"
AUDIT_EXPORT_SCHEMA_VERSION = "studio.audit.export.v1"
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
    last_error: str | None = None

    def to_public_dict(self) -> dict[str, bool | str | None]:
        """Return an operator-safe status dictionary without local paths."""

        return {
            "configured": self.configured,
            "healthy": self.healthy,
            "last_error": self.last_error,
            "path_configured": self.path_configured,
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
    schema_version: str = AUDIT_EXPORT_SCHEMA_VERSION

    def to_public_dict(self) -> dict[str, AuditExportValue]:
        """Return a path-free JSON export payload for admin operators."""

        return {
            "configured": self.configured,
            "event_count": self.event_count,
            "events": [dict(event) for event in self.events],
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
            path_configured=False,
            sink_type="memory",
        )


class JsonlAuditSink:
    """Append-only JSONL audit sink for Studio policy decisions."""

    def __init__(
        self,
        path: Path,
        *,
        rotation_bytes: int | None = None,
        retained_files: int = 5,
    ) -> None:
        self._path = path
        self._rotation_bytes = rotation_bytes
        self._retained_files = retained_files
        self._last_error: str | None = None

    @property
    def path(self) -> Path:
        """Return the configured JSONL audit log path."""

        return self._path

    def record(self, event: AuditEvent) -> None:
        """Append a Studio policy audit event as one JSON object."""

        try:
            preflight_error = self._preflight_error()
            if preflight_error is not None:
                self._last_error = preflight_error
                raise AuditSinkError("Studio audit append failed.")
            self._path.parent.mkdir(parents=True, exist_ok=True)
            previous_event_hash = self._previous_event_hash()
            self._rotate_if_needed()
            event_row = self._build_row(event, previous_event_hash)
            row = json.dumps(
                event_row,
                separators=(",", ":"),
                sort_keys=True,
            )
            with self._path.open("a", encoding="utf-8") as audit_file:
                audit_file.write(f"{row}\n")
        except OSError as exc:
            self._last_error = type(exc).__name__
            raise AuditSinkError("Studio audit append failed.") from exc
        self._last_error = None

    def status(self) -> AuditSinkStatus:
        """Return status for the persistent JSONL audit sink."""

        preflight_error = self._preflight_error()
        last_error = preflight_error or self._last_error
        return AuditSinkStatus(
            configured=True,
            healthy=last_error is None,
            last_error=last_error,
            path_configured=True,
            sink_type="jsonl",
        )

    def export_recent(self, limit: int = 100) -> AuditExport:
        """Export the most recent persisted audit rows without exposing paths.

        Parameters
        ----------
        limit:
            Maximum number of audit events to include. Must be positive.

        Returns
        -------
        AuditExport
            Path-free export payload containing the newest retained rows across
            rotated JSONL files and the active audit log.

        Raises
        ------
        AuditSinkError
            If the sink location is malformed or a stored row is not a JSON
            object with scalar public values.
        """

        if limit < 1:
            raise ValueError("Audit export limit must be positive.")
        preflight_error = self._preflight_error()
        if preflight_error is not None:
            self._last_error = preflight_error
            raise AuditSinkError("Studio audit export failed.")
        rows = self._export_rows()
        truncated = len(rows) > limit
        selected_rows = rows[-limit:]
        return AuditExport(
            configured=True,
            sink_type="jsonl",
            event_count=len(selected_rows),
            truncated=truncated,
            events=tuple(selected_rows),
        )

    def _preflight_error(self) -> str | None:
        if self._path.exists() and self._path.is_dir():
            return "AuditPathIsDirectory"
        if self._path.parent.exists() and not self._path.parent.is_dir():
            return "AuditParentIsNotDirectory"
        return None

    def _build_row(
        self,
        event: AuditEvent,
        previous_event_hash: str | None,
    ) -> dict[str, str | None]:
        row = event.to_json_dict()
        row["previous_event_hash"] = previous_event_hash
        row["event_hash"] = self._event_hash(row)
        return row

    def _rotate_if_needed(self) -> None:
        if self._rotation_bytes is None or self._retained_files == 0:
            return
        if not self._path.exists() or self._path.stat().st_size < self._rotation_bytes:
            return
        oldest_path = self._rotated_path(self._retained_files)
        oldest_path.unlink(missing_ok=True)
        for index in range(self._retained_files - 1, 0, -1):
            source = self._rotated_path(index)
            if source.exists():
                source.replace(self._rotated_path(index + 1))
        self._path.replace(self._rotated_path(1))

    def _rotated_path(self, index: int) -> Path:
        return self._path.with_name(f"{self._path.name}.{index}")

    def _export_paths(self) -> tuple[Path, ...]:
        rotated_paths = tuple(
            self._rotated_path(index)
            for index in range(self._retained_files, 0, -1)
            if self._rotated_path(index).exists()
        )
        return (*rotated_paths, self._path)

    def _export_rows(self) -> list[dict[str, str | None]]:
        rows: list[dict[str, str | None]] = []
        try:
            for path in self._export_paths():
                if not path.exists():
                    continue
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    parsed = json.loads(line)
                    if not isinstance(parsed, dict):
                        self._last_error = "AuditExportInvalidRow"
                        raise AuditSinkError("Studio audit export failed.")
                    rows.append(self._public_export_row(parsed))
        except json.JSONDecodeError as exc:
            self._last_error = "AuditExportInvalidJson"
            raise AuditSinkError("Studio audit export failed.") from exc
        except OSError as exc:
            self._last_error = type(exc).__name__
            raise AuditSinkError("Studio audit export failed.") from exc
        return rows

    def _public_export_row(self, parsed: dict[object, object]) -> dict[str, str | None]:
        row: dict[str, str | None] = {}
        for key, value in parsed.items():
            if not isinstance(key, str) or not (
                isinstance(value, str) or value is None
            ):
                self._last_error = "AuditExportInvalidRow"
                raise AuditSinkError("Studio audit export failed.")
            row[key] = value
        return row

    def _previous_event_hash(self) -> str | None:
        if not self._path.exists():
            return None
        for line in reversed(self._path.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            previous_row = json.loads(line)
            previous_hash = previous_row.get("event_hash")
            return previous_hash if isinstance(previous_hash, str) else None
        return None

    @staticmethod
    def _event_hash(row: dict[str, str | None]) -> str:
        unsigned_row = dict(row)
        unsigned_row.pop("event_hash", None)
        canonical_row = json.dumps(
            unsigned_row,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(canonical_row).hexdigest()


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
            (
                "GET",
                "/api/studio/audit/status",
                RouteVisibility.PUBLIC,
                "studio.audit.status.read",
            ),
            (
                "GET",
                "/api/studio/audit/export",
                RouteVisibility.ADMIN,
                "studio.audit.export",
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
