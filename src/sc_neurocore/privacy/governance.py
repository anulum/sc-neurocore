# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Privacy governance contracts for neural/BCI workflows.

The module defines a deterministic, dependency-free contract model used to
record consent boundaries, retention windows, redaction policy, telemetry
configuration, model/data provenance, integrator responsibilities, and
privacy-feature activation/audit requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple


_ROOT_REQUIRED_FIELDS = (
    "consent_boundary",
    "retention_policy",
    "redaction_policy",
    "telemetry",
    "provenance",
    "integrator",
    "features",
)


_ALLOWED_CONSENT_BASES = (
    "informed_consent",
    "research_authorization",
    "emergency_use",
    "clinical_care",
)


_FEATURE_AUDIT_FLAG = {
    "enable_federated_learning": "federated_learning",
    "enable_differential_privacy": "differential_privacy",
    "enable_telemetry_logging": "telemetry",
}


def _expect_mapping(value: Any, *, field: str) -> Dict[str, Any]:
    """Return ``value`` as dict or raise a typed ValueError."""
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _require_fields(payload: Dict[str, Any], section: str, required: tuple[str, ...]) -> None:
    """Raise when required fields are missing from a manifest section."""
    for field in required:
        if field not in payload:
            raise ValueError(f"Missing required field: {section}.{field}")


def _ensure_positive_days(value: int, name: str) -> int:
    """Validate a duration field."""
    if not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _ensure_non_empty_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _ensure_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _to_str_tuple(values: Any, name: str) -> Tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        raise ValueError(f"{name} must be a sequence of strings")
    if not isinstance(values, Iterable):
        raise ValueError(f"{name} must be a sequence of strings")
    normalized = []
    for item in values:
        normalized.append(_ensure_non_empty_str(item, f"{name} entry"))
    return tuple(normalized)


@dataclass(frozen=True)
class ConsentBoundary:
    """Participant-level legal basis and telemetry permissions."""

    participant_id: str
    consent_basis: str
    allow_telemetry: bool
    allowed_purposes: Tuple[str, ...]
    consent_token: str
    issued_at_unix: int

    def __post_init__(self) -> None:
        """Validate consent identity, legal basis, telemetry flag, and token fields."""
        object.__setattr__(
            self, "participant_id", _ensure_non_empty_str(self.participant_id, "participant_id")
        )
        object.__setattr__(
            self, "consent_basis", _ensure_non_empty_str(self.consent_basis, "consent_basis")
        )
        if self.consent_basis not in _ALLOWED_CONSENT_BASES:
            allowed = ", ".join(_ALLOWED_CONSENT_BASES)
            raise ValueError(f"consent_basis must be one of: {allowed}")
        object.__setattr__(
            self,
            "allow_telemetry",
            _ensure_bool(self.allow_telemetry, "allow_telemetry"),
        )
        if not isinstance(self.issued_at_unix, int) or self.issued_at_unix <= 0:
            raise ValueError("issued_at_unix must be a positive int")
        object.__setattr__(
            self, "allowed_purposes", _to_str_tuple(self.allowed_purposes, "allowed_purposes")
        )
        object.__setattr__(
            self, "consent_token", _ensure_non_empty_str(self.consent_token, "consent_token")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return this consent boundary as a deterministic mapping."""
        return {
            "participant_id": self.participant_id,
            "consent_basis": self.consent_basis,
            "allow_telemetry": self.allow_telemetry,
            "allowed_purposes": list(self.allowed_purposes),
            "consent_token": self.consent_token,
            "issued_at_unix": self.issued_at_unix,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConsentBoundary":
        """Build a consent boundary from a manifest section."""
        payload = _expect_mapping(data, field="consent_boundary")
        _require_fields(
            payload,
            "consent_boundary",
            (
                "participant_id",
                "consent_basis",
                "allow_telemetry",
                "allowed_purposes",
                "consent_token",
                "issued_at_unix",
            ),
        )
        return cls(
            participant_id=payload["participant_id"],
            consent_basis=payload["consent_basis"],
            allow_telemetry=payload["allow_telemetry"],
            allowed_purposes=payload.get("allowed_purposes", ()),
            consent_token=payload["consent_token"],
            issued_at_unix=payload["issued_at_unix"],
        )


@dataclass(frozen=True)
class RetentionPolicy:
    """Retention windows for neural telemetry and artefacts."""

    raw_stream_days: int
    model_artifacts_days: int
    audit_log_days: int
    max_days: int

    def __post_init__(self) -> None:
        """Validate retention windows and enforce the maximum retention horizon."""
        object.__setattr__(
            self,
            "raw_stream_days",
            _ensure_positive_days(self.raw_stream_days, "raw_stream_days"),
        )
        object.__setattr__(
            self,
            "model_artifacts_days",
            _ensure_positive_days(self.model_artifacts_days, "model_artifacts_days"),
        )
        object.__setattr__(
            self,
            "audit_log_days",
            _ensure_positive_days(self.audit_log_days, "audit_log_days"),
        )
        object.__setattr__(self, "max_days", _ensure_positive_days(self.max_days, "max_days"))

        if (
            self.raw_stream_days > self.max_days
            or self.model_artifacts_days > self.max_days
            or self.audit_log_days > self.max_days
        ):
            raise ValueError("all retention windows must be <= max_days")

    def to_dict(self) -> Dict[str, Any]:
        """Return this retention policy as a deterministic mapping."""
        return {
            "raw_stream_days": self.raw_stream_days,
            "model_artifacts_days": self.model_artifacts_days,
            "audit_log_days": self.audit_log_days,
            "max_days": self.max_days,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetentionPolicy":
        """Build a retention policy from a manifest section."""
        payload = _expect_mapping(data, field="retention_policy")
        _require_fields(
            payload,
            "retention_policy",
            (
                "raw_stream_days",
                "model_artifacts_days",
                "audit_log_days",
                "max_days",
            ),
        )
        return cls(
            raw_stream_days=payload["raw_stream_days"],
            model_artifacts_days=payload["model_artifacts_days"],
            audit_log_days=payload["audit_log_days"],
            max_days=payload["max_days"],
        )


@dataclass(frozen=True)
class RedactionPolicy:
    """Field-level redaction policy for protected telemetry and logs."""

    enabled: bool
    fields: Tuple[str, ...]
    replacement: str

    def __post_init__(self) -> None:
        """Validate redaction activation, field list, and replacement marker."""
        object.__setattr__(self, "enabled", _ensure_bool(self.enabled, "enabled"))
        object.__setattr__(self, "fields", _to_str_tuple(self.fields, "fields"))
        if self.enabled and not self.fields:
            raise ValueError("redaction enabled requires at least one field")
        if self.replacement is None:
            raise ValueError("replacement must not be None")

    def to_dict(self) -> Dict[str, Any]:
        """Return this redaction policy as a deterministic mapping."""
        return {
            "enabled": self.enabled,
            "fields": list(self.fields),
            "replacement": self.replacement,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RedactionPolicy":
        """Build a redaction policy from a manifest section."""
        payload = _expect_mapping(data, field="redaction_policy")
        _require_fields(
            payload,
            "redaction_policy",
            ("enabled", "fields", "replacement"),
        )
        return cls(
            enabled=payload["enabled"],
            fields=payload.get("fields", ()),
            replacement=payload.get("replacement", ""),
        )


@dataclass(frozen=True)
class TelemetryPolicy:
    """Telemetry sink and sampling policy."""

    enabled: bool
    sink: str
    sampling_interval_ms: int

    def __post_init__(self) -> None:
        """Validate telemetry activation, sink name, and sampling interval."""
        object.__setattr__(self, "enabled", _ensure_bool(self.enabled, "enabled"))
        object.__setattr__(self, "sink", _ensure_non_empty_str(self.sink, "sink"))
        if self.enabled and self.sampling_interval_ms is None:
            raise ValueError("sampling_interval_ms must be set when telemetry is enabled")
        if not isinstance(self.sampling_interval_ms, int):
            raise ValueError("sampling_interval_ms must be an int")
        if self.sampling_interval_ms <= 0:
            raise ValueError("sampling_interval_ms must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Return this telemetry policy as a deterministic mapping."""
        return {
            "enabled": self.enabled,
            "sink": self.sink,
            "sampling_interval_ms": self.sampling_interval_ms,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TelemetryPolicy":
        """Build a telemetry policy from a manifest section."""
        payload = _expect_mapping(data, field="telemetry")
        _require_fields(payload, "telemetry", ("enabled", "sink", "sampling_interval_ms"))
        return cls(
            enabled=payload["enabled"],
            sink=payload["sink"],
            sampling_interval_ms=payload["sampling_interval_ms"],
        )


@dataclass(frozen=True)
class ProvenanceRecord:
    """Cryptographic provenance record for model and dataset artefacts."""

    artifact_type: str
    artifact_uri: str
    hash_algorithm: str
    artifact_hash: str
    source_system: str

    def __post_init__(self) -> None:
        """Validate provenance artefact identity, hash, and source-system fields."""
        object.__setattr__(
            self, "artifact_type", _ensure_non_empty_str(self.artifact_type, "artifact_type")
        )
        object.__setattr__(
            self, "artifact_uri", _ensure_non_empty_str(self.artifact_uri, "artifact_uri")
        )
        object.__setattr__(
            self, "hash_algorithm", _ensure_non_empty_str(self.hash_algorithm, "hash_algorithm")
        )
        object.__setattr__(
            self, "artifact_hash", _ensure_non_empty_str(self.artifact_hash, "artifact_hash")
        )
        object.__setattr__(
            self,
            "source_system",
            _ensure_non_empty_str(self.source_system, "source_system"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return this provenance record as a deterministic mapping."""
        return {
            "artifact_type": self.artifact_type,
            "artifact_uri": self.artifact_uri,
            "hash_algorithm": self.hash_algorithm,
            "artifact_hash": self.artifact_hash,
            "source_system": self.source_system,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceRecord":
        """Build a provenance record from one manifest list item."""
        payload = _expect_mapping(data, field="provenance entry")
        _require_fields(
            payload,
            "provenance entry",
            (
                "artifact_type",
                "artifact_uri",
                "hash_algorithm",
                "artifact_hash",
                "source_system",
            ),
        )
        if not payload.get("artifact_uri", ""):
            raise ValueError("Provenance entry requires artifact_uri")
        if not payload.get("artifact_hash", ""):
            raise ValueError("Provenance entry requires artifact_hash")
        if not payload.get("hash_algorithm", ""):
            raise ValueError("Provenance entry requires hash_algorithm")
        return cls(
            artifact_type=payload["artifact_type"],
            artifact_uri=payload["artifact_uri"],
            hash_algorithm=payload["hash_algorithm"],
            artifact_hash=payload["artifact_hash"],
            source_system=payload["source_system"],
        )


@dataclass(frozen=True)
class IntegratorResponsibility:
    """Operational responsibilities for an integrator in a deployment pipeline."""

    name: str
    contact: str
    responsibilities: Tuple[str, ...]
    release_approval_required: bool

    def __post_init__(self) -> None:
        """Validate integrator contact details and approval responsibility fields."""
        object.__setattr__(self, "name", _ensure_non_empty_str(self.name, "name"))
        object.__setattr__(self, "contact", _ensure_non_empty_str(self.contact, "contact"))
        object.__setattr__(
            self, "responsibilities", _to_str_tuple(self.responsibilities, "responsibilities")
        )
        object.__setattr__(
            self,
            "release_approval_required",
            _ensure_bool(self.release_approval_required, "release_approval_required"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return this integrator responsibility record as a deterministic mapping."""
        return {
            "name": self.name,
            "contact": self.contact,
            "responsibilities": list(self.responsibilities),
            "release_approval_required": self.release_approval_required,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntegratorResponsibility":
        """Build an integrator responsibility record from a manifest section."""
        payload = _expect_mapping(data, field="integrator")
        _require_fields(
            payload,
            "integrator",
            ("name", "contact", "responsibilities", "release_approval_required"),
        )
        return cls(
            name=payload["name"],
            contact=payload["contact"],
            responsibilities=payload.get("responsibilities", ()),
            release_approval_required=payload["release_approval_required"],
        )


@dataclass(frozen=True)
class PrivacyFeatureFlags:
    """Feature activation and audit flags for a governed workflow."""

    enable_differential_privacy: bool
    enable_federated_learning: bool
    enable_telemetry_logging: bool
    audit_enabled: bool
    audit_flags: Tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate feature toggles and the audit-flag activation contract."""
        object.__setattr__(
            self,
            "enable_differential_privacy",
            _ensure_bool(self.enable_differential_privacy, "enable_differential_privacy"),
        )
        object.__setattr__(
            self,
            "enable_federated_learning",
            _ensure_bool(self.enable_federated_learning, "enable_federated_learning"),
        )
        object.__setattr__(
            self,
            "enable_telemetry_logging",
            _ensure_bool(self.enable_telemetry_logging, "enable_telemetry_logging"),
        )
        object.__setattr__(self, "audit_enabled", _ensure_bool(self.audit_enabled, "audit_enabled"))
        object.__setattr__(self, "audit_flags", _to_str_tuple(self.audit_flags, "audit_flags"))

        if self.audit_enabled and not self.audit_flags:
            raise ValueError("audit_enabled requires audit_flags")

    def to_dict(self) -> Dict[str, Any]:
        """Return this privacy feature flag set as a deterministic mapping."""
        return {
            "enable_differential_privacy": self.enable_differential_privacy,
            "enable_federated_learning": self.enable_federated_learning,
            "enable_telemetry_logging": self.enable_telemetry_logging,
            "audit_enabled": self.audit_enabled,
            "audit_flags": list(self.audit_flags),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PrivacyFeatureFlags":
        """Build privacy feature flags from a manifest section."""
        payload = _expect_mapping(data, field="features")
        _require_fields(
            payload,
            "features",
            (
                "enable_differential_privacy",
                "enable_federated_learning",
                "enable_telemetry_logging",
                "audit_enabled",
                "audit_flags",
            ),
        )
        return cls(
            enable_differential_privacy=payload["enable_differential_privacy"],
            enable_federated_learning=payload["enable_federated_learning"],
            enable_telemetry_logging=payload["enable_telemetry_logging"],
            audit_enabled=payload["audit_enabled"],
            audit_flags=payload.get("audit_flags", ()),
        )


@dataclass(frozen=True)
class GovernanceContract:
    """Full privacy governance contract for BCI/neural workflows."""

    consent_boundary: ConsentBoundary
    retention_policy: RetentionPolicy
    redaction_policy: RedactionPolicy
    telemetry: TelemetryPolicy
    provenance: Tuple[ProvenanceRecord, ...]
    integrator: IntegratorResponsibility
    features: PrivacyFeatureFlags
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        """Enforce cross-section privacy governance invariants."""
        if self.features.enable_telemetry_logging:
            if not self.telemetry.enabled:
                raise ValueError("telemetry_logging requires telemetry enabled")
            if not self.redaction_policy.enabled:
                raise ValueError("telemetry_logging requires redaction")
            if not self.consent_boundary.allow_telemetry:
                raise ValueError("telemetry_logging requires telemetry consent")

        if self.features.enable_differential_privacy and (
            "differential_privacy" not in self.features.audit_flags
        ):
            raise ValueError("differential_privacy requires audit flag 'differential_privacy'")
        if self.features.enable_federated_learning and (
            "federated_learning" not in self.features.audit_flags
        ):
            raise ValueError("federated_learning requires audit flag 'federated_learning'")

        if self.features.enable_telemetry_logging and "telemetry" not in self.features.audit_flags:
            raise ValueError("telemetry_logging requires audit flag 'telemetry'")

    @property
    def audit_required_features(self) -> Tuple[str, ...]:
        """Return sorted feature keys that require audit flags."""
        return tuple(sorted(_FEATURE_AUDIT_FLAG.keys()))

    def active_features(self) -> Tuple[str, ...]:
        """Return names of enabled privacy features in deterministic order."""
        enabled = []
        if self.features.enable_differential_privacy:
            enabled.append("differential_privacy")
        if self.features.enable_federated_learning:
            enabled.append("federated_learning")
        if self.features.enable_telemetry_logging:
            enabled.append("telemetry_logging")
        return tuple(sorted(enabled))

    def to_dict(self) -> Dict[str, Any]:
        """Return a deterministic JSON-serialisable representation."""
        return {
            "consent_boundary": self.consent_boundary.to_dict(),
            "retention_policy": self.retention_policy.to_dict(),
            "redaction_policy": self.redaction_policy.to_dict(),
            "telemetry": self.telemetry.to_dict(),
            "provenance": [record.to_dict() for record in self.provenance],
            "integrator": self.integrator.to_dict(),
            "features": self.features.to_dict(),
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GovernanceContract":
        """Build a full governance contract from a manifest mapping."""
        payload = _expect_mapping(data, field="governance_contract")

        for required in _ROOT_REQUIRED_FIELDS:
            if required not in payload:
                raise ValueError(f"Missing required field: {required}")

        provenance = payload["provenance"]
        if not isinstance(provenance, list):
            raise ValueError("provenance must be a list")

        return cls(
            consent_boundary=ConsentBoundary.from_dict(payload["consent_boundary"]),
            retention_policy=RetentionPolicy.from_dict(payload["retention_policy"]),
            redaction_policy=RedactionPolicy.from_dict(payload["redaction_policy"]),
            telemetry=TelemetryPolicy.from_dict(payload["telemetry"]),
            provenance=tuple(ProvenanceRecord.from_dict(item) for item in provenance),
            integrator=IntegratorResponsibility.from_dict(payload["integrator"]),
            features=PrivacyFeatureFlags.from_dict(payload["features"]),
            schema_version=payload.get("schema_version", "1.0"),
        )
