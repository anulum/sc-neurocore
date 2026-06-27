# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Privacy governance contract tests

from __future__ import annotations

from typing import Any, Dict

import pytest

from sc_neurocore.privacy.governance import (
    ConsentBoundary,
    GovernanceContract,
    IntegratorResponsibility,
    PrivacyFeatureFlags,
    RedactionPolicy,
    RetentionPolicy,
    TelemetryPolicy,
)


def _minimal_contract_payload() -> Dict[str, Any]:
    return {
        "consent_boundary": {
            "participant_id": "subject-001",
            "consent_basis": "informed_consent",
            "allow_telemetry": True,
            "allowed_purposes": ["training", "telemetry", "research"],
            "consent_token": "token-001",
            "issued_at_unix": 1735689600,
        },
        "retention_policy": {
            "raw_stream_days": 30,
            "model_artifacts_days": 365,
            "audit_log_days": 90,
            "max_days": 3650,
        },
        "redaction_policy": {
            "enabled": True,
            "fields": ["patient_id", "channel_id"],
            "replacement": "***",
        },
        "telemetry": {
            "enabled": True,
            "sink": "local-bundle",
            "sampling_interval_ms": 250,
        },
        "provenance": [
            {
                "artifact_type": "model",
                "artifact_uri": "file://models/model-v1.scn",
                "hash_algorithm": "sha256",
                "artifact_hash": "0123456789abcdef" * 4,
                "source_system": "git",
            },
        ],
        "integrator": {
            "name": "hospital-integrator",
            "contact": "security@hospital.example",
            "responsibilities": [
                "remove_patient_identifiers",
                "verify_retention_compliance",
            ],
            "release_approval_required": True,
        },
        "features": {
            "enable_differential_privacy": False,
            "enable_federated_learning": False,
            "enable_telemetry_logging": True,
            "audit_enabled": True,
            "audit_flags": ["telemetry"],
        },
    }


def _valid_contract() -> GovernanceContract:
    payload = _minimal_contract_payload()
    return GovernanceContract.from_dict(payload)


def test_contract_accepts_valid_payload() -> None:
    contract = _valid_contract()

    assert contract.consent_boundary.participant_id == "subject-001"
    assert contract.features.enable_telemetry_logging is True
    assert len(contract.provenance) == 1


def test_missing_required_fields_are_rejected() -> None:
    payload = _minimal_contract_payload()
    payload.pop("consent_boundary")

    with pytest.raises(ValueError, match="Missing required field: consent_boundary"):
        GovernanceContract.from_dict(payload)


def test_invalid_retention_duration_is_rejected() -> None:
    payload = _minimal_contract_payload()
    payload["retention_policy"]["raw_stream_days"] = 0

    with pytest.raises(ValueError, match="raw_stream_days must be positive"):
        GovernanceContract.from_dict(payload)


def test_telemetry_logging_requires_telemetry_enabled() -> None:
    payload = _minimal_contract_payload()
    payload["telemetry"]["enabled"] = False

    with pytest.raises(ValueError, match="requires telemetry enabled"):
        GovernanceContract.from_dict(payload)


def test_telemetry_requires_redaction_and_consent() -> None:
    payload = _minimal_contract_payload()
    payload["redaction_policy"]["enabled"] = False

    with pytest.raises(ValueError, match="telemetry_logging requires redaction"):
        GovernanceContract.from_dict(payload)

    payload = _minimal_contract_payload()
    payload["consent_boundary"]["allow_telemetry"] = False

    with pytest.raises(ValueError, match="telemetry_logging requires telemetry consent"):
        GovernanceContract.from_dict(payload)


def test_provenance_hash_and_uri_required() -> None:
    payload = _minimal_contract_payload()
    payload["provenance"] = [
        {
            "artifact_type": "dataset",
            "artifact_uri": "",
            "hash_algorithm": "sha256",
            "artifact_hash": "",
            "source_system": "git",
        }
    ]

    with pytest.raises(ValueError, match="Provenance entry requires .*artifact_uri"):
        GovernanceContract.from_dict(payload)


def test_feature_audit_flags_required_for_sensitive_features() -> None:
    payload = _minimal_contract_payload()
    payload["features"]["enable_federated_learning"] = True
    payload["features"]["audit_flags"] = ["telemetry"]

    with pytest.raises(
        ValueError, match="federated_learning requires audit flag 'federated_learning'"
    ):
        GovernanceContract.from_dict(payload)


def test_class_helpers_are_deterministic() -> None:
    contract = _valid_contract()
    as_dict = contract.to_dict()

    assert as_dict == contract.to_dict()
    assert set(contract.active_features()) == {"telemetry_logging"}


def test_constructor_roundtrip_for_all_components() -> None:
    payload = _minimal_contract_payload()
    contract = GovernanceContract.from_dict(payload)

    rebuilt = GovernanceContract.from_dict(contract.to_dict())
    assert rebuilt.to_dict() == contract.to_dict()


def test_consent_boundary_rejects_invalid_legal_basis_and_field_types() -> None:
    base = _minimal_contract_payload()["consent_boundary"]

    with pytest.raises(ValueError, match="consent_basis"):
        ConsentBoundary(**{**base, "consent_basis": "verbal_only"})
    with pytest.raises(ValueError, match="allow_telemetry"):
        ConsentBoundary(**{**base, "allow_telemetry": "yes"})
    with pytest.raises(ValueError, match="allowed_purposes"):
        ConsentBoundary(**{**base, "allowed_purposes": "training"})
    with pytest.raises(ValueError, match="issued_at_unix"):
        ConsentBoundary(**{**base, "issued_at_unix": 0})


def test_retention_policy_rejects_non_integer_and_over_max_windows() -> None:
    with pytest.raises(ValueError, match="raw_stream_days must be an int"):
        RetentionPolicy(
            raw_stream_days=1.5,  # type: ignore[arg-type]
            model_artifacts_days=10,
            audit_log_days=10,
            max_days=10,
        )
    with pytest.raises(ValueError, match="retention windows"):
        RetentionPolicy(
            raw_stream_days=11,
            model_artifacts_days=10,
            audit_log_days=10,
            max_days=10,
        )


def test_redaction_and_telemetry_policies_fail_closed_on_malformed_contracts() -> None:
    with pytest.raises(ValueError, match="redaction enabled"):
        RedactionPolicy(enabled=True, fields=(), replacement="***")
    with pytest.raises(ValueError, match="replacement"):
        RedactionPolicy(enabled=False, fields=(), replacement=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sampling_interval_ms"):
        TelemetryPolicy(enabled=True, sink="local", sampling_interval_ms=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sink"):
        TelemetryPolicy(enabled=False, sink="", sampling_interval_ms=100)


def test_integrator_and_feature_flags_reject_missing_audit_contracts() -> None:
    with pytest.raises(ValueError, match="release_approval_required"):
        IntegratorResponsibility(
            name="integrator",
            contact="ops@example.test",
            responsibilities=("redact",),
            release_approval_required="yes",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="audit_enabled requires audit_flags"):
        PrivacyFeatureFlags(
            enable_differential_privacy=False,
            enable_federated_learning=False,
            enable_telemetry_logging=False,
            audit_enabled=True,
            audit_flags=(),
        )
    with pytest.raises(ValueError, match="differential_privacy"):
        GovernanceContract.from_dict(
            {
                **_minimal_contract_payload(),
                "features": {
                    **_minimal_contract_payload()["features"],
                    "enable_differential_privacy": True,
                },
            }
        )
    with pytest.raises(ValueError, match="telemetry_logging requires audit flag"):
        GovernanceContract.from_dict(
            {
                **_minimal_contract_payload(),
                "features": {
                    **_minimal_contract_payload()["features"],
                    "audit_flags": ["non_matching_flag"],
                },
            }
        )


def test_contract_rejects_non_list_provenance_section() -> None:
    payload = _minimal_contract_payload()
    payload["provenance"] = {"artifact_type": "model"}

    with pytest.raises(ValueError, match="provenance must be a list"):
        GovernanceContract.from_dict(payload)


def test_contract_sections_reject_non_mappings_and_missing_fields() -> None:
    with pytest.raises(ValueError, match="consent_boundary must be a mapping"):
        ConsentBoundary.from_dict([])  # type: ignore[arg-type]
    payload = dict(_minimal_contract_payload()["consent_boundary"])
    payload.pop("participant_id")
    with pytest.raises(ValueError, match="consent_boundary.participant_id"):
        ConsentBoundary.from_dict(payload)


def test_sequence_fields_accept_none_for_disabled_policy_and_reject_non_iterables() -> None:
    assert RedactionPolicy(enabled=False, fields=None, replacement="").fields == ()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fields"):
        RedactionPolicy(enabled=False, fields=7, replacement="")  # type: ignore[arg-type]


def test_telemetry_and_provenance_validate_numeric_and_hash_contracts() -> None:
    with pytest.raises(ValueError, match="sampling_interval_ms must be an int"):
        TelemetryPolicy(enabled=True, sink="local", sampling_interval_ms=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sampling_interval_ms must be positive"):
        TelemetryPolicy(enabled=True, sink="local", sampling_interval_ms=0)

    payload = _minimal_contract_payload()["provenance"][0]
    with pytest.raises(ValueError, match="artifact_hash"):
        GovernanceContract.from_dict(
            {**_minimal_contract_payload(), "provenance": [{**payload, "artifact_hash": ""}]}
        )
    with pytest.raises(ValueError, match="hash_algorithm"):
        GovernanceContract.from_dict(
            {**_minimal_contract_payload(), "provenance": [{**payload, "hash_algorithm": ""}]}
        )


def test_active_features_and_audit_required_features_are_deterministic() -> None:
    payload = _minimal_contract_payload()
    payload["features"] = {
        **payload["features"],
        "enable_differential_privacy": True,
        "enable_federated_learning": True,
        "audit_flags": ["telemetry", "differential_privacy", "federated_learning"],
    }

    contract = GovernanceContract.from_dict(payload)

    assert contract.active_features() == (
        "differential_privacy",
        "federated_learning",
        "telemetry_logging",
    )
    assert contract.audit_required_features == (
        "enable_differential_privacy",
        "enable_federated_learning",
        "enable_telemetry_logging",
    )
