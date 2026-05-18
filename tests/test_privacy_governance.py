# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Privacy governance contract tests

from __future__ import annotations

from typing import Any, Dict

import pytest

from sc_neurocore.privacy import GovernanceContract


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
