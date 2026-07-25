# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Privacy governance contract fixtures

"""Shared valid inputs for privacy-governance contract tests."""

from __future__ import annotations

from typing import Any

from sc_neurocore.privacy.governance import GovernanceContract


def minimal_contract_payload() -> dict[str, Any]:
    """Return a valid privacy-governance payload for mutation by tests."""
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


def valid_contract() -> GovernanceContract:
    """Build the canonical valid privacy-governance contract."""
    return GovernanceContract.from_dict(minimal_contract_payload())
