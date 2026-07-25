# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Privacy feature audit contract tests

"""Validate audit requirements for privacy features and integrators."""

from __future__ import annotations

import pytest

from sc_neurocore.privacy.governance import (
    GovernanceContract,
    IntegratorResponsibility,
    PrivacyFeatureFlags,
)
from tests.privacy_governance_support import minimal_contract_payload


def test_feature_audit_flags_required_for_sensitive_features() -> None:
    payload = minimal_contract_payload()
    payload["features"]["enable_federated_learning"] = True
    payload["features"]["audit_flags"] = ["telemetry"]

    with pytest.raises(
        ValueError, match="federated_learning requires audit flag 'federated_learning'"
    ):
        GovernanceContract.from_dict(payload)


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
                **minimal_contract_payload(),
                "features": {
                    **minimal_contract_payload()["features"],
                    "enable_differential_privacy": True,
                },
            }
        )
    with pytest.raises(ValueError, match="telemetry_logging requires audit flag"):
        GovernanceContract.from_dict(
            {
                **minimal_contract_payload(),
                "features": {
                    **minimal_contract_payload()["features"],
                    "audit_flags": ["non_matching_flag"],
                },
            }
        )


def test_active_features_and_audit_required_features_are_deterministic() -> None:
    payload = minimal_contract_payload()
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
