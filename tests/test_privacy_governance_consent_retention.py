# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Privacy consent and retention policy tests

"""Validate consent boundaries and bounded retention windows."""

from __future__ import annotations

import pytest

from sc_neurocore.privacy.governance import ConsentBoundary, GovernanceContract, RetentionPolicy
from tests.privacy_governance_support import minimal_contract_payload


def test_invalid_retention_duration_is_rejected() -> None:
    payload = minimal_contract_payload()
    payload["retention_policy"]["raw_stream_days"] = 0

    with pytest.raises(ValueError, match="raw_stream_days must be positive"):
        GovernanceContract.from_dict(payload)


def test_consent_boundary_rejects_invalid_legal_basis_and_field_types() -> None:
    base = minimal_contract_payload()["consent_boundary"]

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
