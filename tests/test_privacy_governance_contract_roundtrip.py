# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Privacy governance contract construction tests

"""Validate governance-contract construction and deterministic round trips."""

from __future__ import annotations

import pytest

from sc_neurocore.privacy.governance import ConsentBoundary, GovernanceContract
from tests.privacy_governance_support import minimal_contract_payload, valid_contract


def test_contract_accepts_valid_payload() -> None:
    contract = valid_contract()

    assert contract.consent_boundary.participant_id == "subject-001"
    assert contract.features.enable_telemetry_logging is True
    assert len(contract.provenance) == 1


def test_missing_required_fields_are_rejected() -> None:
    payload = minimal_contract_payload()
    payload.pop("consent_boundary")

    with pytest.raises(ValueError, match="Missing required field: consent_boundary"):
        GovernanceContract.from_dict(payload)


def test_class_helpers_are_deterministic() -> None:
    contract = valid_contract()
    as_dict = contract.to_dict()

    assert as_dict == contract.to_dict()
    assert set(contract.active_features()) == {"telemetry_logging"}


def test_constructor_roundtrip_for_all_components() -> None:
    contract = GovernanceContract.from_dict(minimal_contract_payload())

    rebuilt = GovernanceContract.from_dict(contract.to_dict())
    assert rebuilt.to_dict() == contract.to_dict()


def test_contract_sections_reject_non_mappings_and_missing_fields() -> None:
    with pytest.raises(ValueError, match="consent_boundary must be a mapping"):
        ConsentBoundary.from_dict([])  # type: ignore[arg-type]
    payload = dict(minimal_contract_payload()["consent_boundary"])
    payload.pop("participant_id")
    with pytest.raises(ValueError, match="consent_boundary.participant_id"):
        ConsentBoundary.from_dict(payload)
