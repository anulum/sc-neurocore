# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Privacy telemetry and redaction policy tests

"""Validate telemetry consent, transport settings, and redaction requirements."""

from __future__ import annotations

import pytest

from sc_neurocore.privacy.governance import (
    GovernanceContract,
    RedactionPolicy,
    TelemetryPolicy,
)
from tests.privacy_governance_support import minimal_contract_payload


def test_telemetry_logging_requires_telemetry_enabled() -> None:
    payload = minimal_contract_payload()
    payload["telemetry"]["enabled"] = False

    with pytest.raises(ValueError, match="requires telemetry enabled"):
        GovernanceContract.from_dict(payload)


def test_telemetry_requires_redaction_and_consent() -> None:
    payload = minimal_contract_payload()
    payload["redaction_policy"]["enabled"] = False

    with pytest.raises(ValueError, match="telemetry_logging requires redaction"):
        GovernanceContract.from_dict(payload)

    payload = minimal_contract_payload()
    payload["consent_boundary"]["allow_telemetry"] = False

    with pytest.raises(ValueError, match="telemetry_logging requires telemetry consent"):
        GovernanceContract.from_dict(payload)


def test_redaction_and_telemetry_policies_fail_closed_on_malformed_contracts() -> None:
    with pytest.raises(ValueError, match="redaction enabled"):
        RedactionPolicy(enabled=True, fields=(), replacement="***")
    with pytest.raises(ValueError, match="replacement"):
        RedactionPolicy(enabled=False, fields=(), replacement=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sampling_interval_ms"):
        TelemetryPolicy(enabled=True, sink="local", sampling_interval_ms=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sink"):
        TelemetryPolicy(enabled=False, sink="", sampling_interval_ms=100)


def test_sequence_fields_accept_none_for_disabled_policy_and_reject_non_iterables() -> None:
    assert RedactionPolicy(enabled=False, fields=None, replacement="").fields == ()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fields"):
        RedactionPolicy(enabled=False, fields=7, replacement="")  # type: ignore[arg-type]
