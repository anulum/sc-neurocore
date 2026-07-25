# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Privacy governance provenance tests

"""Validate provenance shape, identifiers, hashes, and telemetry numeric fields."""

from __future__ import annotations

import pytest

from sc_neurocore.privacy.governance import GovernanceContract, TelemetryPolicy
from tests.privacy_governance_support import minimal_contract_payload


def test_provenance_hash_and_uri_required() -> None:
    payload = minimal_contract_payload()
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


def test_contract_rejects_non_list_provenance_section() -> None:
    payload = minimal_contract_payload()
    payload["provenance"] = {"artifact_type": "model"}

    with pytest.raises(ValueError, match="provenance must be a list"):
        GovernanceContract.from_dict(payload)


def test_telemetry_and_provenance_validate_numeric_and_hash_contracts() -> None:
    with pytest.raises(ValueError, match="sampling_interval_ms must be an int"):
        TelemetryPolicy(enabled=True, sink="local", sampling_interval_ms=1.5)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sampling_interval_ms must be positive"):
        TelemetryPolicy(enabled=True, sink="local", sampling_interval_ms=0)

    provenance = minimal_contract_payload()["provenance"][0]
    with pytest.raises(ValueError, match="artifact_hash"):
        GovernanceContract.from_dict(
            {
                **minimal_contract_payload(),
                "provenance": [{**provenance, "artifact_hash": ""}],
            }
        )
    with pytest.raises(ValueError, match="hash_algorithm"):
        GovernanceContract.from_dict(
            {
                **minimal_contract_payload(),
                "provenance": [{**provenance, "hash_algorithm": ""}],
            }
        )
