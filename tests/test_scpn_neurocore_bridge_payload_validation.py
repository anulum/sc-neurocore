# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN bridge payload-validation contracts

from __future__ import annotations


import pytest

from scpn_neurocore.bridge import load_power_grid
from tests.scpn_neurocore_bridge_support import refresh_artifact_hash, validate_qpu_payload


def test_validate_qpu_artifact_payload_accepts_round_trip_artifact() -> None:
    payload = load_power_grid(n=4, source_mode="fixture").to_qpu_artifact_dict()

    validate_qpu_payload(payload)


def test_validate_qpu_artifact_payload_rejects_tampered_artifact_hash() -> None:
    payload = load_power_grid(n=4, source_mode="fixture").to_qpu_artifact_dict()
    payload["source_name"] = "tampered-grid"

    with pytest.raises(ValueError, match="artifact_sha256"):
        validate_qpu_payload(payload)


def test_validate_qpu_artifact_payload_rejects_malformed_arrays_and_hashes() -> None:
    payload = load_power_grid(n=4, source_mode="fixture").to_qpu_artifact_dict()
    bad_cases = [
        ("K_nm", {"K_nm": [[0.0, 0.5], [0.5, 0.0], [0.0, 0.0]]}),
        ("omega", {"omega": [1.0, 1.0, 1.0]}),
        ("theta0", {"theta0": [0.0, 0.0, 0.0]}),
        ("layer_assignments", {"layer_assignments": [0, 1, 1, 3]}),
        ("K_nm_sha256", {"hashes": {**payload["hashes"], "K_nm_sha256": "0" * 64}}),
        ("theta0_sha256", {"hashes": {**payload["hashes"], "theta0_sha256": "f" * 64}}),
    ]

    for match, overrides in bad_cases:
        bad_payload = load_power_grid(n=4, source_mode="fixture").to_qpu_artifact_dict()
        bad_payload.update(overrides)
        refresh_artifact_hash(bad_payload)

        with pytest.raises(ValueError, match=match):
            validate_qpu_payload(bad_payload)


def test_validate_qpu_artifact_payload_rejects_malformed_metadata_and_source() -> None:
    payload = load_power_grid(n=4, source_mode="fixture").to_qpu_artifact_dict()
    bad_cases = [
        ("schema_version", {"schema_version": "old"}),
        ("source_mode", {"source_mode": "unknown"}),
        ("source_name", {"source_name": ""}),
        ("metadata", {"metadata": ["not", "mapping"]}),
        ("strict finite JSON", {"metadata": {"bad": float("nan")}}),
        ("artifact_sha256", {"artifact_sha256": "not-a-sha256"}),
    ]

    for match, overrides in bad_cases:
        bad_payload = load_power_grid(n=4, source_mode="fixture").to_qpu_artifact_dict()
        bad_payload.update(overrides)
        if match not in {"artifact_sha256", "strict finite JSON"}:
            refresh_artifact_hash(bad_payload)

        with pytest.raises(ValueError, match=match):
            validate_qpu_payload(bad_payload)
