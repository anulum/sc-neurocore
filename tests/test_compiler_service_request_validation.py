# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compiler-service request validation contracts

"""Request and digital-twin sync validation tests for the compiler boundary."""

from __future__ import annotations

from typing import Any

import pytest

from sc_neurocore.compiler_service import CompilerServiceRequest, DigitalTwinSyncContract
from sc_neurocore.optimizer.sc_optimizer import LayerProfile

from .compiler_service_support import _request, _target


def test_compiler_service_request_manifest_contains_target_network_and_sync() -> None:
    manifest = _request("weights").to_dict()

    assert manifest["request_id"] == "req-001"
    assert manifest["evidence_payload_present"] is True
    assert manifest["target"]["budget"]["max_luts"] == 12_000
    assert manifest["network"][1]["is_critical_path"] is True
    assert manifest["twin_sync"]["twin_nodes"] == 2


def test_compiler_service_request_validation() -> None:
    with pytest.raises(ValueError, match="network"):
        CompilerServiceRequest(
            request_id="bad",
            target=_target(),
            network=(),
            changed_fields=("weights",),
            twin_sync=DigitalTwinSyncContract(session_id="twin-a"),
        )


@pytest.mark.parametrize(
    "sync",
    [
        {"session_id": ""},
        {"session_id": "x", "twin_nodes": 0},
        {"session_id": "x", "checkpoint_interval_ns": 0},
        {"session_id": "x", "max_drift_us": -1.0},
        {"session_id": "x", "event_channels": ()},
    ],
)
def test_digital_twin_sync_contract_validation(sync: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        DigitalTwinSyncContract(**sync)


@pytest.mark.parametrize(
    "overrides",
    [
        {"request_id": ""},
        {"changed_fields": ()},
        {"objective": ""},
    ],
)
def test_compiler_service_request_field_validation(overrides: dict[str, Any]) -> None:
    """Each required scalar field on the request must be non-empty: a blank
    request id, an empty changed-field tuple, and a blank objective are all
    rejected at construction."""
    kwargs: dict[str, Any] = {
        "request_id": "req-x",
        "target": _target(),
        "network": (LayerProfile("input", mac_count=64),),
        "changed_fields": ("weights",),
        "twin_sync": DigitalTwinSyncContract(session_id="twin-a"),
    }
    kwargs.update(overrides)
    with pytest.raises(ValueError):
        CompilerServiceRequest(**kwargs)
