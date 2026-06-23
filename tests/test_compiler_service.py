# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for compiler-service contract boundary

"""Tests for compiler-service digital-twin and live-update contracts."""

from __future__ import annotations

from typing import Any

import pytest

from sc_neurocore.compiler_service import (
    CompilerServiceRequest,
    DigitalTwinSyncContract,
    LiveUpdateKind,
    LiveUpdatePolicy,
    build_compiler_service_contract,
    build_compiler_service_response,
    plan_live_update,
)
from sc_neurocore.optimizer.sc_optimizer import HardwareBudget, LayerProfile
from sc_neurocore.optimizer.surrogate_sc_optimizer import (
    SurrogateLayerConfig,
    SurrogateOptimizerReport,
    TargetHardwareProfile,
)


def _target(name: str = "pynq-z2") -> TargetHardwareProfile:
    return TargetHardwareProfile(
        name=name,
        budget=HardwareBudget(max_luts=12_000, max_power_mw=2_500.0, max_latency_cycles=1024),
    )


def _request(*changed_fields: str) -> CompilerServiceRequest:
    return CompilerServiceRequest(
        request_id="req-001",
        target=_target(),
        network=(
            LayerProfile("input", mac_count=128),
            LayerProfile("hidden", mac_count=256, is_critical_path=True),
        ),
        changed_fields=changed_fields or ("weights",),
        twin_sync=DigitalTwinSyncContract(session_id="twin-a", twin_nodes=2),
        evidence_payload={"results": []},
    )


def test_build_compiler_service_contract_is_deterministic() -> None:
    manifest = build_compiler_service_contract(
        targets=(_target("z-target"), _target("a-target")),
        sync=DigitalTwinSyncContract(session_id="session-1"),
    )

    assert list(manifest) == [
        "schema_version",
        "service_status",
        "supported_targets",
        "sync_contract",
        "update_policy",
    ]
    assert manifest["service_status"] == "contract_only"
    assert [target["name"] for target in manifest["supported_targets"]] == [
        "a-target",
        "z-target",
    ]
    assert manifest["sync_contract"]["session_id"] == "session-1"


def test_compiler_service_contract_rejects_duplicate_targets() -> None:
    with pytest.raises(ValueError, match="target names"):
        build_compiler_service_contract(
            targets=(_target("dup"), _target("dup")),
            sync=DigitalTwinSyncContract(session_id="session-1"),
        )


def test_live_update_policy_classifies_hot_swap() -> None:
    package = plan_live_update(_request("weights", "lfsr_seeds"))

    assert package.kind is LiveUpdateKind.HOT_SWAP
    assert package.requires_full_resynthesis is False
    assert "digital_twin_replay" in package.validation_gates


def test_live_update_policy_classifies_partial_reconfiguration() -> None:
    package = plan_live_update(_request("routing_table", "aer_route_overlay"))

    assert package.kind is LiveUpdateKind.PARTIAL_RECONFIGURATION
    assert package.requires_full_resynthesis is False


def test_live_update_policy_classifies_full_resynthesis() -> None:
    package = plan_live_update(_request("weights", "clock"))

    assert package.kind is LiveUpdateKind.FULL_RESYNTHESIS_REQUIRED
    assert package.requires_full_resynthesis is True


def test_live_update_policy_rejects_unknown_field() -> None:
    with pytest.raises(ValueError, match="unknown update fields"):
        LiveUpdatePolicy().classify(("unknown",))


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


def test_build_response_serialises_optimizer_report() -> None:
    report = SurrogateOptimizerReport(
        config={
            "hidden": SurrogateLayerConfig(
                bitstream_length=256,
                decorrelator="LFSR",
                mode="SC",
                precision_bits=8,
                lfsr_polynomial="x16+x14+x13+x11+1",
                luts_used=512,
                power_used=4.5,
                latency_cycles=256,
                accuracy_score=0.98,
                utility_score=0.91,
            )
        },
        total_luts=512,
        total_power_mw=4.5,
        total_latency_cycles=256,
        mean_accuracy=0.98,
        training_points=32,
        target_name="pynq-z2",
    )

    response = build_compiler_service_response(
        _request("bitstream_length"), optimiser_report=report
    )
    payload = response.to_dict()

    assert payload["accepted"] is True
    assert payload["optimiser_report"]["feasible"] is True
    assert payload["optimiser_report"]["config"]["hidden"]["bitstream_length"] == 256
    assert payload["update_package"]["kind"] == "hot_swap"


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


def test_classify_rejects_empty_changed_fields() -> None:
    """Classifying no changed fields is a contract violation, raised before the
    known-field set is consulted."""
    with pytest.raises(ValueError, match="changed_fields must be non-empty"):
        LiveUpdatePolicy().classify(())


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


def test_build_compiler_service_contract_rejects_empty_targets() -> None:
    """A service boundary with no supported targets is a misconfiguration."""
    with pytest.raises(ValueError, match="targets must be non-empty"):
        build_compiler_service_contract(
            targets=(),
            sync=DigitalTwinSyncContract(session_id="session-1"),
        )


def test_response_diagnostics_vary_by_update_kind() -> None:
    """Full-resynthesis and partial-reconfiguration packages each carry their
    own dedicated diagnostic message, distinct from the hot-swap default."""
    full = build_compiler_service_response(_request("clock")).to_dict()
    partial = build_compiler_service_response(_request("aer_route_overlay")).to_dict()

    assert full["update_package"]["kind"] == "full_resynthesis_required"
    assert "full synthesis toolchain" in full["diagnostics"][0]
    assert partial["update_package"]["kind"] == "partial_reconfiguration"
    assert "partial reconfiguration" in partial["diagnostics"][0]


def test_response_without_optimiser_report_serialises_null() -> None:
    """A response built without an optimiser report serialises the report slot
    as null rather than an empty object."""
    payload = build_compiler_service_response(_request("weights")).to_dict()
    assert payload["optimiser_report"] is None
