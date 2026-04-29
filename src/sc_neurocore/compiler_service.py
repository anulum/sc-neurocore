# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compiler service contract boundary

"""Compiler-service contracts for digital-twin and live-FPGA update loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from sc_neurocore.optimizer.sc_optimizer import HardwareBudget, LayerProfile
from sc_neurocore.optimizer.surrogate_sc_optimizer import (
    SurrogateLayerConfig,
    SurrogateOptimizerReport,
    TargetHardwareProfile,
)


class LiveUpdateKind(str, Enum):
    """Kinds of update package a compiler service may emit."""

    HOT_SWAP = "hot_swap"
    PARTIAL_RECONFIGURATION = "partial_reconfiguration"
    FULL_RESYNTHESIS_REQUIRED = "full_resynthesis_required"


@dataclass(frozen=True)
class DigitalTwinSyncContract:
    """Digital-twin sync requirements for a compiler-service request."""

    session_id: str
    twin_nodes: int = 1
    checkpoint_interval_ns: int = 1_000_000
    max_drift_us: float = 100.0
    require_replay_verification: bool = True
    event_channels: tuple[str, ...] = (
        "synthesis_observation",
        "device_telemetry",
        "fault_event",
        "update_ack",
    )

    def __post_init__(self) -> None:
        if not self.session_id:
            raise ValueError("session_id must be non-empty")
        if self.twin_nodes <= 0:
            raise ValueError("twin_nodes must be positive")
        if self.checkpoint_interval_ns <= 0:
            raise ValueError("checkpoint_interval_ns must be positive")
        if self.max_drift_us < 0:
            raise ValueError("max_drift_us must be non-negative")
        if not self.event_channels:
            raise ValueError("event_channels must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible sync contract."""

        return {
            "checkpoint_interval_ns": self.checkpoint_interval_ns,
            "event_channels": list(self.event_channels),
            "max_drift_us": self.max_drift_us,
            "require_replay_verification": self.require_replay_verification,
            "session_id": self.session_id,
            "twin_nodes": self.twin_nodes,
        }


@dataclass(frozen=True)
class LiveUpdatePolicy:
    """Policy deciding whether a compiler update needs re-synthesis."""

    hot_swap_fields: tuple[str, ...] = (
        "weights",
        "thresholds",
        "lfsr_seeds",
        "bitstream_length",
        "decorrelator",
        "routing_table",
    )
    partial_reconfiguration_fields: tuple[str, ...] = (
        "tile_mapping",
        "aer_route_overlay",
        "layer_enable_mask",
    )
    full_resynthesis_fields: tuple[str, ...] = (
        "hdl",
        "clock",
        "pdk",
        "io_shape",
        "layer_count",
        "top_module",
    )
    validation_gates: tuple[str, ...] = (
        "digital_twin_replay",
        "telemetry_bounds",
        "rollback_package",
    )

    def classify(self, changed_fields: tuple[str, ...]) -> LiveUpdateKind:
        """Classify changed fields into the minimum required update kind."""

        if not changed_fields:
            raise ValueError("changed_fields must be non-empty")
        known = (
            set(self.hot_swap_fields)
            | set(self.partial_reconfiguration_fields)
            | set(self.full_resynthesis_fields)
        )
        unknown = sorted(set(changed_fields) - known)
        if unknown:
            raise ValueError(f"unknown update fields: {', '.join(unknown)}")
        if any(field in self.full_resynthesis_fields for field in changed_fields):
            return LiveUpdateKind.FULL_RESYNTHESIS_REQUIRED
        if any(field in self.partial_reconfiguration_fields for field in changed_fields):
            return LiveUpdateKind.PARTIAL_RECONFIGURATION
        return LiveUpdateKind.HOT_SWAP

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible policy manifest."""

        return {
            "full_resynthesis_fields": list(self.full_resynthesis_fields),
            "hot_swap_fields": list(self.hot_swap_fields),
            "partial_reconfiguration_fields": list(self.partial_reconfiguration_fields),
            "validation_gates": list(self.validation_gates),
        }


@dataclass(frozen=True)
class CompilerServiceRequest:
    """One request accepted by the compiler-service contract boundary."""

    request_id: str
    target: TargetHardwareProfile
    network: tuple[LayerProfile, ...]
    changed_fields: tuple[str, ...]
    twin_sync: DigitalTwinSyncContract
    evidence_payload: dict[str, Any] | None = None
    objective: str = "minimise_power_under_accuracy_budget"

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.network:
            raise ValueError("network must contain at least one layer")
        if not self.changed_fields:
            raise ValueError("changed_fields must be non-empty")
        if not self.objective:
            raise ValueError("objective must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic request manifest."""

        return {
            "changed_fields": list(self.changed_fields),
            "evidence_payload_present": self.evidence_payload is not None,
            "network": [_layer_to_dict(layer) for layer in self.network],
            "objective": self.objective,
            "request_id": self.request_id,
            "target": _target_to_dict(self.target),
            "twin_sync": self.twin_sync.to_dict(),
        }


@dataclass(frozen=True)
class LiveUpdatePackage:
    """Update package planned for a live FPGA/digital-twin loop."""

    package_id: str
    request_id: str
    target_name: str
    kind: LiveUpdateKind
    changed_fields: tuple[str, ...]
    requires_full_resynthesis: bool
    validation_gates: tuple[str, ...]
    rollback_required: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible package summary."""

        return {
            "changed_fields": list(self.changed_fields),
            "kind": self.kind.value,
            "package_id": self.package_id,
            "request_id": self.request_id,
            "requires_full_resynthesis": self.requires_full_resynthesis,
            "rollback_required": self.rollback_required,
            "target_name": self.target_name,
            "validation_gates": list(self.validation_gates),
        }


@dataclass(frozen=True)
class CompilerServiceResponse:
    """Deterministic response from the contract boundary."""

    request: CompilerServiceRequest
    update_package: LiveUpdatePackage
    optimiser_report: SurrogateOptimizerReport | None = None
    service_status: str = "contract_only"
    accepted: bool = True
    diagnostics: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible compiler-service response."""

        return {
            "accepted": self.accepted,
            "diagnostics": list(self.diagnostics),
            "optimiser_report": _report_to_dict(self.optimiser_report),
            "request": self.request.to_dict(),
            "service_status": self.service_status,
            "update_package": self.update_package.to_dict(),
        }


def build_compiler_service_contract(
    *,
    targets: tuple[TargetHardwareProfile, ...],
    sync: DigitalTwinSyncContract,
    policy: LiveUpdatePolicy | None = None,
) -> dict[str, Any]:
    """Build a deterministic compiler-service boundary manifest."""

    if not targets:
        raise ValueError("targets must be non-empty")
    target_names = [target.name for target in targets]
    if len(set(target_names)) != len(target_names):
        raise ValueError("target names must be unique")
    active_policy = policy or LiveUpdatePolicy()
    return {
        "schema_version": "1.0",
        "service_status": "contract_only",
        "supported_targets": [
            _target_to_dict(target) for target in sorted(targets, key=_target_key)
        ],
        "sync_contract": sync.to_dict(),
        "update_policy": active_policy.to_dict(),
    }


def plan_live_update(
    request: CompilerServiceRequest,
    policy: LiveUpdatePolicy | None = None,
) -> LiveUpdatePackage:
    """Classify a request into a live-update package."""

    active_policy = policy or LiveUpdatePolicy()
    kind = active_policy.classify(request.changed_fields)
    return LiveUpdatePackage(
        package_id=f"{request.request_id}:{kind.value}",
        request_id=request.request_id,
        target_name=request.target.name,
        kind=kind,
        changed_fields=request.changed_fields,
        requires_full_resynthesis=kind is LiveUpdateKind.FULL_RESYNTHESIS_REQUIRED,
        validation_gates=active_policy.validation_gates,
    )


def build_compiler_service_response(
    request: CompilerServiceRequest,
    *,
    optimiser_report: SurrogateOptimizerReport | None = None,
    policy: LiveUpdatePolicy | None = None,
) -> CompilerServiceResponse:
    """Build a response without invoking a network service or toolchain."""

    update_package = plan_live_update(request, policy=policy)
    diagnostics = _response_diagnostics(update_package)
    return CompilerServiceResponse(
        request=request,
        update_package=update_package,
        optimiser_report=optimiser_report,
        diagnostics=diagnostics,
    )


def _response_diagnostics(update_package: LiveUpdatePackage) -> tuple[str, ...]:
    if update_package.kind is LiveUpdateKind.FULL_RESYNTHESIS_REQUIRED:
        return ("full synthesis toolchain must run before hardware update",)
    if update_package.kind is LiveUpdateKind.PARTIAL_RECONFIGURATION:
        return ("partial reconfiguration artefact must pass digital-twin replay",)
    return ("hot-swap package can be applied after validation gates pass",)


def _target_key(target: TargetHardwareProfile) -> str:
    return target.name


def _target_to_dict(target: TargetHardwareProfile) -> dict[str, Any]:
    budget = target.budget
    return {
        "accuracy_weight": target.accuracy_weight,
        "budget": _budget_to_dict(budget),
        "latency_weight": target.latency_weight,
        "lut_weight": target.lut_weight,
        "name": target.name,
        "power_weight": target.power_weight,
    }


def _budget_to_dict(budget: HardwareBudget) -> dict[str, Any]:
    return {
        "max_latency_cycles": budget.max_latency_cycles,
        "max_luts": budget.max_luts,
        "max_power_mw": budget.max_power_mw,
    }


def _layer_to_dict(layer: LayerProfile) -> dict[str, Any]:
    return {
        "id": layer.id,
        "is_critical_path": layer.is_critical_path,
        "mac_count": layer.mac_count,
    }


def _report_to_dict(report: SurrogateOptimizerReport | None) -> dict[str, Any] | None:
    if report is None:
        return None
    return {
        "config": {
            layer_id: _surrogate_layer_config_to_dict(config)
            for layer_id, config in sorted(report.config.items())
        },
        "feasible": report.feasible,
        "mean_accuracy": report.mean_accuracy,
        "rejected_layers": list(report.rejected_layers),
        "target_name": report.target_name,
        "total_latency_cycles": report.total_latency_cycles,
        "total_luts": report.total_luts,
        "total_power_mw": report.total_power_mw,
        "training_points": report.training_points,
    }


def _surrogate_layer_config_to_dict(config: SurrogateLayerConfig) -> dict[str, Any]:
    return {
        "accuracy_score": config.accuracy_score,
        "bitstream_length": config.bitstream_length,
        "decorrelator": config.decorrelator,
        "latency_cycles": config.latency_cycles,
        "lfsr_polynomial": config.lfsr_polynomial,
        "luts_used": config.luts_used,
        "mode": config.mode,
        "power_used": config.power_used,
        "precision_bits": config.precision_bits,
        "utility_score": config.utility_score,
    }
