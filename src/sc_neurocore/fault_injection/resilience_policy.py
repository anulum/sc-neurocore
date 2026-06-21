# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Seeded fault-response policy

"""Seeded fault-injection feedback for graceful SC degradation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from sc_neurocore.fault_injection.fault_injection import FaultInjector, FaultModel
from sc_neurocore.stochastic_doctor.diagnostics import (
    AuditSeverity,
    BitstreamAuditReport,
    StochasticDoctor,
)


class DegradationAction(Enum):
    """Runtime action recommended after seeded fault diagnosis."""

    NOMINAL = "nominal"
    EXTEND_BITSTREAM = "extend_bitstream"
    REPLAY_WITH_SEED = "replay_with_seed"


@dataclass(frozen=True)
class SeededFaultObservation:
    """Fault-injection observation linked to deterministic replay seed."""

    layer_id: str
    seed: int
    fault_model: FaultModel
    ber: float
    affected_bits: int
    bitstream_length: int
    affected_ratio: float
    audit: BitstreamAuditReport

    def __post_init__(self) -> None:
        if not isinstance(self.layer_id, str) or not self.layer_id.strip():
            raise ValueError("layer_id must be a non-empty string")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not isinstance(self.fault_model, FaultModel):
            raise ValueError("fault_model must be a FaultModel")
        if isinstance(self.ber, bool) or not isinstance(self.ber, int | float):
            raise ValueError("ber must be numeric")
        if not np.isfinite(float(self.ber)) or float(self.ber) < 0.0 or float(self.ber) > 1.0:
            raise ValueError("ber must be a finite value in [0, 1]")
        for field_name in ("affected_bits", "bitstream_length"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.affected_bits > self.bitstream_length:
            raise ValueError("affected_bits cannot exceed bitstream_length")
        if isinstance(self.affected_ratio, bool) or not isinstance(
            self.affected_ratio, int | float
        ):
            raise ValueError("affected_ratio must be numeric")
        if (
            not np.isfinite(float(self.affected_ratio))
            or float(self.affected_ratio) < 0.0
            or float(self.affected_ratio) > 1.0
        ):
            raise ValueError("affected_ratio must be a finite value in [0, 1]")
        if not isinstance(self.audit, BitstreamAuditReport):
            raise ValueError("audit must be a BitstreamAuditReport")


@dataclass(frozen=True)
class DegradationPlan:
    """Graceful-degradation decision derived from seeded diagnostics."""

    action: DegradationAction
    observation: SeededFaultObservation
    recommended_bitstream_length: int
    replay_seed: int
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.action, DegradationAction):
            raise ValueError("action must be a DegradationAction")
        if not isinstance(self.observation, SeededFaultObservation):
            raise ValueError("observation must be a SeededFaultObservation")
        if (
            isinstance(self.recommended_bitstream_length, bool)
            or not isinstance(self.recommended_bitstream_length, int)
            or self.recommended_bitstream_length <= 0
        ):
            raise ValueError("recommended_bitstream_length must be a positive integer")
        if isinstance(self.replay_seed, bool) or not isinstance(self.replay_seed, int):
            raise ValueError("replay_seed must be an integer")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready summary without expanding full bitstreams."""
        return {
            "action": self.action.value,
            "layer_id": self.observation.layer_id,
            "seed": self.observation.seed,
            "fault_model": self.observation.fault_model.value,
            "ber": self.observation.ber,
            "affected_bits": self.observation.affected_bits,
            "affected_ratio": self.observation.affected_ratio,
            "audit_status": self.observation.audit.status.value,
            "max_correlation": self.observation.audit.max_correlation,
            "recommended_bitstream_length": self.recommended_bitstream_length,
            "replay_seed": self.replay_seed,
            "reason": self.reason,
        }


@dataclass
class GracefulDegradationPolicy:
    """Combine seeded fault injection with stochastic-doctor diagnosis."""

    doctor: StochasticDoctor = field(default_factory=StochasticDoctor)
    warning_affected_ratio: float = 0.01
    critical_affected_ratio: float = 0.05
    warning_length_multiplier: int = 2
    critical_length_multiplier: int = 4
    max_bitstream_length: int = 8192

    def __post_init__(self) -> None:
        if not isinstance(self.doctor, StochasticDoctor):
            raise ValueError("doctor must be a StochasticDoctor")
        for field_name in ("warning_affected_ratio", "critical_affected_ratio"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{field_name} must be numeric")
            value_f = float(value)
            if not np.isfinite(value_f) or value_f < 0.0 or value_f > 1.0:
                raise ValueError(f"{field_name} must be a finite value in [0, 1]")
        if self.warning_affected_ratio > self.critical_affected_ratio:
            raise ValueError("warning_affected_ratio cannot exceed critical_affected_ratio")
        for field_name in ("warning_length_multiplier", "critical_length_multiplier"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.warning_length_multiplier > self.critical_length_multiplier:
            raise ValueError("warning_length_multiplier cannot exceed critical_length_multiplier")
        if (
            isinstance(self.max_bitstream_length, bool)
            or not isinstance(self.max_bitstream_length, int)
            or self.max_bitstream_length <= 0
        ):
            raise ValueError("max_bitstream_length must be a positive integer")

    def evaluate(
        self,
        bitstreams: np.ndarray[Any, Any],
        *,
        layer_id: str,
        fault_model: FaultModel,
        ber: float,
        seed: int,
    ) -> DegradationPlan:
        """Inject seeded faults, audit the layer, and recommend degradation."""
        if not isinstance(layer_id, str) or not layer_id.strip():
            raise ValueError("layer_id must be a non-empty string")
        if not isinstance(fault_model, FaultModel):
            raise ValueError("fault_model must be a FaultModel")
        if isinstance(ber, bool) or not isinstance(ber, int | float):
            raise ValueError("ber must be a finite value in [0, 1]")
        if not np.isfinite(float(ber)) or float(ber) < 0.0 or float(ber) > 1.0:
            raise ValueError("ber must be a finite value in [0, 1]")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        streams = self._validate_bitstreams(bitstreams)
        corrupted, affected = self._inject_layer(streams, fault_model, ber, seed)
        audit = self.doctor.audit_layer(layer_id, corrupted)
        affected_ratio = affected / max(corrupted.size, 1)
        observation = SeededFaultObservation(
            layer_id=layer_id,
            seed=seed,
            fault_model=fault_model,
            ber=ber,
            affected_bits=affected,
            bitstream_length=corrupted.shape[1],
            affected_ratio=affected_ratio,
            audit=audit,
        )
        return self._plan(observation)

    def _inject_layer(
        self,
        bitstreams: np.ndarray[Any, Any],
        fault_model: FaultModel,
        ber: float,
        seed: int,
    ) -> tuple[np.ndarray[Any, Any], int]:
        corrupted = np.empty_like(bitstreams)
        affected_total = 0
        for row_index, stream in enumerate(bitstreams):
            injector = FaultInjector(seed=seed + row_index)
            corrupted_row, affected = injector.inject(stream, fault_model, ber)
            corrupted[row_index] = corrupted_row
            affected_total += affected
        return corrupted, affected_total

    def _plan(self, observation: SeededFaultObservation) -> DegradationPlan:
        status = observation.audit.status
        if (
            status == AuditSeverity.CRITICAL
            or observation.affected_ratio >= self.critical_affected_ratio
        ):
            return self._make_plan(
                DegradationAction.REPLAY_WITH_SEED,
                observation,
                self.critical_length_multiplier,
                "critical diagnostic or affected-bit ratio",
            )
        if (
            status == AuditSeverity.WARNING
            or observation.affected_ratio >= self.warning_affected_ratio
        ):
            return self._make_plan(
                DegradationAction.EXTEND_BITSTREAM,
                observation,
                self.warning_length_multiplier,
                "warning diagnostic or affected-bit ratio",
            )
        return DegradationPlan(
            action=DegradationAction.NOMINAL,
            observation=observation,
            recommended_bitstream_length=min(
                observation.bitstream_length, self.max_bitstream_length
            ),
            replay_seed=observation.seed,
            reason="diagnostics within policy thresholds",
        )

    def _make_plan(
        self,
        action: DegradationAction,
        observation: SeededFaultObservation,
        multiplier: int,
        reason: str,
    ) -> DegradationPlan:
        if isinstance(multiplier, bool) or not isinstance(multiplier, int) or multiplier <= 0:
            raise ValueError("multiplier must be a positive integer")
        recommended = min(
            self.max_bitstream_length,
            max(observation.bitstream_length, observation.bitstream_length * multiplier),
        )
        return DegradationPlan(
            action=action,
            observation=observation,
            recommended_bitstream_length=recommended,
            replay_seed=observation.seed,
            reason=reason,
        )

    @staticmethod
    def _validate_bitstreams(bitstreams: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        if not isinstance(bitstreams, np.ndarray):
            raise ValueError("bitstreams must be a numpy.ndarray")
        if not np.issubdtype(bitstreams.dtype, np.number):
            raise ValueError("bitstreams must have numeric dtype")
        if not np.isfinite(bitstreams.astype(np.float64)).all():
            raise ValueError("bitstreams must contain only finite values")
        streams = np.asarray(bitstreams, dtype=np.uint8)
        if streams.ndim != 2:
            raise ValueError("bitstreams must have shape (neurons, bits)")
        if streams.shape[0] == 0 or streams.shape[1] == 0:
            raise ValueError("bitstreams must contain at least one neuron and one bit")
        if not np.all((streams == 0) | (streams == 1)):
            raise ValueError("bitstreams must contain only 0/1 values")
        return streams
