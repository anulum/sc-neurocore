# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fault-injection resilience mode

"""Deterministic fault-injection resilience mode for SC bitstream layers.

The mode runs seeded Bernoulli fault trials directly on the supplied
stochastic-computing bitstreams, records population-probability drift, and
combines the measurements with the graceful-degradation policy. Radiation
profiles are engineering stress presets; mission acceptance still requires a
project-specific radiation environment analysis and hardware evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from sc_neurocore.fault_injection.fault_injection import (
    FaultInjector,
    FaultModel,
    RadiationProfile,
)
from sc_neurocore.fault_injection.resilience_policy import (
    DegradationAction,
    DegradationPlan,
    GracefulDegradationPolicy,
)


_ACTION_RANK: dict[DegradationAction, int] = {
    DegradationAction.NOMINAL: 0,
    DegradationAction.EXTEND_BITSTREAM: 1,
    DegradationAction.REPLAY_WITH_SEED: 2,
}


@dataclass(frozen=True)
class ResilienceModeConfig:
    """Configuration for a resilience-mode run over one bitstream layer."""

    layer_id: str
    radiation_profile: RadiationProfile
    fault_models: tuple[FaultModel, ...] = (
        FaultModel.BIT_FLIP,
        FaultModel.STUCK_AT_0,
        FaultModel.STUCK_AT_1,
        FaultModel.DROPOUT,
    )
    num_trials: int = 128
    seed: int = 0
    policy: GracefulDegradationPolicy = field(default_factory=GracefulDegradationPolicy)

    def __post_init__(self) -> None:
        if not isinstance(self.layer_id, str) or not self.layer_id.strip():
            raise ValueError("layer_id must be a non-empty string")
        if not isinstance(self.radiation_profile, RadiationProfile):
            raise ValueError("radiation_profile must be a RadiationProfile")
        if not isinstance(self.fault_models, tuple) or not self.fault_models:
            raise ValueError("fault_models must be a non-empty tuple")
        if any(not isinstance(model, FaultModel) for model in self.fault_models):
            raise ValueError("fault_models must contain FaultModel entries")
        if (
            isinstance(self.num_trials, bool)
            or not isinstance(self.num_trials, int)
            or self.num_trials <= 0
        ):
            raise ValueError("num_trials must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if not isinstance(self.policy, GracefulDegradationPolicy):
            raise ValueError("policy must be a GracefulDegradationPolicy")


@dataclass(frozen=True)
class ResilienceModeTrialReport:
    """Aggregate measurements for one fault model."""

    fault_model: FaultModel
    ber: float
    num_trials: int
    bit_count: int
    expected_affected_bits: float
    observed_mean_affected_bits: float
    observed_std_affected_bits: float
    mean_probability_error: float
    p95_probability_error: float
    p99_probability_error: float
    max_probability_error: float
    degradation_plan: DegradationPlan

    def __post_init__(self) -> None:
        if not isinstance(self.fault_model, FaultModel):
            raise ValueError("fault_model must be a FaultModel")
        if (
            isinstance(self.ber, bool)
            or not np.isfinite(float(self.ber))
            or not (0.0 <= float(self.ber) <= 1.0)
        ):
            raise ValueError("ber must be a finite value in [0, 1]")
        for field_name in ("num_trials", "bit_count"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        for field_name in (
            "expected_affected_bits",
            "observed_mean_affected_bits",
            "observed_std_affected_bits",
            "mean_probability_error",
            "p95_probability_error",
            "p99_probability_error",
            "max_probability_error",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not np.isfinite(float(value))
            ):
                raise ValueError(f"{field_name} must be a finite numeric value")
            if float(value) < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.p99_probability_error < self.p95_probability_error:
            raise ValueError("p99_probability_error must be >= p95_probability_error")
        if self.max_probability_error < self.p99_probability_error:
            raise ValueError("max_probability_error must be >= p99_probability_error")
        if self.max_probability_error < self.mean_probability_error:
            raise ValueError("max_probability_error must be >= mean_probability_error")
        if self.observed_mean_affected_bits > self.bit_count:
            raise ValueError("observed_mean_affected_bits cannot exceed bit_count")
        if not isinstance(self.degradation_plan, DegradationPlan):
            raise ValueError("degradation_plan must be a DegradationPlan")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report."""
        return {
            "fault_model": self.fault_model.value,
            "ber": self.ber,
            "num_trials": self.num_trials,
            "bit_count": self.bit_count,
            "expected_affected_bits": self.expected_affected_bits,
            "observed_mean_affected_bits": self.observed_mean_affected_bits,
            "observed_std_affected_bits": self.observed_std_affected_bits,
            "mean_probability_error": self.mean_probability_error,
            "p95_probability_error": self.p95_probability_error,
            "p99_probability_error": self.p99_probability_error,
            "max_probability_error": self.max_probability_error,
            "degradation_plan": self.degradation_plan.to_dict(),
        }


@dataclass(frozen=True)
class ResilienceModeReport:
    """Full resilience-mode output for one layer and radiation profile."""

    layer_id: str
    radiation_profile: RadiationProfile
    seed: int
    input_shape: tuple[int, int]
    nominal_probability: float
    recommended_action: DegradationAction
    trial_reports: tuple[ResilienceModeTrialReport, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.layer_id, str) or not self.layer_id.strip():
            raise ValueError("layer_id must be a non-empty string")
        if not isinstance(self.radiation_profile, RadiationProfile):
            raise ValueError("radiation_profile must be a RadiationProfile")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if (
            not isinstance(self.input_shape, tuple)
            or len(self.input_shape) != 2
            or any(
                isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in self.input_shape
            )
        ):
            raise ValueError("input_shape must be a 2-tuple of positive integers")
        if not np.isfinite(float(self.nominal_probability)) or not (
            0.0 <= float(self.nominal_probability) <= 1.0
        ):
            raise ValueError("nominal_probability must be a finite value in [0, 1]")
        if not isinstance(self.recommended_action, DegradationAction):
            raise ValueError("recommended_action must be a DegradationAction")
        if not isinstance(self.trial_reports, tuple) or not self.trial_reports:
            raise ValueError("trial_reports must be a non-empty tuple")
        if any(not isinstance(report, ResilienceModeTrialReport) for report in self.trial_reports):
            raise ValueError("trial_reports must contain ResilienceModeTrialReport entries")

    @property
    def requires_replay(self) -> bool:
        """Whether any fault model requires deterministic replay."""
        return self.recommended_action == DegradationAction.REPLAY_WITH_SEED

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report."""
        return {
            "layer_id": self.layer_id,
            "radiation_profile": {
                "name": self.radiation_profile.name,
                "ber": self.radiation_profile.ber,
                "description": self.radiation_profile.description,
            },
            "seed": self.seed,
            "input_shape": list(self.input_shape),
            "nominal_probability": self.nominal_probability,
            "recommended_action": self.recommended_action.value,
            "requires_replay": self.requires_replay,
            "trial_reports": [report.to_dict() for report in self.trial_reports],
        }


class FaultInjectionResilienceMode:
    """Run seeded fault-injection trials and degradation policy together."""

    def __init__(self, config: ResilienceModeConfig) -> None:
        if not isinstance(config, ResilienceModeConfig):
            raise ValueError("config must be a ResilienceModeConfig")
        self.config = config

    def run(self, bitstreams: np.ndarray[Any, Any]) -> ResilienceModeReport:
        """Evaluate resilience for a binary ``(neurons, bits)`` layer."""
        streams = self._validate_bitstreams(bitstreams)
        flat = streams.reshape(-1)
        nominal_probability = float(np.mean(flat))
        trial_reports = tuple(
            self._run_fault_model(streams, flat, model, model_index)
            for model_index, model in enumerate(self.config.fault_models)
        )
        recommended = max(
            (report.degradation_plan.action for report in trial_reports),
            key=lambda action: _ACTION_RANK[action],
        )
        return ResilienceModeReport(
            layer_id=self.config.layer_id,
            radiation_profile=self.config.radiation_profile,
            seed=self.config.seed,
            input_shape=(int(streams.shape[0]), int(streams.shape[1])),
            nominal_probability=nominal_probability,
            recommended_action=recommended,
            trial_reports=trial_reports,
        )

    def _run_fault_model(
        self,
        streams: np.ndarray[Any, Any],
        flat: np.ndarray[Any, Any],
        fault_model: FaultModel,
        model_index: int,
    ) -> ResilienceModeTrialReport:
        errors = np.empty(self.config.num_trials, dtype=np.float64)
        affected = np.empty(self.config.num_trials, dtype=np.float64)
        nominal_probability = float(np.mean(flat))
        for trial_index in range(self.config.num_trials):
            seed = self.config.seed + model_index * self.config.num_trials + trial_index
            corrupted, affected_bits = FaultInjector(seed=seed).inject(
                flat,
                fault_model,
                self.config.radiation_profile.ber,
            )
            affected[trial_index] = affected_bits
            errors[trial_index] = abs(nominal_probability - float(np.mean(corrupted)))

        plan = self.config.policy.evaluate(
            streams,
            layer_id=self.config.layer_id,
            fault_model=fault_model,
            ber=self.config.radiation_profile.ber,
            seed=self.config.seed + model_index * self.config.num_trials,
        )
        return ResilienceModeTrialReport(
            fault_model=fault_model,
            ber=self.config.radiation_profile.ber,
            num_trials=self.config.num_trials,
            bit_count=int(flat.size),
            expected_affected_bits=float(flat.size * self.config.radiation_profile.ber),
            observed_mean_affected_bits=float(np.mean(affected)),
            observed_std_affected_bits=float(np.std(affected)),
            mean_probability_error=float(np.mean(errors)),
            p95_probability_error=float(np.percentile(errors, 95)),
            p99_probability_error=float(np.percentile(errors, 99)),
            max_probability_error=float(np.max(errors)),
            degradation_plan=plan,
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
