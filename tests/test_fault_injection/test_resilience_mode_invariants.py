# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for resilience-mode dataclass invariants and input validation

"""Contracts for resilience-mode config/report invariants and bitstream validation."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from sc_neurocore.fault_injection import (
    FaultInjectionResilienceMode,
    FaultModel,
    RadiationProfile,
    ResilienceModeConfig,
)
from sc_neurocore.fault_injection.resilience_policy import GracefulDegradationPolicy

_BITSTREAMS = np.array([[0, 1, 0, 1, 1, 0, 0, 1], [1, 0, 0, 1, 0, 1, 1, 0]], dtype=np.uint8)


def _mode() -> FaultInjectionResilienceMode:
    return FaultInjectionResilienceMode(
        ResilienceModeConfig(
            layer_id="layer0",
            radiation_profile=RadiationProfile("test", 0.25, "deterministic stress"),
            fault_models=(FaultModel.BIT_FLIP,),
            num_trials=8,
            seed=7,
            policy=GracefulDegradationPolicy(
                warning_affected_ratio=0.01, critical_affected_ratio=0.9
            ),
        )
    )


@pytest.mark.parametrize(
    "override",
    [
        {"radiation_profile": "not a profile"},
        {"fault_models": ("not a model",)},
        {"policy": "not a policy"},
    ],
)
def test_config_rejects_invalid_fields(override: dict[str, Any]) -> None:
    """ResilienceModeConfig rejects a wrong-typed radiation profile, fault model or policy."""
    with pytest.raises(ValueError):
        dataclasses.replace(_mode().config, **override)


@pytest.mark.parametrize(
    "override",
    [
        {"fault_model": "not a model"},
        {"ber": 2.0},
        {"num_trials": 0},
        {"expected_affected_bits": float("inf")},
        {"p95_probability_error": 0.5, "p99_probability_error": 0.1},
        {"p95_probability_error": 0.1, "p99_probability_error": 0.5, "max_probability_error": 0.2},
        {
            "mean_probability_error": 0.9,
            "p95_probability_error": 0.1,
            "p99_probability_error": 0.2,
            "max_probability_error": 0.3,
        },
        {"observed_mean_affected_bits": 1.0e6},
        {"degradation_plan": "not a plan"},
    ],
)
def test_trial_report_rejects_invalid_fields(override: dict[str, Any]) -> None:
    """Each ResilienceModeTrialReport ordering/range invariant rejects its bad field."""
    trial = _mode().run(_BITSTREAMS).trial_reports[0]
    with pytest.raises(ValueError):
        dataclasses.replace(trial, **override)


@pytest.mark.parametrize(
    "override",
    [
        {"layer_id": "  "},
        {"radiation_profile": "not a profile"},
        {"seed": True},
        {"input_shape": (0, 0)},
        {"recommended_action": "not an action"},
        {"trial_reports": ()},
        {"trial_reports": ("not a report",)},
    ],
)
def test_report_rejects_invalid_fields(override: dict[str, Any]) -> None:
    """Each ResilienceModeReport invariant rejects its malformed field."""
    report = _mode().run(_BITSTREAMS)
    with pytest.raises(ValueError):
        dataclasses.replace(report, **override)


def test_run_rejects_non_numeric_bitstreams() -> None:
    """A non-numeric bitstream dtype is rejected before processing."""
    with pytest.raises(ValueError, match="numeric dtype"):
        _mode().run(np.array([["a", "b"]], dtype="<U1"))


def test_run_rejects_empty_bitstreams() -> None:
    """An empty bitstream array is rejected for having no neurons or bits."""
    with pytest.raises(ValueError, match="at least one neuron and one bit"):
        _mode().run(np.zeros((0, 8), dtype=np.uint8))


def test_run_rejects_non_2d_bitstreams() -> None:
    """A bitstream array that is not 2-D is rejected."""
    with pytest.raises(ValueError, match="shape"):
        _mode().run(np.array([0, 1, 0, 1], dtype=np.uint8))
