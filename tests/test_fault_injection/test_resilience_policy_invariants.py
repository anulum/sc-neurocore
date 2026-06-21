# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for graceful-degradation policy invariants

"""Contracts for graceful-degradation policy dataclass invariants and validation."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from sc_neurocore.fault_injection import FaultModel
from sc_neurocore.fault_injection.resilience_policy import (
    DegradationPlan,
    GracefulDegradationPolicy,
)

_BITSTREAMS = np.array([[0, 1, 0, 1, 1, 0], [1, 0, 1, 0, 0, 1]], dtype=np.uint8)


def _plan() -> DegradationPlan:
    policy = GracefulDegradationPolicy(warning_affected_ratio=0.01, critical_affected_ratio=0.9)
    return policy.evaluate(
        _BITSTREAMS, layer_id="L", fault_model=FaultModel.BIT_FLIP, ber=0.5, seed=1
    )


@pytest.mark.parametrize(
    "override",
    [
        {"layer_id": "  "},
        {"seed": True},
        {"fault_model": "not a model"},
        {"ber": "not numeric"},
        {"ber": 2.0},
        {"affected_bits": -1},
        {"affected_ratio": "not numeric"},
        {"affected_ratio": 2.0},
        {"audit": "not an audit"},
    ],
)
def test_seeded_observation_rejects_invalid_fields(override: dict[str, Any]) -> None:
    """Each SeededFaultObservation invariant rejects its malformed field."""
    observation = _plan().observation
    with pytest.raises(ValueError):
        dataclasses.replace(observation, **override)


@pytest.mark.parametrize(
    "override",
    [
        {"action": "not an action"},
        {"observation": "not an observation"},
        {"replay_seed": True},
        {"reason": "  "},
    ],
)
def test_degradation_plan_rejects_invalid_fields(override: dict[str, Any]) -> None:
    """Each DegradationPlan invariant rejects its malformed field."""
    plan = _plan()
    with pytest.raises(ValueError):
        dataclasses.replace(plan, **override)


def test_policy_rejects_invalid_doctor_and_ratio() -> None:
    """GracefulDegradationPolicy rejects a wrong-typed doctor and a non-numeric ratio."""
    with pytest.raises(ValueError, match="doctor"):
        GracefulDegradationPolicy(doctor="not a doctor")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="numeric"):
        GracefulDegradationPolicy(warning_affected_ratio="high")  # type: ignore[arg-type]


def test_evaluate_rejects_non_numeric_ber() -> None:
    """evaluate rejects a non-numeric bit-error rate."""
    policy = GracefulDegradationPolicy()
    with pytest.raises(ValueError, match="ber"):
        policy.evaluate(
            _BITSTREAMS,
            layer_id="L",
            fault_model=FaultModel.BIT_FLIP,
            ber="bad",  # type: ignore[arg-type]
            seed=1,
        )


@pytest.mark.parametrize(
    ("bitstreams", "message"),
    [
        (np.array([["a", "b"]], dtype="<U1"), "numeric dtype"),
        (np.array([0, 1, 0, 1], dtype=np.uint8), "shape"),
        (np.zeros((0, 4), dtype=np.uint8), "at least one neuron and one bit"),
    ],
)
def test_evaluate_validates_bitstreams(bitstreams: np.ndarray[Any, Any], message: str) -> None:
    """evaluate validates bitstream dtype, rank and non-emptiness."""
    policy = GracefulDegradationPolicy()
    with pytest.raises(ValueError, match=message):
        policy.evaluate(bitstreams, layer_id="L", fault_model=FaultModel.BIT_FLIP, ber=0.5, seed=1)
