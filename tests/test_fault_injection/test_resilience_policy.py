# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Seeded fault-response policy tests

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.fault_injection import (
    DegradationAction,
    FaultModel,
    GracefulDegradationPolicy,
)


def test_zero_ber_keeps_nominal_plan_and_replay_seed() -> None:
    bitstreams = np.array([[0, 1, 0, 1]], dtype=np.uint8)
    policy = GracefulDegradationPolicy()

    plan = policy.evaluate(
        bitstreams,
        layer_id="L0",
        fault_model=FaultModel.BIT_FLIP,
        ber=0.0,
        seed=123,
    )

    assert plan.action == DegradationAction.NOMINAL
    assert plan.replay_seed == 123
    assert plan.recommended_bitstream_length == 4
    assert plan.observation.affected_bits == 0
    assert plan.to_dict()["audit_status"] == "OK"


def test_identical_streams_trigger_replay_with_seed() -> None:
    bitstreams = np.tile(np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8), (3, 1))
    policy = GracefulDegradationPolicy(max_bitstream_length=64)

    plan = policy.evaluate(
        bitstreams,
        layer_id="correlated",
        fault_model=FaultModel.BIT_FLIP,
        ber=0.0,
        seed=77,
    )

    assert plan.action == DegradationAction.REPLAY_WITH_SEED
    assert plan.replay_seed == 77
    assert plan.recommended_bitstream_length == 32
    assert plan.observation.audit.status.value == "CRITICAL"


def test_seeded_fault_observation_is_deterministic() -> None:
    rng = np.random.default_rng(42)
    bitstreams = rng.integers(0, 2, size=(4, 64), dtype=np.uint8)
    policy = GracefulDegradationPolicy()

    first = policy.evaluate(
        bitstreams,
        layer_id="L1",
        fault_model=FaultModel.STUCK_AT_0,
        ber=0.2,
        seed=5,
    )
    second = policy.evaluate(
        bitstreams,
        layer_id="L1",
        fault_model=FaultModel.STUCK_AT_0,
        ber=0.2,
        seed=5,
    )

    assert first.to_dict() == second.to_dict()


def test_affected_ratio_can_extend_bitstream_without_critical_audit() -> None:
    bitstreams = np.array([[1, 0] * 32], dtype=np.uint8)
    policy = GracefulDegradationPolicy(
        warning_affected_ratio=0.001,
        critical_affected_ratio=1.0,
    )

    plan = policy.evaluate(
        bitstreams,
        layer_id="ratio",
        fault_model=FaultModel.BIT_FLIP,
        ber=0.05,
        seed=9,
    )

    assert plan.action == DegradationAction.EXTEND_BITSTREAM
    assert plan.recommended_bitstream_length == 128
    assert plan.observation.affected_bits > 0


def test_rejects_non_binary_bitstreams() -> None:
    policy = GracefulDegradationPolicy()

    with pytest.raises(ValueError, match="numpy.ndarray"):
        policy.evaluate(  # type: ignore[arg-type]
            [[0, 1]],
            layer_id="bad",
            fault_model=FaultModel.BIT_FLIP,
            ber=0.0,
            seed=1,
        )
    with pytest.raises(ValueError, match="finite"):
        policy.evaluate(
            np.array([[0.0, np.nan]], dtype=np.float64),
            layer_id="bad",
            fault_model=FaultModel.BIT_FLIP,
            ber=0.0,
            seed=1,
        )
    with pytest.raises(ValueError, match="0/1"):
        policy.evaluate(
            np.array([[0, 2]], dtype=np.uint8),
            layer_id="bad",
            fault_model=FaultModel.BIT_FLIP,
            ber=0.0,
            seed=1,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"layer_id": ""}, "layer_id"),
        ({"fault_model": "bit_flip"}, "fault_model"),
        ({"ber": 1.1}, "ber"),
        ({"seed": 1.5}, "seed"),
    ],
)
def test_evaluate_rejects_invalid_call_contracts(kwargs, match) -> None:
    policy = GracefulDegradationPolicy()
    values = {
        "layer_id": "L0",
        "fault_model": FaultModel.BIT_FLIP,
        "ber": 0.0,
        "seed": 1,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=match):
        policy.evaluate(np.array([[0, 1]], dtype=np.uint8), **values)  # type: ignore[arg-type]


def test_observation_affected_bits_are_bounded_by_layer_size() -> None:
    policy = GracefulDegradationPolicy()
    bitstreams = np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.uint8)
    plan = policy.evaluate(
        bitstreams,
        layer_id="L0",
        fault_model=FaultModel.BIT_FLIP,
        ber=0.2,
        seed=5,
    )
    assert 0 <= plan.observation.affected_bits <= bitstreams.size


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"warning_affected_ratio": -0.1}, "warning_affected_ratio"),
        ({"critical_affected_ratio": 1.1}, "critical_affected_ratio"),
        ({"warning_affected_ratio": 0.5, "critical_affected_ratio": 0.1}, "cannot exceed"),
        ({"warning_length_multiplier": 0}, "warning_length_multiplier"),
        ({"critical_length_multiplier": 0}, "critical_length_multiplier"),
        ({"warning_length_multiplier": 4, "critical_length_multiplier": 2}, "cannot exceed"),
        ({"max_bitstream_length": 0}, "max_bitstream_length"),
    ],
)
def test_rejects_invalid_policy_contracts(kwargs, match) -> None:
    with pytest.raises(ValueError, match=match):
        GracefulDegradationPolicy(**kwargs)  # type: ignore[arg-type]
