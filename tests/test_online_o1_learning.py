# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for O(1) online learning contracts

from __future__ import annotations

from typing import Any

import pytest

from sc_neurocore.learning.online_o1 import (
    OnlineO1Config,
    OnlineO1Synapse,
    build_online_o1_memory_proof,
)


def test_online_o1_synapse_uses_bounded_state_and_saturating_reward_updates() -> None:
    config = OnlineO1Config(
        weight_bits=8,
        trace_bits=6,
        reward_bits=4,
        learning_shift=2,
        trace_decay_shift=1,
    )
    synapse = OnlineO1Synapse(config=config, initial_weight=120)

    weights = []
    for _ in range(64):
        snapshot = synapse.step(pre_spike=True, post_spike=True, reward=7)
        weights.append(snapshot.weight)
        assert 0 <= snapshot.weight <= 255
        assert 0 <= snapshot.pre_trace <= 63
        assert 0 <= snapshot.post_trace <= 63
        assert -32 <= snapshot.eligibility <= 31

    assert weights[-1] == 255
    assert synapse.state_fields == ("weight", "pre_trace", "post_trace", "eligibility")
    assert synapse.state_bit_count == config.per_synapse_state_bits


def test_online_o1_synapse_depresses_with_negative_reward_and_preserves_bounds() -> None:
    config = OnlineO1Config(weight_bits=8, trace_bits=7, reward_bits=4, learning_shift=3)
    synapse = OnlineO1Synapse(config=config, initial_weight=120)

    # Build positive eligibility from pre-before-post pairings.
    for _ in range(8):
        synapse.step(pre_spike=True, post_spike=False, reward=0)
        synapse.step(pre_spike=False, post_spike=True, reward=0)
    assert synapse.snapshot().eligibility > 0

    for _ in range(64):
        snapshot = synapse.step(pre_spike=False, post_spike=False, reward=-7)
        assert 0 <= snapshot.weight <= 255
        assert -64 <= snapshot.eligibility <= 63

    assert synapse.snapshot().weight == 0


def test_online_o1_memory_proof_is_independent_of_sequence_length() -> None:
    config = OnlineO1Config(weight_bits=10, trace_bits=8, reward_bits=5)

    short = build_online_o1_memory_proof(n_synapses=12, config=config, sequence_length=8)
    long = build_online_o1_memory_proof(n_synapses=12, config=config, sequence_length=8192)

    assert short == long
    assert short["schema_version"] == "sc-neurocore.online-o1.memory-proof.v1"
    assert short["n_synapses"] == 12
    assert short["state_fields"] == ["weight", "pre_trace", "post_trace", "eligibility"]
    assert short["per_synapse_state_bits"] == 34
    assert short["total_state_bits"] == 408
    assert short["sequence_length_independent"] is True
    assert short["hidden_history_fields"] == []


def test_online_o1_config_rejects_non_hardware_bounded_parameters() -> None:
    with pytest.raises(ValueError, match="weight_bits"):
        OnlineO1Config(weight_bits=0)
    with pytest.raises(ValueError, match="weight_bits"):
        OnlineO1Config(weight_bits=32)
    with pytest.raises(ValueError, match="trace_bits"):
        OnlineO1Config(trace_bits=1)
    with pytest.raises(ValueError, match="trace_bits"):
        OnlineO1Config(trace_bits=31)
    with pytest.raises(ValueError, match="reward_bits"):
        OnlineO1Config(reward_bits=0)
    with pytest.raises(ValueError, match="reward_bits"):
        OnlineO1Config(reward_bits=31)
    with pytest.raises(ValueError, match="learning_shift"):
        OnlineO1Config(learning_shift=-1)
    with pytest.raises(ValueError, match="learning_shift"):
        OnlineO1Config(learning_shift=31)
    with pytest.raises(ValueError, match="trace_decay_shift"):
        OnlineO1Config(trace_decay_shift=-1)
    with pytest.raises(ValueError, match="trace_decay_shift"):
        OnlineO1Config(trace_decay_shift=31)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weight_bits": True},
        {"trace_bits": False},
        {"reward_bits": 4.0},
        {"learning_shift": 2.0},
        {"trace_decay_shift": "2"},
    ],
)
def test_online_o1_config_rejects_bool_and_non_integral_domains(
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(TypeError):
        OnlineO1Config(**kwargs)


def test_online_o1_synapse_rejects_non_integral_state_inputs() -> None:
    config = OnlineO1Config(weight_bits=8, trace_bits=6, reward_bits=4)

    with pytest.raises(TypeError, match="initial_weight"):
        OnlineO1Synapse(config=config, initial_weight=True)
    non_integral_initial_weight: Any = 1.5
    with pytest.raises(TypeError, match="initial_weight"):
        OnlineO1Synapse(config=config, initial_weight=non_integral_initial_weight)
    with pytest.raises(ValueError, match="initial_weight"):
        OnlineO1Synapse(config=config, initial_weight=-1)
    invalid_config: Any = "not-a-config"
    with pytest.raises(TypeError, match="config"):
        OnlineO1Synapse(config=invalid_config, initial_weight=0)

    synapse = OnlineO1Synapse(config=config, initial_weight=12)
    with pytest.raises(TypeError, match="reward"):
        synapse.step(pre_spike=True, post_spike=False, reward=False)
    non_integral_reward: Any = 2.5
    with pytest.raises(TypeError, match="reward"):
        synapse.step(pre_spike=True, post_spike=False, reward=non_integral_reward)


def test_online_o1_scnir_annotation_is_deterministic_and_claim_bounded() -> None:
    config = OnlineO1Config(weight_bits=9, trace_bits=7, reward_bits=4)

    annotation = config.to_scnir_annotation(rule_id="reward_stdp_o1")

    assert annotation == {
        "schema_version": "sc-neurocore.online-o1.annotation.v1",
        "rule_id": "reward_stdp_o1",
        "rule_family": "reward_modulated_stdp",
        "state_fields": ["weight", "pre_trace", "post_trace", "eligibility"],
        "per_synapse_state_bits": 30,
        "weight_bits": 9,
        "trace_bits": 7,
        "reward_bits": 4,
        "learning_shift": 4,
        "trace_decay_shift": 4,
        "saturation_policy": "signed_eligibility_unsigned_weight",
        "hidden_history_fields": [],
        "sequence_length_independent": True,
    }


def test_online_o1_rejects_unknown_rule_family_and_empty_annotation_id() -> None:
    invalid_rule_family: Any = "not-supported"
    with pytest.raises(ValueError, match="rule_family"):
        OnlineO1Config(rule_family=invalid_rule_family)

    with pytest.raises(ValueError, match="rule_id"):
        OnlineO1Config().to_scnir_annotation(rule_id="")


def test_online_o1_zero_decay_shift_preserves_traces_and_eligibility() -> None:
    config = OnlineO1Config(
        weight_bits=8,
        trace_bits=6,
        reward_bits=4,
        learning_shift=3,
        trace_decay_shift=0,
    )
    synapse = OnlineO1Synapse(config=config, initial_weight=10)

    synapse.step(pre_spike=True, post_spike=False, reward=0)
    positive = synapse.step(pre_spike=False, post_spike=True, reward=1)
    repeated = synapse.step(pre_spike=False, post_spike=False, reward=1)

    assert positive.pre_trace == config.max_trace
    assert positive.post_trace == config.max_trace
    assert positive.eligibility == config.max_eligibility
    assert repeated.pre_trace == positive.pre_trace
    assert repeated.post_trace == positive.post_trace
    assert repeated.eligibility == positive.eligibility


@pytest.mark.parametrize(
    ("n_synapses", "sequence_length"),
    [
        (-1, None),
        (0, -1),
    ],
)
def test_online_o1_memory_proof_rejects_negative_domains(
    n_synapses: int, sequence_length: int | None
) -> None:
    config = OnlineO1Config()
    with pytest.raises(ValueError):
        build_online_o1_memory_proof(
            n_synapses=n_synapses, config=config, sequence_length=sequence_length
        )


@pytest.mark.parametrize(
    ("n_synapses", "sequence_length"),
    [
        (True, None),
        (0, False),
        (1.0, None),
        (0, 1.0),
    ],
)
def test_online_o1_memory_proof_rejects_bool_and_non_integral_domains(
    n_synapses: Any,
    sequence_length: Any,
) -> None:
    config = OnlineO1Config()
    with pytest.raises(TypeError):
        build_online_o1_memory_proof(
            n_synapses=n_synapses, config=config, sequence_length=sequence_length
        )
