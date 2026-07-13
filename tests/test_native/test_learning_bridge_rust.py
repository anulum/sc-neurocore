# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Scalar Rust learning-wrapper tests

"""Tests for scalar native ownership, validation, and batched boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore._native import learning_runtime as runtime
from sc_neurocore._native.learning_rust import (
    RustEligentLearner,
    RustOnlineO1Synapse,
    RustPlasticityRule,
)

from test_learning_bridge_support import FakeCdll, FakeLearningLib

pytest_plugins = ("test_learning_bridge_support",)


def test_native_constructors_fail_when_library_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime, "_HAS_LEARNING", False)
    monkeypatch.setattr(runtime, "_lib", None)
    for constructor in (RustOnlineO1Synapse, RustPlasticityRule, RustEligentLearner):
        with pytest.raises(RuntimeError, match="not available"):
            constructor()


def test_online_o1_lifecycle_and_saturation(
    fake_learning_lib: FakeLearningLib,
) -> None:
    synapse = RustOnlineO1Synapse(initial_weight=(1 << 32) + 1)
    snapshot = synapse.step(pre_spike=True, post_spike=False, reward=1 << 40)
    assert (snapshot.weight, snapshot.pre_trace, snapshot.post_trace, snapshot.eligibility) == (
        22,
        48,
        63,
        31,
    )
    assert synapse.per_synapse_state_bits == 26
    synapse.close()
    synapse.close()
    assert fake_learning_lib.destroyed["online"] == [fake_learning_lib.online_ptr]
    with pytest.raises(RuntimeError, match="closed"):
        synapse.__enter__()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weight_bits": 0},
        {"trace_bits": 1},
        {"reward_bits": 31},
        {"learning_shift": 31},
        {"trace_decay_shift": 31},
        {"initial_weight": -1},
    ],
)
def test_online_o1_rejects_invalid_configuration(
    fake_learning_lib: FakeLearningLib, kwargs: dict[str, int]
) -> None:
    del fake_learning_lib
    with pytest.raises(ValueError):
        RustOnlineO1Synapse(**kwargs)


def test_online_o1_rejects_invalid_events(fake_learning_lib: FakeLearningLib) -> None:
    del fake_learning_lib
    synapse = RustOnlineO1Synapse()
    with pytest.raises(TypeError, match="pre_spike"):
        synapse.step(pre_spike=1, post_spike=False, reward=0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="reward"):
        synapse.step(pre_spike=True, post_spike=False, reward=1.5)  # type: ignore[arg-type]


def test_online_o1_requires_extension_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime, "_HAS_LEARNING", True)
    monkeypatch.setattr(runtime, "_lib", FakeCdll({"create_online_o1_synapse"}))
    with pytest.raises(RuntimeError, match="required symbol"):
        RustOnlineO1Synapse()


def test_online_o1_rejects_null_handle(fake_learning_lib: FakeLearningLib) -> None:
    fake_learning_lib.online_ptr = 0
    with pytest.raises(ValueError, match="configuration"):
        RustOnlineO1Synapse()


def test_rule_scalar_batch_and_context_lifecycle(fake_learning_lib: FakeLearningLib) -> None:
    with RustPlasticityRule(weight=0.25) as rule:
        rule.step(True, False, dt=0.2, reward=-0.4)
        rule.step_batched([True, False], [False, True], [0.1, -0.1], dt=0.3)
        assert rule.weight == pytest.approx(0.75)
        rule.reset()
    pre, post, reward, timestep = fake_learning_lib.rule_steps[0]
    assert (pre, post) == (True, False)
    assert reward == pytest.approx(-0.4)
    assert timestep == pytest.approx(0.2)
    assert fake_learning_lib.batch_counts == [2]
    assert fake_learning_lib.destroyed["rule"] == [fake_learning_lib.rule_ptr]


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"rule_type": 7}, ValueError),
        ({"weight": 1.1}, ValueError),
        ({"param_a": -1.0}, ValueError),
        ({"param_b": float("nan")}, ValueError),
    ],
)
def test_rule_constructor_validates_domains(
    fake_learning_lib: FakeLearningLib,
    kwargs: dict[str, object],
    error: type[Exception],
) -> None:
    del fake_learning_lib
    with pytest.raises(error):
        RustPlasticityRule(**kwargs)


def test_rule_rejects_null_handle(fake_learning_lib: FakeLearningLib) -> None:
    fake_learning_lib.rule_ptr = 0
    with pytest.raises(RuntimeError, match="construction failed"):
        RustPlasticityRule()


def test_rule_rejects_unsafe_events_and_batches(fake_learning_lib: FakeLearningLib) -> None:
    del fake_learning_lib
    rule = RustPlasticityRule()
    with pytest.raises(TypeError, match="pre_spike"):
        rule.step(1, False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dt"):
        rule.step(True, False, dt=0.0)
    with pytest.raises(ValueError, match="finite"):
        rule.step(True, False, reward=float("nan"))
    with pytest.raises(ValueError, match="length"):
        rule.step_batched([True], [False, True], [0.0])
    with pytest.raises(ValueError, match="one-dimensional"):
        rule.step_batched(np.ones((1, 1), dtype=bool), [False], [0.0])


def test_learner_scalar_batch_and_finalizer(fake_learning_lib: FakeLearningLib) -> None:
    learner = RustEligentLearner()
    learner.step(True, False, global_reward=0.5, dt=0.25)
    learner.step_batched([True, False], [False, True], [0.1, 0.2], dt=0.5)
    learner.__del__()
    assert fake_learning_lib.learner_steps == [(True, False, 0.5, 0.25)]
    assert fake_learning_lib.batch_counts == [2]
    assert fake_learning_lib.destroyed["learner"] == [fake_learning_lib.learner_ptr]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"threshold": 0.0},
        {"target_rate": -0.1},
        {"weight": 2.0},
    ],
)
def test_learner_constructor_validates_domains(
    fake_learning_lib: FakeLearningLib, kwargs: dict[str, float]
) -> None:
    del fake_learning_lib
    with pytest.raises(ValueError):
        RustEligentLearner(**kwargs)


def test_learner_rejects_null_and_invalid_events(fake_learning_lib: FakeLearningLib) -> None:
    fake_learning_lib.learner_ptr = 0
    with pytest.raises(RuntimeError, match="construction failed"):
        RustEligentLearner()
    fake_learning_lib.learner_ptr = 303
    learner = RustEligentLearner()
    with pytest.raises(TypeError, match="fired"):
        learner.step(1, False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="length"):
        learner.step_batched([True], [False, True], [0.0])
