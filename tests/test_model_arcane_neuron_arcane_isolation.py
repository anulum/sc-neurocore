# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestArcaneIsolation from former test_model_arcane_neuron.py

"""Focused suite: TestArcaneIsolation from former test_model_arcane_neuron.py."""

from __future__ import annotations

from tests.model_arcane_neuron_support import *  # noqa: F403

class TestArcaneIsolation:
    def test_defaults(self):
        n = ArcaneNeuron()
        assert n.v_fast == 0.0 and n.v_work == 0.0 and n.v_deep == 0.0
        assert n.tau_fast == 5.0 and n.tau_work == 200.0 and n.tau_deep == 10000.0
        assert n.theta == 1.0 and n.w_gate.shape == (4,) and n.w_pred.shape == (3,)

    def test_step_returns_binary(self):
        assert ArcaneNeuron().step(0.0) in (0, 1)

    def test_five_subsystems_evolve(self):
        n = ArcaneNeuron()
        for _ in range(500):
            n.step(2.0)
        state = n.get_state()
        assert state["v_fast"] != 0.0 or state["v_work"] != 0.0
        assert state["novelty"] != 0.0
        assert state["confidence"] != 0.5

    def test_state_finite(self):
        n = ArcaneNeuron()
        for _ in range(50000):
            n.step(2.0)
        state = n.get_state()
        for key, val in state.items():
            if isinstance(val, (int, float)):
                assert np.isfinite(val), f"{key} = {val}"

    def test_get_state_keys(self):
        n = ArcaneNeuron()
        state = n.get_state()
        expected_keys = {
            "v_fast",
            "v_work",
            "v_deep",
            "confidence",
            "novelty",
            "surprise",
            "prediction",
            "identity_drift",
            "meta_lr",
            "total_steps",
        }
        assert set(state.keys()) == expected_keys

    def test_state_properties_and_recent_activity(self):
        n = ArcaneNeuron()
        assert n.identity_state == n.v_deep
        assert n.confidence == n._confidence
        assert n.novelty == n._novelty
        assert n.identity_drift == n._identity_drift
        assert n.get_recent_pre_activity() == 0.0
        n.step(2.0)
        assert n.get_recent_pre_activity() in (0.0, 1.0)

    def test_three_compartments_use_closed_form_relaxation_without_spike(self):
        """Fast, working, and deep states follow exact first-order updates."""
        n = ArcaneNeuron(v_fast=0.4, v_work=0.2, v_deep=0.01, theta=100.0, dt=25.0)
        n._spike_history = [0] * 50
        n._novelty_history = [0.2] * 20
        current = 1.5
        confidence = 1.0 - np.mean(n._novelty_history)
        gate_input = (
            n.w_gate[0] * current
            + n.w_gate[1] * n.v_fast
            + n.w_gate[2] * n.v_work
            + n.w_gate[3] * confidence
        )
        gate = _stable_sigmoid(gate_input)
        expected_fast = _exact_relaxation(n.v_fast, gate * current, n.dt, n.tau_fast)
        expected_prediction = float(np.dot(n.w_pred, [expected_fast, n.v_work, n.v_deep]))
        expected_novelty = _stable_sigmoid(
            n.kappa * (abs(expected_fast - expected_prediction) - n.surprise_baseline)
        )
        expected_work = _exact_relaxation(n.v_work, 0.0, n.dt, n.tau_work)
        expected_deep_drive = n.alpha_d * expected_work * expected_novelty
        expected_deep = _exact_relaxation(n.v_deep, expected_deep_drive, n.dt, n.tau_deep)

        spike = n.step(current)

        assert spike == 0
        assert n.v_fast == pytest.approx(expected_fast)
        assert n.v_work == pytest.approx(expected_work)
        assert n.v_deep == pytest.approx(expected_deep)
        assert n._prediction == pytest.approx(expected_prediction)
        assert n._novelty == pytest.approx(expected_novelty)

    def test_large_timestep_relaxation_remains_bounded(self):
        n = ArcaneNeuron(v_fast=1000.0, v_work=5.0, v_deep=1.0, theta=1.0e9, dt=1000.0)
        n._spike_history = [0] * 50
        n.step(0.0)
        assert 0.0 <= n.v_fast <= 1000.0
        assert 0.0 <= n.v_work <= 5.0

    @pytest.mark.parametrize(
        "field",
        ["v_fast", "v_work", "v_deep", "tau_fast", "tau_work", "tau_deep", "dt"],
    )
    def test_rejects_corrupted_runtime_state_before_mutation(self, field: str):
        n = ArcaneNeuron(v_fast=0.25, v_work=0.1, v_deep=0.01)
        setattr(n, field, np.nan)
        corrupted = n.get_state()
        with pytest.raises(ValueError, match="ArcaneNeuron"):
            n.step(0.5)
        after = n.get_state()
        for key, expected in corrupted.items():
            if isinstance(expected, float) and np.isnan(expected):
                assert np.isnan(after[key])
            else:
                assert after[key] == expected

    def test_rejects_corrupted_weight_vector_before_mutation(self):
        n = ArcaneNeuron(v_fast=0.25, v_work=0.1, v_deep=0.01)
        before = n.get_state()
        n.w_gate = np.array([0.8, np.nan, 0.05, 0.05])
        with pytest.raises(ValueError, match="ArcaneNeuron"):
            n.step(0.5)
        assert n.get_state() == before

    def test_rejects_non_finite_current_before_mutation(self):
        n = ArcaneNeuron(v_fast=0.25, v_work=0.1, v_deep=0.01)
        before = n.get_state()
        with pytest.raises(ValueError, match="current"):
            n.step(np.inf)
        assert n.get_state() == before

    def test_rejects_predictor_overflow_before_mutation(self):
        n = ArcaneNeuron(v_fast=0.25, v_work=0.1, v_deep=0.01)
        before = n.get_state()
        n.w_pred = np.array([1.0e308, 1.0e308, 1.0e308])
        with np.errstate(over="ignore"), pytest.raises(ValueError, match="predictor candidate"):
            n.step(0.5)
        assert n.get_state() == before

    def test_rejects_deep_candidate_overflow_before_mutation(self):
        n = ArcaneNeuron(v_fast=0.0, v_work=1.0e308, v_deep=0.01, theta=100.0)
        before = n.get_state()
        n.alpha_d = 1.0e308
        with pytest.raises(ValueError, match="deep compartment"):
            n.step(0.0)
        assert n.get_state() == before

    def test_sigmoid_saturates_infinite_gate_input(self):
        assert ArcaneNeuron._sigmoid(np.inf) == 1.0
        assert ArcaneNeuron._sigmoid(-np.inf) == 0.0

    @pytest.mark.parametrize(
        ("mutator", "message"),
        [
            (lambda n: setattr(n, "theta", 0.0), "theta"),
            (lambda n: setattr(n, "alpha_w", -1.0), "coupling"),
            (lambda n: setattr(n, "w_pred", np.array([0.6, np.nan, 0.1])), "w_pred"),
            (lambda n: setattr(n, "_spike_history", []), "history buffers"),
            (lambda n: setattr(n, "_spike_history", [0.5] * 50), "spike history"),
            (lambda n: setattr(n, "_novelty_history", [np.nan] * 20), "novelty history"),
            (lambda n: setattr(n, "_hist_idx", -1), "history counters"),
        ],
    )
    def test_rejects_invalid_runtime_contracts_before_mutation(self, mutator, message):
        n = ArcaneNeuron(v_fast=0.25, v_work=0.1, v_deep=0.01)
        before_steps = n.get_state()["total_steps"]
        mutator(n)
        with pytest.raises(ValueError, match=message):
            n.step(0.5)
        assert n.get_state()["total_steps"] == before_steps
