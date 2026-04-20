# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ArcaneZenith Cognitive Core tests

"""Multi-angle tests for ``sc_neurocore.arcane_zenith``.

The ArcaneZenith core glues an ``ArcaneNeuron`` (5-compartment
self-referential cognition neuron) to four reward-modulated STDP
plasticity rules whose scalar weights ``w ∈ [0, 1]`` are mapped via a
sharpened sigmoid to biologically plausible ranges for the neuron's
four meta-parameters:

    tau_deep           ∈ [1000, 50000] ms
    surprise_baseline  ∈ [0.01, 0.5]
    delta_conf         ∈ [0, 1]
    lr_base            ∈ [0.001, 0.1]

Tests cover: the sigmoid mapping itself (monotonicity, endpoints,
midpoint), construction via factory + direct, the ``step`` contract,
biological-range invariants across many steps, ``step_from_bio_rates``
with arbitrary firing-rate dicts (including empty), ``reset`` semantics
(spike compartments clear, identity persists), ``get_state`` /
``get_state_dict`` round-trip, and an end-to-end stability check.

Tests use the ``"torch"`` plasticity backend so the suite runs on
machines without ``libautonomous_learning``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from sc_neurocore.arcane_zenith import (
    ArcaneZenithCognitiveCore,
    create_arcane_neuron_with_zenith_plasticity,
)
from sc_neurocore.neurons.models.arcane_neuron import ArcaneNeuron


# ---------------------------------------------------------------------------
# Sigmoid mapping — the only closed-form mathematical primitive in the module.
# ---------------------------------------------------------------------------


class TestSigmoidMapping:
    """``_map_to_range`` = sigmoid(10*(w-0.5)) interpolated into [min, max]."""

    @pytest.fixture
    def core(self) -> ArcaneZenithCognitiveCore:
        return create_arcane_neuron_with_zenith_plasticity(backend="torch")

    def test_endpoint_w_zero_approaches_min(self, core):
        # sigmoid(-5) ≈ 0.0067 → result ≈ min + 0.0067*(max-min)
        out = core._map_to_range(0.0, 10.0, 110.0)
        assert 10.0 <= out <= 11.0
        expected = 10.0 + (1.0 / (1.0 + math.exp(5.0))) * 100.0
        assert abs(out - expected) < 1e-6

    def test_endpoint_w_one_approaches_max(self, core):
        # sigmoid(+5) ≈ 0.9933 → result ≈ min + 0.9933*(max-min)
        out = core._map_to_range(1.0, 10.0, 110.0)
        assert 109.0 <= out <= 110.0
        expected = 10.0 + (1.0 / (1.0 + math.exp(-5.0))) * 100.0
        assert abs(out - expected) < 1e-6

    def test_midpoint_w_half_is_exact_midpoint(self, core):
        out = core._map_to_range(0.5, 10.0, 110.0)
        assert abs(out - 60.0) < 1e-9

    def test_strict_monotonic_in_weight(self, core):
        samples = np.linspace(0.0, 1.0, 41)
        mapped = [core._map_to_range(float(w), 0.0, 1.0) for w in samples]
        assert all(mapped[i + 1] > mapped[i] for i in range(len(mapped) - 1))

    def test_clamp_above_one_saturates_at_max(self, core):
        # Sigmoid saturates: extreme w still cannot exceed max by construction.
        assert core._map_to_range(5.0, 10.0, 20.0) == pytest.approx(20.0, abs=1e-3)

    def test_clamp_below_zero_saturates_at_min(self, core):
        assert core._map_to_range(-5.0, 10.0, 20.0) == pytest.approx(10.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Construction + factory surface.
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_factory_returns_instance(self):
        core = create_arcane_neuron_with_zenith_plasticity(backend="torch")
        assert isinstance(core, ArcaneZenithCognitiveCore)

    def test_direct_init_wires_arcane_neuron(self):
        core = ArcaneZenithCognitiveCore(backend="torch")
        assert isinstance(core.neuron, ArcaneNeuron)

    def test_four_independent_plasticity_rules(self):
        core = ArcaneZenithCognitiveCore(backend="torch")
        rules = (core.tau_rule, core.nov_rule, core.conf_rule, core.lr_rule)
        # Same concrete class but four distinct object identities.
        assert len({id(r) for r in rules}) == 4

    def test_initial_weights_follow_design(self):
        core = ArcaneZenithCognitiveCore(backend="torch")
        # Documented initialisation in arcane_zenith.py:
        # tau=0.5, nov=0.2, conf=0.3, lr=0.1
        assert float(core.tau_rule.get_weights()[0]) == pytest.approx(0.5, abs=1e-6)
        assert float(core.nov_rule.get_weights()[0]) == pytest.approx(0.2, abs=1e-6)
        assert float(core.conf_rule.get_weights()[0]) == pytest.approx(0.3, abs=1e-6)
        assert float(core.lr_rule.get_weights()[0]) == pytest.approx(0.1, abs=1e-6)

    def test_unknown_backend_rejected(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            ArcaneZenithCognitiveCore(backend="not-a-backend")


# ---------------------------------------------------------------------------
# step() contract.
# ---------------------------------------------------------------------------


class TestStep:
    @pytest.fixture
    def core(self) -> ArcaneZenithCognitiveCore:
        return create_arcane_neuron_with_zenith_plasticity(backend="torch")

    def test_step_returns_spike_bit(self, core):
        out = core.step(5.0)
        assert out in (0, 1)

    def test_step_advances_neuron_clock(self, core):
        core.step(0.0)
        core.step(0.0)
        core.step(0.0)
        assert core.neuron.get_state()["total_steps"] == 3

    def test_step_keeps_tau_deep_in_biological_range(self, core):
        for _ in range(200):
            core.step(2.5)
        assert 1000.0 <= core.neuron.tau_deep <= 50000.0

    def test_step_keeps_surprise_baseline_in_biological_range(self, core):
        for _ in range(200):
            core.step(2.5)
        assert 0.01 <= core.neuron.surprise_baseline <= 0.5

    def test_step_keeps_delta_conf_in_biological_range(self, core):
        for _ in range(200):
            core.step(2.5)
        assert 0.0 <= core.neuron.delta_conf <= 1.0

    def test_step_keeps_lr_base_in_biological_range(self, core):
        for _ in range(200):
            core.step(2.5)
        assert 0.001 <= core.neuron.lr_base <= 0.1

    def test_step_zero_current_runs_without_error(self, core):
        # No input → neuron stays sub-threshold, but the plasticity rules
        # still step and the meta-parameters still track to biological
        # ranges deterministically.
        for _ in range(50):
            core.step(0.0)
        assert 1000.0 <= core.neuron.tau_deep <= 50000.0


# ---------------------------------------------------------------------------
# step_from_bio_rates — wiring from a multi-channel rate dict.
# ---------------------------------------------------------------------------


class TestStepFromBioRates:
    @pytest.fixture
    def core(self) -> ArcaneZenithCognitiveCore:
        return create_arcane_neuron_with_zenith_plasticity(backend="torch")

    def test_populated_dict_advances_by_one_step(self, core):
        core.step_from_bio_rates({0: 10.0, 1: 20.0, 2: 30.0})
        assert core.neuron.get_state()["total_steps"] == 1

    def test_empty_dict_treated_as_zero_current(self, core):
        # Empty dict → mean 0.0 → equivalent to step(0.0). Must not raise.
        core.step_from_bio_rates({})
        assert core.neuron.get_state()["total_steps"] == 1

    def test_multiple_calls_keep_parameters_bounded(self, core):
        for i in range(100):
            core.step_from_bio_rates({0: float(i % 50), 1: float((i * 3) % 40)})
        assert 1000.0 <= core.neuron.tau_deep <= 50000.0
        assert 0.01 <= core.neuron.surprise_baseline <= 0.5
        assert 0.0 <= core.neuron.delta_conf <= 1.0
        assert 0.001 <= core.neuron.lr_base <= 0.1


# ---------------------------------------------------------------------------
# reset — spike compartments clear, identity persists.
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_fast_and_working_compartments(self):
        core = ArcaneZenithCognitiveCore(backend="torch")
        for _ in range(50):
            core.step(5.0)
        core.reset()
        assert core.neuron.v_fast == 0.0
        assert core.neuron.v_work == 0.0

    def test_reset_preserves_identity_deep_compartment(self):
        # v_deep is the identity of the neuron; reset() must not clear it.
        core = ArcaneZenithCognitiveCore(backend="torch")
        for _ in range(200):
            core.step(2.5)
        v_deep_before = core.neuron.v_deep
        core.reset()
        assert core.neuron.v_deep == v_deep_before

    def test_reset_zeroes_identity_drift_accumulator(self):
        core = ArcaneZenithCognitiveCore(backend="torch")
        for _ in range(50):
            core.step(5.0)
        core.reset()
        assert core.neuron.identity_drift == 0.0


# ---------------------------------------------------------------------------
# get_state + state-dict serialisation.
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_get_state_contains_neuron_and_four_weights(self):
        core = ArcaneZenithCognitiveCore(backend="torch")
        state = core.get_state()
        # Neuron state keys:
        for key in ("v_fast", "v_work", "v_deep", "confidence", "novelty"):
            assert key in state
        # Plasticity weight keys:
        for key in ("w_tau", "w_nov", "w_conf", "w_lr"):
            assert key in state
            assert isinstance(state[key], float)

    def test_state_dict_roundtrip_restores_four_weights(self):
        src = ArcaneZenithCognitiveCore(backend="torch")
        # Drive the rules so weights diverge from the defaults.
        for _ in range(100):
            src.step(3.0)
        sd = src.get_state_dict()
        assert set(sd.keys()) == {"tau_rule", "nov_rule", "conf_rule", "lr_rule"}

        dst = ArcaneZenithCognitiveCore(backend="torch")
        dst.load_state_dict(sd)
        for name in ("tau_rule", "nov_rule", "conf_rule", "lr_rule"):
            src_w = float(getattr(src, name).get_weights()[0])
            dst_w = float(getattr(dst, name).get_weights()[0])
            assert src_w == pytest.approx(dst_w, abs=1e-6)


# ---------------------------------------------------------------------------
# End-to-end integration — bounded identity dynamics over many steps.
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_long_run_keeps_all_meta_parameters_bounded(self):
        """Across 1000 steps of varied input, the four meta-parameters must
        stay strictly inside the biological ranges the module documents.
        Any escape indicates either a broken clamp in ``_map_to_range`` or
        an out-of-bounds weight leaking from the plasticity rule."""
        core = create_arcane_neuron_with_zenith_plasticity(backend="torch")
        rng = np.random.default_rng(seed=20260420)
        for _ in range(1000):
            current = float(rng.uniform(-5.0, 10.0))
            core.step(current)
            assert 1000.0 <= core.neuron.tau_deep <= 50000.0
            assert 0.01 <= core.neuron.surprise_baseline <= 0.5
            assert 0.0 <= core.neuron.delta_conf <= 1.0
            assert 0.001 <= core.neuron.lr_base <= 0.1

    def test_identity_drift_monotonic_non_decreasing(self):
        """``identity_drift`` accumulates |Δv_deep|, so it may only ever
        grow (or stay flat when v_deep is stationary)."""
        core = create_arcane_neuron_with_zenith_plasticity(backend="torch")
        drifts = [core.neuron.identity_drift]
        for _ in range(200):
            core.step(3.0)
            drifts.append(core.neuron.identity_drift)
        assert all(drifts[i + 1] >= drifts[i] for i in range(len(drifts) - 1))
