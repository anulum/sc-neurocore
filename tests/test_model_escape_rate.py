# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: EscapeRateNeuron

"""Full pipeline tests for the maintained Gerstner-2000 EscapeRate cell.

The exact constant-current RC candidate drives the finite-step probability
``1 - exp(-rho_0 * exp((v-v_threshold)/delta_u) * dt)``. One private seeded
LFSR16 trial then commits either ``v_reset`` or that membrane candidate.
"""

from __future__ import annotations

import time
import math
from typing import cast

import numpy as np
import pytest

from sc_neurocore.neurons._stochastic_threshold import (
    DEFAULT_LFSR16_SEED,
    LFSR16_ADVANCES_PER_TRIAL,
    Lfsr16Threshold,
    lfsr16_advance,
    lfsr16_trial_sample,
    probability_to_lfsr16_threshold,
)
from sc_neurocore.neurons.models.escape_rate import EscapeRateNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, isi, firing_rate


def _run(neuron: EscapeRateNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestEscapeRateIsolation:
    def test_construction_all_defaults(self):
        n = EscapeRateNeuron()
        assert n.v == -70.0 and n.v_rest == -70.0 and n.v_reset == -70.0
        assert n.v_threshold == -50.0 and n.tau_m == 10.0
        assert n.rho_0 == 0.001 and n.delta_u == 3.0 and n.resistance == 1.0

    def test_step_returns_binary(self):
        assert EscapeRateNeuron().step(0.0) in (0, 1)

    def test_state_evolves(self):
        n = EscapeRateNeuron()
        v0 = n.v
        n.step(30.0)
        assert n.v != v0

    def test_state_finite_long_run(self):
        n = EscapeRateNeuron()
        for _ in range(100000):
            n.step(40.0)
        assert np.isfinite(n.v)

    def test_reset(self):
        n = EscapeRateNeuron()
        for _ in range(100):
            n.step(50.0)
        n.reset()
        assert n.v == n.v_rest


class TestEscapeRateStochasticMechanism:
    """Core: P(spike) = ρ₀·exp((V-θ)/Δu)·dt. Bernoulli trial each step."""

    def test_stochastic_spiking(self):
        n = EscapeRateNeuron()
        spikes = sum(n.step(40.0) for _ in range(50000))
        assert spikes > 100

    def test_two_runs_differ(self):
        """Distinct explicit seeds produce distinct reproducible spike trains."""
        n1 = EscapeRateNeuron(seed=1)
        n2 = EscapeRateNeuron(seed=2)
        t1 = [n1.step(40.0) for _ in range(1000)]
        t2 = [n2.step(40.0) for _ in range(1000)]
        assert t1 != t2

    def test_explicit_seed_reset_replays_the_complete_stream(self):
        first = EscapeRateNeuron(seed=0x1234)
        expected = [first.step(40.0) for _ in range(4096)]
        expected_state = (first.v, first.rng_state)
        first.reset()
        actual = [first.step(40.0) for _ in range(4096)]
        assert actual == expected
        assert (first.v, first.rng_state) == expected_state

    def test_zero_seed_uses_the_documented_nonzero_hardware_fallback(self):
        neuron = EscapeRateNeuron(seed=0)
        assert neuron.initial_seed == neuron.rng_state == DEFAULT_LFSR16_SEED

    def test_default_seed_matches_schema_and_hardware_reproducibility(self):
        first = EscapeRateNeuron()
        second = EscapeRateNeuron()
        assert first.initial_seed == second.initial_seed == DEFAULT_LFSR16_SEED
        assert [first.step(40.0) for _ in range(1024)] == [second.step(40.0) for _ in range(1024)]

    @pytest.mark.parametrize("seed", [True, -1, 0x10000, 1.5])
    def test_invalid_explicit_seed_is_rejected(self, seed: object):
        with pytest.raises(ValueError, match="seed"):
            EscapeRateNeuron(seed=cast(int | None, seed))

    def test_trial_sample_is_exactly_eight_primitive_advances(self):
        expected = DEFAULT_LFSR16_SEED
        for _ in range(LFSR16_ADVANCES_PER_TRIAL):
            expected = lfsr16_advance(expected)
        assert lfsr16_trial_sample(DEFAULT_LFSR16_SEED) == expected

    def test_probability_threshold_endpoints_and_interior_quantisation(self):
        assert probability_to_lfsr16_threshold(0.0) == 0
        assert probability_to_lfsr16_threshold(1.0) == 0x10000
        assert probability_to_lfsr16_threshold(0.25) == 16_384
        sampler = Lfsr16Threshold(DEFAULT_LFSR16_SEED)
        before = sampler.state
        sampler.trial(0.0)
        assert sampler.state != before

    def test_rate_increases_with_input(self):
        n_low = EscapeRateNeuron()
        n_high = EscapeRateNeuron()
        s_low = sum(n_low.step(20.0) for _ in range(50000))
        s_high = sum(n_high.step(40.0) for _ in range(50000))
        assert s_high > s_low

    def test_zero_input_silent(self):
        """V_ss = -70, far below theta=-50 → P(spike) ≈ 0."""
        n = EscapeRateNeuron()
        spikes = sum(n.step(0.0) for _ in range(50000))
        assert spikes == 0

    def test_escape_probability_uses_bounded_hazard_transform(self):
        """Finite-step escape probability is 1 - exp(-rho(V) dt), not clipped rho dt."""
        n = EscapeRateNeuron(v=-50.0, v_threshold=-50.0, rho_0=0.2, dt=2.0)
        expected = 1.0 - math.exp(-0.4)
        assert n._spike_probability(n.v_threshold) == pytest.approx(expected)

    def test_high_escape_rate_saturates_without_invalid_probability(self):
        n = EscapeRateNeuron(v=1000.0)
        assert n.step(0.0) == 1
        assert n.v == n.v_reset


class TestEscapeRateAnalytical:
    def test_v_steady_state(self):
        """V_ss = V_rest + R·I. Mean V should be near V_ss."""
        n = EscapeRateNeuron()
        for _ in range(10000):
            n.step(20.0)
        vs = []
        for _ in range(10000):
            n.step(20.0)
            vs.append(n.v)
        mean_v = np.mean(vs)
        v_ss = n.v_rest + n.resistance * 20.0
        assert abs(mean_v - v_ss) < 5.0

    def test_membrane_equation_one_step(self):
        """V_next = V_inf + (V - V_inf) * exp(-dt / tau_m)."""
        n = EscapeRateNeuron(seed=999)
        v0 = n.v
        I = 15.0
        v_inf = n.v_rest + n.resistance * I
        expected = v_inf + (v0 - v_inf) * math.exp(-n.dt / n.tau_m)
        n.step(I)
        if n.v != n.v_reset:
            assert abs(n.v - expected) < 1e-10

    def test_membrane_exact_flow_separates_from_forward_euler(self):
        n = EscapeRateNeuron(v=-65.0, dt=5.0, rho_0=1.0e-12, seed=999)
        v0 = n.v
        current = 10.0
        v_inf = n.v_rest + n.resistance * current
        euler = v0 + (-(v0 - n.v_rest) + n.resistance * current) / n.tau_m * n.dt
        expected = v_inf + (v0 - v_inf) * math.exp(-n.dt / n.tau_m)
        spike = n.step(current)
        assert spike == 0
        assert abs(n.v - expected) < 1e-10
        assert abs(n.v - euler) > 1e-3

    def test_rho0_scales_rate(self):
        n_low = EscapeRateNeuron(rho_0=0.0001)
        n_high = EscapeRateNeuron(rho_0=0.01)
        s_low = sum(n_low.step(30.0) for _ in range(50000))
        s_high = sum(n_high.step(30.0) for _ in range(50000))
        assert s_high > s_low

    def test_delta_u_controls_sensitivity(self):
        n_narrow = EscapeRateNeuron(delta_u=1.5)
        n_wide = EscapeRateNeuron(delta_u=6.0)
        s_narrow = sum(n_narrow.step(30.0) for _ in range(50000))
        s_wide = sum(n_wide.step(30.0) for _ in range(50000))
        assert s_narrow != s_wide


class TestEscapeRateISI:
    def test_isi_variability(self):
        """Stochastic → CV(ISI) > 0."""
        n = EscapeRateNeuron()
        spikes = _run(n, current=40.0, steps=100000)
        if len(spikes) >= 50:
            isis = np.diff(spikes).astype(float)
            cv = np.std(isis) / np.mean(isis)
            assert cv > 0.1

    def test_higher_current_shorter_isi(self):
        n30 = EscapeRateNeuron()
        n40 = EscapeRateNeuron()
        s30 = _run(n30, current=30.0, steps=50000)
        s40 = _run(n40, current=40.0, steps=50000)
        if len(s30) > 10 and len(s40) > 10:
            assert np.mean(np.diff(s40)) < np.mean(np.diff(s30))


class TestEscapeRateParameters:
    def test_tau_m_controls_v_dynamics(self):
        n_fast = EscapeRateNeuron(tau_m=2.0)
        n_slow = EscapeRateNeuron(tau_m=50.0)
        n_fast.step(20.0)
        n_slow.step(20.0)
        assert abs(n_fast.v - (-70.0)) > abs(n_slow.v - (-70.0))

    def test_resistance_scales_input(self):
        n_low = EscapeRateNeuron(resistance=0.5)
        n_high = EscapeRateNeuron(resistance=2.0)
        n_low.step(20.0)
        n_high.step(20.0)
        assert abs(n_high.v - (-70.0)) > abs(n_low.v - (-70.0))


class TestEscapeRateValidation:
    @pytest.mark.parametrize("field", ["v", "v_rest", "v_reset", "v_threshold"])
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_voltage_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            EscapeRateNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["tau_m", "rho_0", "delta_u", "resistance", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_scale_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            EscapeRateNeuron(**{field: value})

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_voltage_mutation(self, current: float):
        n = EscapeRateNeuron(v=-65.0)
        before = n.v
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert n.v == before

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("tau_m", 0.0, "tau_m"),
            ("rho_0", -1.0, "rho_0"),
            ("delta_u", 0.0, "delta_u"),
            ("resistance", np.nan, "resistance"),
            ("dt", np.inf, "dt"),
            ("v_rest", np.nan, "v_rest"),
        ],
    )
    def test_rejects_corrupted_runtime_state_before_voltage_mutation(
        self, field: str, value: float, message: str
    ):
        n = EscapeRateNeuron(v=-65.0)
        setattr(n, field, value)
        before = -65.0
        with pytest.raises(ValueError, match=message):
            n.step(1.0)
        assert n.v == before

    def test_rejects_non_finite_voltage_candidate_before_reset_mutation(self):
        n = EscapeRateNeuron(v=-65.0, v_threshold=1.0e308, resistance=1.0e308)
        before = n.v
        with pytest.raises(ValueError, match="voltage candidate"):
            n.step(1.0e308)
        assert n.v == before

    def test_rejects_non_finite_hazard_before_random_draw(self):
        n = EscapeRateNeuron(v=-50.0, rho_0=1.0e308, dt=10.0, seed=42)
        before = (n.v, n.rng_state)
        with pytest.raises(ValueError, match="escape hazard"):
            n.step(20.0)
        assert (n.v, n.rng_state) == before

    def test_python_batch_failure_restores_voltage_and_rng(self, monkeypatch):
        n = EscapeRateNeuron(v=-65.0, seed=42)
        before = (n.v, n.rng_state)
        original_step = EscapeRateNeuron.step
        calls = 0

        def fail_after_one_step(neuron: EscapeRateNeuron, current: float) -> int:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise FloatingPointError("injected batch failure")
            return original_step(neuron, current)

        monkeypatch.setattr(EscapeRateNeuron, "step", fail_after_one_step)
        with pytest.raises(FloatingPointError, match="injected batch failure"):
            n.simulate(10, current=20.0, backend="python")
        assert (n.v, n.rng_state) == before

    def test_malformed_native_result_is_rejected_before_state_commit(self, monkeypatch):
        from sc_neurocore.accel import escape_rate as backends

        n = EscapeRateNeuron(v=-65.0, seed=42)
        before = (n.v, n.rng_state)
        monkeypatch.setattr(backends, "_HAS_RUST", True)
        monkeypatch.setattr(
            backends,
            "simulate_rust",
            lambda *_args: (np.array([123.0]), np.array([1.5]), 123.0, 1.5),
        )
        with pytest.raises(FloatingPointError, match="non-binary events"):
            n.simulate(1, current=20.0, backend="rust")
        assert (n.v, n.rng_state) == before


class TestEscapeRatePerformance:
    def test_isolation_throughput(self):
        n = EscapeRateNeuron()
        for _ in range(1000):
            n.step(30.0)
        N = 50000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(30.0)
        elapsed = time.perf_counter() - t0
        # Coarse regression smoke only: CI runs this concurrently under xdist;
        # controlled performance claims live in bench_model_escape_rate.py.
        assert N / elapsed > 10000

    def test_network_throughput(self):
        pop = Population(EscapeRateNeuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 50 * 500 / elapsed > 5000


class TestEscapeRatePipeline:
    def test_population(self):
        assert Population(EscapeRateNeuron, n=10, label="esc").n == 10

    def test_network_spikes(self):
        pop = Population(EscapeRateNeuron, n=20, label="esc")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        """Projection accepted by Network — source fires, graph valid."""
        src = Population(EscapeRateNeuron, n=10, label="src")
        tgt = Population(EscapeRateNeuron, n=10, label="tgt")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=50.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_analysis_pipeline(self):
        n = EscapeRateNeuron()
        train = np.array([float(n.step(40.0)) for _ in range(50000)])
        sc = spike_count(train)
        assert sc >= 50
        isis = isi(train, dt=0.001)
        assert len(isis) >= 10
        rate = firing_rate(train, dt=0.001)
        assert rate > 0
        duration = 50000 * 0.001
        assert abs(rate - sc / duration) < 100.0
