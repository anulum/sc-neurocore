# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AlphaNeuron module contract tests

"""Module-specific pipeline and numerical tests for AlphaNeuron (Rall 1967).

Dual excitatory/inhibitory alpha-synapse currents. step(exc_current, inh_current).
Inhibition suppresses excitatory drive. Benchmark evidence is recorded in
benchmarks/results/local_python_2026-06-01_alpha.json."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.alpha import AlphaNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate


def _run(neuron: AlphaNeuron, exc: float, steps: int, inh: float = 0.0) -> list[int]:
    return [t for t in range(steps) if neuron.step(exc, inh) == 1]


def _drive_contribution(
    current_delta: float, rise_delta: float, tau_drive: float, tau_v: float, dt: float
) -> float:
    rate_v = 1.0 / tau_v
    rate_drive = 1.0 / tau_drive
    decay_v = math.exp(-dt / tau_v)
    decay_drive = math.exp(-dt / tau_drive)
    if math.isclose(rate_v, rate_drive, rel_tol=0.0, abs_tol=1.0e-14):
        return rate_v * decay_v * (
            current_delta * dt + rise_delta * dt * dt / (2.0 * tau_drive)
        )
    rate_delta = rate_v - rate_drive
    first_order = current_delta * (decay_drive - decay_v) / rate_delta
    second_order = (
        rise_delta
        / tau_drive
        * (decay_drive * (rate_delta * dt - 1.0) + decay_v)
        / (rate_delta * rate_delta)
    )
    return rate_v * (first_order + second_order)


def _exact_alpha_reference(
    neuron: AlphaNeuron, exc_current: float, inh_current: float
) -> tuple[int, float, float, float, float, float]:
    a_exc_ss = neuron.tau_exc * exc_current
    a_inh_ss = neuron.tau_inh * inh_current
    a_exc_delta = neuron.a_exc - a_exc_ss
    a_inh_delta = neuron.a_inh - a_inh_ss
    i_exc_delta = neuron.i_exc - a_exc_ss
    i_inh_delta = neuron.i_inh - a_inh_ss

    decay_exc = math.exp(-neuron.dt / neuron.tau_exc)
    decay_inh = math.exp(-neuron.dt / neuron.tau_inh)
    a_exc_next = a_exc_ss + a_exc_delta * decay_exc
    a_inh_next = a_inh_ss + a_inh_delta * decay_inh
    i_exc_next = a_exc_ss + decay_exc * (
        i_exc_delta + a_exc_delta * neuron.dt / neuron.tau_exc
    )
    i_inh_next = a_inh_ss + decay_inh * (
        i_inh_delta + a_inh_delta * neuron.dt / neuron.tau_inh
    )

    v_steady = neuron.v_rest + a_exc_ss - a_inh_ss
    v_next = (
        v_steady
        + (neuron.v - v_steady) * math.exp(-neuron.dt / neuron.tau_v)
        + _drive_contribution(
            i_exc_delta, a_exc_delta, neuron.tau_exc, neuron.tau_v, neuron.dt
        )
        - _drive_contribution(
            i_inh_delta, a_inh_delta, neuron.tau_inh, neuron.tau_v, neuron.dt
        )
    )
    if v_next >= neuron.v_threshold:
        return 1, neuron.v_rest, a_exc_next, i_exc_next, a_inh_next, i_inh_next
    return 0, v_next, a_exc_next, i_exc_next, a_inh_next, i_inh_next


class TestAlphaIsolation:
    def test_defaults(self):
        n = AlphaNeuron()
        assert n.v == 0.0 and n.a_exc == 0.0 and n.i_exc == 0.0
        assert n.a_inh == 0.0 and n.i_inh == 0.0
        assert n.tau_v == 20.0 and n.tau_exc == 5.0 and n.tau_inh == 10.0

    def test_step_returns_binary(self):
        assert AlphaNeuron().step(0.0) in (0, 1)

    def test_dual_input_signature(self):
        n = AlphaNeuron()
        s = n.step(1.0, 0.5)
        assert s in (0, 1)

    def test_three_variables_evolve(self):
        n = AlphaNeuron()
        for _ in range(100):
            n.step(1.0, 0.3)
        assert n.v != 0.0 and n.a_exc != 0.0 and n.i_exc != 0.0
        assert n.a_inh != 0.0 and n.i_inh != 0.0

    def test_state_finite(self):
        n = AlphaNeuron()
        for _ in range(50000):
            n.step(1.0, 0.3)
        assert all(np.isfinite(v) for v in [n.v, n.a_exc, n.i_exc, n.a_inh, n.i_inh])

    def test_reset(self):
        n = AlphaNeuron()
        for _ in range(100):
            n.step(2.0)
        n.reset()
        assert n.v == n.v_rest and n.a_exc == 0.0 and n.i_exc == 0.0
        assert n.a_inh == 0.0 and n.i_inh == 0.0


class TestAlphaSynapticCurrents:
    def test_exc_charges_i_exc(self):
        n = AlphaNeuron(v_threshold=100.0)
        n.step(1.0, 0.0)
        assert n.a_exc > n.i_exc > 0.0 and n.a_inh == 0.0 and n.i_inh == 0.0

    def test_inh_charges_i_inh(self):
        n = AlphaNeuron(v_threshold=100.0)
        n.step(0.0, 1.0)
        assert n.a_inh > n.i_inh > 0.0 and n.a_exc == 0.0 and n.i_exc == 0.0

    def test_exc_drives_v_up(self):
        n = AlphaNeuron(v_threshold=100.0)
        for _ in range(100):
            n.step(1.0, 0.0)
        assert n.v > 0.0

    def test_inh_drives_v_down(self):
        """Inhibition opposes excitation: net V = i_exc - i_inh."""
        n = AlphaNeuron(v_threshold=100.0)
        for _ in range(100):
            n.step(0.0, 1.0)
        assert n.v < 0.0

    def test_inhibition_suppresses_spiking(self):
        """Sustained inhibition prevents spikes even with high excitation."""
        n_exc = AlphaNeuron()
        n_bal = AlphaNeuron()
        s_exc = len(_run(n_exc, exc=2.0, steps=5000))
        s_bal = len(_run(n_bal, exc=2.0, steps=5000, inh=2.0))
        assert s_exc > s_bal, f"Exc only: {s_exc}, balanced: {s_bal}"

    def test_i_exc_decays_with_tau_exc(self):
        """i_exc decays with tau_exc when input is removed."""
        n = AlphaNeuron(v_threshold=100.0)
        for _ in range(100):
            n.step(1.0)
        i_exc_charged = n.i_exc
        n.step(0.0)  # remove input
        assert n.i_exc < i_exc_charged  # decayed

    def test_alpha_function_dynamics(self):
        """Constant excitatory drive follows the exact alpha-filter cascade."""
        n = AlphaNeuron(v_threshold=100.0)
        I = 1.0
        n.step(I)
        expected = n.tau_exc * I * (
            1.0 - math.exp(-n.dt / n.tau_exc) * (1.0 + n.dt / n.tau_exc)
        )
        assert abs(n.i_exc - expected) < 1e-12

    def test_exact_linear_flow_matches_closed_form(self):
        n = AlphaNeuron(
            v=0.3,
            a_exc=0.9,
            i_exc=0.7,
            a_inh=0.25,
            i_inh=0.2,
            v_threshold=100.0,
            dt=0.75,
        )
        expected = _exact_alpha_reference(n, exc_current=0.8, inh_current=0.1)

        got = n.step(0.8, 0.1)

        assert got == expected[0]
        assert n.v == pytest.approx(expected[1], abs=1e-12)
        assert n.a_exc == pytest.approx(expected[2], abs=1e-12)
        assert n.i_exc == pytest.approx(expected[3], abs=1e-12)
        assert n.a_inh == pytest.approx(expected[4], abs=1e-12)
        assert n.i_inh == pytest.approx(expected[5], abs=1e-12)

    def test_equal_tau_linear_flow_uses_exact_limit(self):
        n = AlphaNeuron(
            v=0.25,
            a_exc=0.75,
            i_exc=0.5,
            a_inh=0.2,
            i_inh=0.125,
            v_threshold=100.0,
            tau_v=5.0,
            tau_exc=5.0,
            tau_inh=7.0,
            dt=3.0,
        )
        expected = _exact_alpha_reference(n, exc_current=0.4, inh_current=0.05)

        n.step(0.4, 0.05)

        assert n.v == pytest.approx(expected[1], abs=1e-12)
        assert n.a_exc == pytest.approx(expected[2], abs=1e-12)
        assert n.i_exc == pytest.approx(expected[3], abs=1e-12)
        assert n.a_inh == pytest.approx(expected[4], abs=1e-12)
        assert n.i_inh == pytest.approx(expected[5], abs=1e-12)

    def test_large_timestep_decays_without_euler_overshoot(self):
        n = AlphaNeuron(
            v=2.0,
            a_exc=4.0,
            i_exc=2.0,
            a_inh=0.0,
            i_inh=0.0,
            v_threshold=100.0,
            tau_v=5.0,
            tau_exc=5.0,
            tau_inh=5.0,
            dt=50.0,
        )

        n.step(0.0, 0.0)

        assert 0.0 <= n.a_exc <= 4.0
        assert 0.0 <= n.i_exc <= 2.0
        assert n.a_inh == 0.0
        assert n.i_inh == 0.0
        assert 0.0 <= n.v <= 2.0


class TestAlphaFI:
    def test_zero_silent(self):
        n = AlphaNeuron()
        assert len(_run(n, exc=0.0, steps=5000)) == 0

    def test_monotonic_fi(self):
        rates = []
        for I in [0.5, 1.0, 2.0, 5.0]:
            n = AlphaNeuron()
            rates.append(len(_run(n, exc=I, steps=5000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))

    def test_suprathreshold_fires(self):
        n = AlphaNeuron()
        assert len(_run(n, exc=2.0, steps=5000)) >= 100


class TestAlphaParameters:
    def test_tau_exc_affects_integration(self):
        n_fast = AlphaNeuron(tau_exc=2.0)
        n_slow = AlphaNeuron(tau_exc=20.0)
        s_fast = len(_run(n_fast, exc=1.0, steps=5000))
        s_slow = len(_run(n_slow, exc=1.0, steps=5000))
        assert s_fast != s_slow

    @pytest.mark.parametrize("dt", [0.5, 1.0, 2.0])
    def test_dt_stability(self, dt: float):
        n = AlphaNeuron(dt=dt)
        for _ in range(5000):
            n.step(1.0)
        assert np.isfinite(n.v)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = AlphaNeuron()
            trace = [(n.step(1.0, 0.3), n.v, n.i_exc, n.i_inh) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestAlphaValidation:
    @pytest.mark.parametrize(
        "field", ["v", "a_exc", "i_exc", "a_inh", "i_inh", "v_rest", "v_threshold"]
    )
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_state_and_voltage_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            AlphaNeuron(**{field: value})

    @pytest.mark.parametrize("field", ["tau_v", "tau_exc", "tau_inh", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_time_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            AlphaNeuron(**{field: value})

    @pytest.mark.parametrize(
        ("exc_current", "inh_current"), [(np.nan, 0.0), (np.inf, 0.0), (0.0, -np.inf)]
    )
    def test_rejects_non_finite_currents_before_state_mutation(
        self, exc_current: float, inh_current: float
    ):
        n = AlphaNeuron(v=0.25, a_exc=0.6, i_exc=0.5, a_inh=0.2, i_inh=0.125)
        before = (n.v, n.a_exc, n.i_exc, n.a_inh, n.i_inh)
        with pytest.raises(ValueError, match="current"):
            n.step(exc_current, inh_current)
        assert (n.v, n.a_exc, n.i_exc, n.a_inh, n.i_inh) == before

    def test_rejects_corrupted_runtime_parameter_before_state_mutation(self):
        n = AlphaNeuron(v=0.25, a_exc=0.6, i_exc=0.5, a_inh=0.2, i_inh=0.125)
        before = (n.v, n.a_exc, n.i_exc, n.a_inh, n.i_inh)
        n.tau_v = 0.0
        with pytest.raises(ValueError, match="tau_v"):
            n.step(1.0, 0.5)
        assert (n.v, n.a_exc, n.i_exc, n.a_inh, n.i_inh) == before

    def test_rejects_non_finite_candidate_before_state_mutation(self):
        n = AlphaNeuron(v=0.25, a_exc=0.6, i_exc=0.5, a_inh=0.2, i_inh=0.125)
        before = (n.v, n.a_exc, n.i_exc, n.a_inh, n.i_inh)
        with pytest.raises(ValueError, match="exact-flow"):
            n.step(1.0e308, 0.0)
        assert (n.v, n.a_exc, n.i_exc, n.a_inh, n.i_inh) == before


class TestAlphaPerformance:
    def test_isolation_throughput(self):
        n = AlphaNeuron()
        N = 50000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(1.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 50000

    def test_network_throughput(self):
        pop = Population(AlphaNeuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 50 * 500 / elapsed > 5000


class TestAlphaPipeline:
    def test_population(self):
        assert Population(AlphaNeuron, n=10, label="alpha").n == 10

    def test_network_spikes(self):
        pop = Population(AlphaNeuron, n=10, label="alpha")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(AlphaNeuron, n=10, label="src")
        tgt = Population(AlphaNeuron, n=10, label="tgt")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=1.0, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self):
        n = AlphaNeuron()
        train = np.array([float(n.step(1.0)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 50
        rate = firing_rate(train, dt=0.001)
        assert rate > 0
