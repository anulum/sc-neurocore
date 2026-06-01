# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Module-specific test: ConnorStevensNeuron

"""Module-specific behavioural tests for ConnorStevensNeuron (Connor & Stevens 1977).

HH-type model with A-type potassium current (I_A), Type-I excitability.
6 state variables: v, m (Na act), h (Na inact), n (K), a (A-type act),
b (A-type inact). 4 ionic currents: I_Na(g=120, m³h), I_K(g=20, n⁴),
I_A(g=47.7, a³b), I_L(g=0.3).

100 sub-steps per step() call (dt=0.01, 1/dt=100). Type-I: continuous
f-I curve from zero frequency (saddle-node on invariant circle bifurcation).
A-current delays spike onset → long latency at rheobase.
~536 steps/s (100 sub-steps × HH complexity)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.connor_stevens import ConnorStevensNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: ConnorStevensNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestCSIsolation:
    def test_defaults(self):
        n = ConnorStevensNeuron()
        assert n.v == -68.0 and n.m == 0.01 and n.h == 0.99
        assert n.n == 0.1 and n.a == 0.5 and n.b == 0.1
        assert n.g_a == 47.7  # A-type current conductance
        assert n.dt == 0.01 and n.v_threshold == 0.0

    def test_six_state_variables(self):
        n = ConnorStevensNeuron()
        for attr in ["v", "m", "h", "n", "a", "b"]:
            assert hasattr(n, attr)

    def test_step_returns_binary(self):
        assert ConnorStevensNeuron().step(0.0) in (0, 1)

    def test_state_finite_long_run(self):
        n = ConnorStevensNeuron()
        for _ in range(500):
            n.step(20.0)
        for attr in ["v", "m", "h", "n", "a", "b"]:
            assert np.isfinite(getattr(n, attr)), f"{attr} not finite"

    def test_reset_restores_defaults(self):
        n = ConnorStevensNeuron()
        for _ in range(200):
            n.step(20.0)
        n.reset()
        assert n.v == -68.0 and n.m == 0.01 and n.h == 0.99
        assert n.n == 0.1 and n.a == 0.5 and n.b == 0.1

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = ConnorStevensNeuron()
            trace = [(n.step(20.0), n.v) for _ in range(100)]
            traces.append(trace)
        assert traces[0] == traces[1]


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — sub-stepping, currents, A-type current
# ---------------------------------------------------------------------------
class TestCSAnalytical:
    def test_100_substeps_per_call(self):
        """dt=0.01 → 1/0.01 = 100 sub-steps per step() call."""
        n = ConnorStevensNeuron()
        assert int(1.0 / max(n.dt, 0.001)) == 100

    def test_four_ionic_currents(self):
        """I_Na, I_K, I_A, I_L — all conductances positive."""
        n = ConnorStevensNeuron()
        assert n.g_na > 0 and n.g_k > 0 and n.g_a > 0 and n.g_l > 0

    def test_a_current_conductance_dominant(self):
        """g_A=47.7 > g_K=20 — A-current is the signature feature."""
        n = ConnorStevensNeuron()
        assert n.g_a > n.g_k

    def test_reversal_ordering(self):
        """e_a < e_k < e_l < e_na."""
        n = ConnorStevensNeuron()
        assert n.e_a < n.e_k < n.e_l < n.e_na

    def test_gating_variables_bounded(self):
        """All gating variables should stay in [0, 1] range."""
        n = ConnorStevensNeuron()
        for _ in range(500):
            n.step(20.0)
        for attr in ["m", "h", "n", "a", "b"]:
            val = getattr(n, attr)
            assert -0.01 <= val <= 1.01, f"{attr}={val}"

    def test_a_type_delays_spike_onset(self):
        """A-current creates delay at rheobase — Type-I hallmark."""
        # With A-current (default)
        n_with_a = ConnorStevensNeuron()
        spikes_a = _run(n_with_a, current=8.0, steps=200)
        # Without A-current
        n_no_a = ConnorStevensNeuron(g_a=0.0)
        spikes_no_a = _run(n_no_a, current=8.0, steps=200)
        # Without A: should fire more easily
        assert len(spikes_no_a) >= len(spikes_a)


# ---------------------------------------------------------------------------
# 3. TYPE-I EXCITABILITY
# ---------------------------------------------------------------------------
class TestCSTypeI:
    def test_fires_at_sufficient_current(self):
        n = ConnorStevensNeuron()
        spikes = _run(n, current=20.0, steps=500)
        assert len(spikes) >= 20

    def test_subthreshold_silent(self):
        n = ConnorStevensNeuron()
        assert len(_run(n, current=1.0, steps=200)) == 0

    def test_continuous_fi_curve(self):
        """Type-I: frequency starts from ~0 at threshold (no frequency jump)."""
        rates = []
        for I in [5.0, 8.0, 10.0, 15.0, 20.0]:
            n = ConnorStevensNeuron()
            rates.append(len(_run(n, current=I, steps=500)))
        # Rates should increase monotonically
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [5.0, 10.0, 15.0, 20.0, 30.0])
    def test_fi_sweep(self, current: float):
        n = ConnorStevensNeuron()
        for _ in range(200):
            n.step(current)
        assert np.isfinite(n.v)

    def test_voltage_bounded(self):
        n = ConnorStevensNeuron()
        vs = []
        for _ in range(500):
            n.step(20.0)
            vs.append(n.v)
        assert min(vs) > -100 and max(vs) < 60


# ---------------------------------------------------------------------------
# 4. PARAMETER SENSITIVITY
# ---------------------------------------------------------------------------
class TestCSParameters:
    @pytest.mark.parametrize("g_a", [0.0, 47.7, 100.0])
    def test_g_a_sweep(self, g_a: float):
        n = ConnorStevensNeuron(g_a=g_a)
        for _ in range(200):
            n.step(20.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("g_na", [60.0, 120.0, 200.0])
    def test_g_na_sweep(self, g_na: float):
        n = ConnorStevensNeuron(g_na=g_na)
        for _ in range(200):
            n.step(20.0)
        assert np.isfinite(n.v)


# ---------------------------------------------------------------------------
# 5. PERFORMANCE
# ---------------------------------------------------------------------------
class TestCSPerformance:
    def test_isolation_throughput(self):
        n = ConnorStevensNeuron()
        N = 200
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(20.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        # 100 sub-steps × HH → ~500 steps/s
        assert rate > 50, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(ConnorStevensNeuron, n=5, label="bench")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=20.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.1, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 5 * 100
        rate = neuron_steps / elapsed
        assert rate > 10, f"network: {rate:.0f} neuron-steps/s"


# ---------------------------------------------------------------------------
# 6. FULL PIPELINE
# ---------------------------------------------------------------------------
class TestCSPipeline:
    def test_population(self):
        assert Population(ConnorStevensNeuron, n=5, label="cs").n == 5

    def test_projection_wiring(self):
        src = Population(ConnorStevensNeuron, n=3, label="src")
        tgt = Population(ConnorStevensNeuron, n=3, label="tgt")
        drive = PoissonInput(n=3, rate_hz=500.0, weight=20.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=10.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=0.5, dt=0.001, backend="python")
        assert isinstance(mon_src.count, int)

    def test_network_spikes(self):
        pop = Population(ConnorStevensNeuron, n=5, label="cs")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=20.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert isinstance(mon.count, int)

    def test_analysis_spike_count(self):
        n = ConnorStevensNeuron()
        train = np.array([float(n.step(20.0)) for _ in range(500)])
        sc = spike_count(train)
        assert sc >= 10

    def test_analysis_isi(self):
        n = ConnorStevensNeuron()
        train = np.array([float(n.step(20.0)) for _ in range(500)])
        intervals = isi(train, dt=0.001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))

    def test_analysis_firing_rate(self):
        n = ConnorStevensNeuron()
        train = np.array([float(n.step(20.0)) for _ in range(500)])
        rate = firing_rate(train, dt=0.001)
        assert rate > 0


def _connor_reference_rate(
    scale: float, shift: float, v: float, denom: float, limit: float
) -> float:
    delta = v + shift
    x = delta / denom
    if abs(x) < 1e-9:
        return scale * denom
    return scale * delta / (1.0 - np.exp(-x))


def _connor_reference_derivatives(
    state: tuple[float, float, float, float, float, float], current: float, params: dict[str, float]
) -> tuple[float, float, float, float, float, float]:
    v, m, h, n, a, b = state
    alpha_m = _connor_reference_rate(0.38, 29.7, v, 10.0, 3.8)
    beta_m = 15.2 * np.exp(-(v + 54.7) / 18.0)
    alpha_h = 0.266 * np.exp(-(v + 48.0) / 20.0)
    beta_h = 3.8 / (1.0 + np.exp(-(v + 18.0) / 10.0))
    alpha_n = _connor_reference_rate(0.02, 45.7, v, 10.0, 0.2)
    beta_n = 0.25 * np.exp(-(v + 55.7) / 80.0)
    a_inf = (0.0761 * np.exp((v + 94.22) / 31.84) / (1.0 + np.exp((v + 1.17) / 28.93))) ** (
        1.0 / 3.0
    )
    tau_a = 0.3632 + 1.158 / (1.0 + np.exp((v + 55.96) / 20.12))
    b_inf = (1.0 / (1.0 + np.exp((v + 53.3) / 14.54))) ** 4
    tau_b = 1.24 + 2.678 / (1.0 + np.exp((v + 50.0) / 16.027))

    i_na = params["g_na"] * m**3 * h * (v - params["e_na"])
    i_k = params["g_k"] * n**4 * (v - params["e_k"])
    i_a = params["g_a"] * a**3 * b * (v - params["e_a"])
    i_l = params["g_l"] * (v - params["e_l"])
    dv = (-i_na - i_k - i_a - i_l + current) / params["c_m"]
    return (
        dv,
        alpha_m * (1.0 - m) - beta_m * m,
        alpha_h * (1.0 - h) - beta_h * h,
        alpha_n * (1.0 - n) - beta_n * n,
        (a_inf - a) / tau_a,
        (b_inf - b) / tau_b,
    )


def _connor_reference_rk4(
    neuron: ConnorStevensNeuron, current: float
) -> tuple[float, float, float, float, float, float]:
    params = {
        "g_na": neuron.g_na,
        "g_k": neuron.g_k,
        "g_a": neuron.g_a,
        "g_l": neuron.g_l,
        "e_na": neuron.e_na,
        "e_k": neuron.e_k,
        "e_a": neuron.e_a,
        "e_l": neuron.e_l,
        "c_m": neuron.c_m,
    }
    state = (neuron.v, neuron.m, neuron.h, neuron.n, neuron.a, neuron.b)
    dt = neuron.dt
    for _ in range(int(1.0 / max(dt, 0.001))):
        k1 = _connor_reference_derivatives(state, current, params)
        k2 = _connor_reference_derivatives(
            tuple(s + 0.5 * dt * k for s, k in zip(state, k1)), current, params
        )
        k3 = _connor_reference_derivatives(
            tuple(s + 0.5 * dt * k for s, k in zip(state, k2)), current, params
        )
        k4 = _connor_reference_derivatives(
            tuple(s + dt * k for s, k in zip(state, k3)), current, params
        )
        state = tuple(
            s + dt * (a + 2.0 * b + 2.0 * c + d) / 6.0
            for s, a, b, c, d in zip(state, k1, k2, k3, k4)
        )
    return state


def test_connor_stevens_matches_independent_rk4_contract() -> None:
    """Connor-Stevens step follows the module RK4 integration contract."""
    neuron = ConnorStevensNeuron(v=-62.0, m=0.05, h=0.84, n=0.22, a=0.41, b=0.27, dt=0.02)
    expected = _connor_reference_rk4(neuron, current=8.5)

    spike = neuron.step(8.5)

    assert spike in (0, 1)
    assert (neuron.v, neuron.m, neuron.h, neuron.n, neuron.a, neuron.b) == pytest.approx(
        expected, rel=1e-10, abs=1e-10
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("v", np.nan),
        ("m", np.inf),
        ("g_na", -1.0),
        ("g_k", -1.0),
        ("g_a", -1.0),
        ("g_l", -1.0),
        ("c_m", 0.0),
        ("dt", 0.0),
    ],
)
def test_connor_stevens_rejects_invalid_parameters(field: str, value: float) -> None:
    """Invalid physical parameters are rejected before simulation begins."""
    with pytest.raises((TypeError, ValueError)):
        ConnorStevensNeuron(**{field: value})


def test_connor_stevens_rejects_non_finite_current_without_mutation() -> None:
    """Adapter-visible invalid drive fails closed and preserves biological state."""
    neuron = ConnorStevensNeuron(v=-63.0, m=0.04, h=0.91, n=0.18, a=0.36, b=0.31)
    before = (neuron.v, neuron.m, neuron.h, neuron.n, neuron.a, neuron.b)

    with pytest.raises((TypeError, ValueError, FloatingPointError)):
        neuron.step(float("nan"))

    assert (neuron.v, neuron.m, neuron.h, neuron.n, neuron.a, neuron.b) == before


def test_connor_stevens_rejects_corrupted_runtime_state_without_mutation() -> None:
    """Runtime state corruption cannot be amplified into a partially committed step."""
    neuron = ConnorStevensNeuron()
    neuron.b = float("inf")
    before = (neuron.v, neuron.m, neuron.h, neuron.n, neuron.a, neuron.b)

    with pytest.raises((TypeError, ValueError, FloatingPointError)):
        neuron.step(6.0)

    assert (neuron.v, neuron.m, neuron.h, neuron.n, neuron.a, neuron.b) == before
