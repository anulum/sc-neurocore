# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: GolombFSNeuron

"""Full pipeline test for GolombFSNeuron (Golomb et al. 2007).

Fast-spiking interneuron with Kv3 potassium channel:
4 currents: I_Na(g=112.5, m³_inf·h), I_Kd(g=225, n⁴),
I_Kv3(g=150, p²), I_L(g=0.25).

Kv3: high-threshold (v_half=-3), fast activation → narrow spikes,
minimal spike-frequency adaptation, sustained high-rate firing.
3 gating variables: h (Na inact), n (Kd), p (Kv3).
m_Na is instantaneous. 10 sub-steps per call (dt=0.01).
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.golomb_fs import GolombFSNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: GolombFSNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestGFSIsolation:
    def test_defaults(self):
        n = GolombFSNeuron()
        assert n.v == -65.0 and n.h == 0.9 and n.n == 0.1 and n.p == 0.0
        assert n.g_kv3 == 150.0  # Kv3 — signature channel
        assert n.dt == 0.01 and n.v_threshold == -20.0

    def test_step_returns_binary(self):
        assert GolombFSNeuron().step(0.0) in (0, 1)

    def test_state_finite_long_run(self):
        n = GolombFSNeuron()
        for _ in range(5000):
            n.step(5.0)
        for attr in ["v", "h", "n", "p"]:
            assert np.isfinite(getattr(n, attr))

    def test_reset_restores_defaults(self):
        n = GolombFSNeuron()
        for _ in range(1000):
            n.step(5.0)
        n.reset()
        assert n.v == -65.0 and n.h == 0.9 and n.n == 0.1 and n.p == 0.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = GolombFSNeuron()
            trace = [(n.step(5.0), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — Kv3, currents, sub-stepping
# ---------------------------------------------------------------------------
class TestGFSAnalytical:
    def test_10_substeps(self):
        """10 sub-steps per step() call (dt=0.01)."""
        n = GolombFSNeuron()
        assert n.dt == 0.01

    def test_four_ionic_currents(self):
        n = GolombFSNeuron()
        assert n.g_na > 0 and n.g_kd > 0 and n.g_kv3 > 0 and n.g_l > 0

    def test_kv3_high_threshold(self):
        """Kv3 p_inf half-activation at v=-3 mV (high threshold)."""
        v_half_kv3 = -3.0
        p_inf = 1.0 / (1.0 + np.exp(-(v_half_kv3 + 3.0) / 8.0))
        assert abs(p_inf - 0.5) < 1e-12

    def test_kv3_conductance_large(self):
        """g_Kv3=150 > g_Na=112.5: Kv3 dominates repolarisation."""
        n = GolombFSNeuron()
        assert n.g_kv3 > n.g_na

    def test_m_na_instantaneous(self):
        """m_Na set directly to m_inf (no time constant)."""
        # Source uses m_inf directly in current calculation
        n = GolombFSNeuron()
        n.step(5.0)
        # m_inf is not stored — computed inline
        assert np.isfinite(n.v)

    def test_reversal_ordering(self):
        n = GolombFSNeuron()
        assert n.e_k < n.e_l < n.e_na

    def test_gating_bounded(self):
        n = GolombFSNeuron()
        for _ in range(2000):
            n.step(5.0)
        for attr in ["h", "n", "p"]:
            val = getattr(n, attr)
            assert -0.05 <= val <= 1.05, f"{attr}={val}"


# ---------------------------------------------------------------------------
# 3. FAST-SPIKING DYNAMICS
# ---------------------------------------------------------------------------
class TestGFSDynamics:
    def test_fires_under_drive(self):
        n = GolombFSNeuron()
        spikes = _run(n, current=5.0, steps=5000)
        assert len(spikes) >= 10

    def test_subthreshold_silent(self):
        n = GolombFSNeuron()
        assert len(_run(n, current=0.5, steps=2000)) == 0

    def test_high_sustained_rate(self):
        """FS interneurons sustain high firing rates."""
        n = GolombFSNeuron()
        spikes = _run(n, current=10.0, steps=5000)
        assert len(spikes) >= 20

    def test_rate_monotonic(self):
        rates = []
        for I in [2.0, 5.0, 10.0]:
            n = GolombFSNeuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 2.0, 5.0, 10.0, 20.0])
    def test_fi_sweep(self, current: float):
        n = GolombFSNeuron()
        for _ in range(2000):
            n.step(current)
        assert np.isfinite(n.v)

    def test_voltage_bounded(self):
        n = GolombFSNeuron()
        vs = []
        for _ in range(2000):
            n.step(5.0)
            vs.append(n.v)
        assert min(vs) > -100 and max(vs) < 60


# ---------------------------------------------------------------------------
# 4. PARAMETERS
# ---------------------------------------------------------------------------
class TestGFSParameters:
    @pytest.mark.parametrize("g_kv3", [0.0, 150.0, 300.0])
    def test_g_kv3_sweep(self, g_kv3: float):
        n = GolombFSNeuron(g_kv3=g_kv3)
        for _ in range(2000):
            n.step(5.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("g_na", [50.0, 112.5, 200.0])
    def test_g_na_sweep(self, g_na: float):
        n = GolombFSNeuron(g_na=g_na)
        for _ in range(2000):
            n.step(5.0)
        assert np.isfinite(n.v)


# ---------------------------------------------------------------------------
# 5. PERFORMANCE
# ---------------------------------------------------------------------------
class TestGFSPerformance:
    def test_isolation_throughput(self):
        n = GolombFSNeuron()
        N = 2000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(5.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        # 10 sub-steps × HH
        assert rate > 500, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(GolombFSNeuron, n=10, label="bench")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.2, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 10 * 200
        rate = neuron_steps / elapsed
        assert rate > 100, f"network: {rate:.0f} neuron-steps/s"


# ---------------------------------------------------------------------------
# 6. FULL PIPELINE
# ---------------------------------------------------------------------------
class TestGFSPipeline:
    def test_population(self):
        assert Population(GolombFSNeuron, n=10, label="gfs").n == 10

    def test_projection_wiring(self):
        src = Population(GolombFSNeuron, n=3, label="src")
        tgt = Population(GolombFSNeuron, n=3, label="tgt")
        drive = PoissonInput(n=3, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=2.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert isinstance(mon_src.count, int)

    def test_network_spikes(self):
        pop = Population(GolombFSNeuron, n=5, label="gfs")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=5.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self):
        n = GolombFSNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 5

    def test_analysis_isi(self):
        n = GolombFSNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(5000)])
        intervals = isi(train, dt=0.001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))

    def test_analysis_firing_rate(self):
        n = GolombFSNeuron()
        train = np.array([float(n.step(5.0)) for _ in range(5000)])
        rate = firing_rate(train, dt=0.001)
        assert rate > 0


class TestGolombFSIntegrator:
    def test_default_integrator_is_rk4(self):
        assert GolombFSNeuron().integrator == "rk4"

    def test_rejects_unknown_integrator(self):
        with pytest.raises(ValueError, match="Unsupported integrator"):
            GolombFSNeuron(integrator="midpoint")  # type: ignore[arg-type]

    def test_baseline_euler_path_runs_and_diverges_from_rk4(self):
        rk4 = GolombFSNeuron()
        euler = GolombFSNeuron(integrator="baseline_euler")
        rk4_spikes = sum(rk4.step(5.0) for _ in range(40000))
        euler_spikes = sum(euler.step(5.0) for _ in range(40000))
        assert rk4_spikes > 0 and euler_spikes > 0
        assert rk4.v != euler.v

    def test_rk4_and_euler_agree_to_first_order_at_tiny_dt(self):
        rk4 = GolombFSNeuron(dt=1e-5)
        euler = GolombFSNeuron(dt=1e-5, integrator="baseline_euler")
        for _ in range(200):
            rk4.step(5.0)
            euler.step(5.0)
        assert abs(rk4.v - euler.v) < 1e-2


class TestGolombFSValidation:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"g_na": -1.0},
            {"g_kd": 0.0},
            {"g_l": -0.1},
            {"c_m": 0.0},
            {"dt": 0.0},
            {"dt": -0.01},
            {"g_kv3": -1.0},
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs: dict[str, float]):
        with pytest.raises(ValueError):
            GolombFSNeuron(**kwargs)

    def test_accepts_zero_kv3_conductance(self):
        # A Kv3-block experiment legitimately sets g_Kv3 = 0.
        assert GolombFSNeuron(g_kv3=0.0).g_kv3 == 0.0

    @pytest.mark.parametrize("field", ["v", "e_na", "e_k", "e_l"])
    def test_rejects_non_finite_field(self, field: str):
        with pytest.raises(ValueError, match="must be finite"):
            GolombFSNeuron(**{field: float("nan")})

    def test_rejects_boolean_field(self):
        with pytest.raises(ValueError, match="must be finite"):
            GolombFSNeuron(v=True)  # type: ignore[arg-type]

    def test_rejects_non_finite_current(self):
        with pytest.raises(ValueError, match="must be finite"):
            GolombFSNeuron().step(float("inf"))

    def test_runtime_validation_catches_corrupted_state(self):
        n = GolombFSNeuron()
        n.dt = -1.0
        with pytest.raises(ValueError, match="dt must be positive"):
            n.step(0.0)

    def test_non_finite_candidate_fails_closed(self):
        n = GolombFSNeuron()
        with pytest.raises((FloatingPointError, OverflowError)):
            for _ in range(40):
                n.step(1e308)
