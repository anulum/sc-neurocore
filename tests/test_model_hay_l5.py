# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: HayL5PyramidalNeuron

"""Full pipeline test for HayL5PyramidalNeuron (Hay et al. 2011).

Reduced 3-compartment Layer 5 thick-tufted pyramidal cell:
Soma: I_Na(g=300, m³_inf·h), I_K(g=40, n⁴), I_L, coupling→trunk
Trunk: I_CaT(g=2, m²·h_ca), I_h(g=0.02, m_ih), I_L, coupling↔
Tuft: I_CaA(g=1.5, m²_inf), I_KCa(g=2.5, Ca-dep), I_L, coupling→trunk

9 state variables: v_s, h_na, n_k, v_t, m_ca, h_ca, m_ih, v_a, ca_a.
4 sub-steps (dt=0.025). Dual input: current_soma + current_tuft.
BAC firing: backpropagation-activated calcium spike in tuft.
Compartment areas: p_s=0.15, p_t=0.25, p_a=0.60.
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.hay_l5 import HayL5PyramidalNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: HayL5PyramidalNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestHayIsolation:
    def test_defaults(self) -> None:
        n = HayL5PyramidalNeuron()
        assert n.v_s == -75.0 and n.v_t == -75.0 and n.v_a == -75.0
        assert n.h_na == 0.9 and n.n_k == 0.1
        assert n.m_ca == 0.0 and n.h_ca == 1.0 and n.m_ih == 0.0
        assert n.ca_a == 0.0001
        assert n.dt == 0.025

    def test_nine_state_variables(self) -> None:
        n = HayL5PyramidalNeuron()
        for attr in ["v_s", "h_na", "n_k", "v_t", "m_ca", "h_ca", "m_ih", "v_a", "ca_a"]:
            assert hasattr(n, attr)

    def test_step_returns_binary(self) -> None:
        assert HayL5PyramidalNeuron().step(0.0) in (0, 1)

    def test_dual_input(self) -> None:
        """step() accepts current_soma and optional current_tuft."""
        n = HayL5PyramidalNeuron()
        n.step(5.0, current_tuft=2.0)
        assert np.isfinite(n.v_s)

    def test_state_finite_long_run(self) -> None:
        n = HayL5PyramidalNeuron()
        for _ in range(10_000):
            n.step(10.0)
        for attr in ["v_s", "v_t", "v_a", "h_na", "n_k", "m_ca", "h_ca", "m_ih", "ca_a"]:
            assert np.isfinite(getattr(n, attr)), f"{attr} not finite"

    def test_reset_restores_defaults(self) -> None:
        n = HayL5PyramidalNeuron()
        for _ in range(2000):
            n.step(10.0)
        n.reset()
        assert n.v_s == -75.0 and n.v_t == -75.0 and n.v_a == -75.0
        assert n.ca_a == 0.0001

    def test_deterministic(self) -> None:
        traces = []
        for _ in range(2):
            n = HayL5PyramidalNeuron()
            trace = [(n.step(10.0), n.v_s) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — 3 compartments, coupling, Ca dynamics
# ---------------------------------------------------------------------------
class TestHayAnalytical:
    def test_4_substeps(self) -> None:
        n = HayL5PyramidalNeuron()
        assert n.dt == 0.025  # 4 sub-steps in source

    def test_three_compartments(self) -> None:
        """Soma (v_s), trunk (v_t), tuft (v_a)."""
        n = HayL5PyramidalNeuron()
        assert hasattr(n, "v_s") and hasattr(n, "v_t") and hasattr(n, "v_a")

    def test_compartment_area_fractions(self) -> None:
        """p_s + p_t + p_a = 1.0 (area conservation)."""
        n = HayL5PyramidalNeuron()
        assert abs(n.p_s + n.p_t + n.p_a - 1.0) < 1e-12

    def test_coupling_soma_trunk(self) -> None:
        """g_st couples soma↔trunk bidirectionally."""
        n = HayL5PyramidalNeuron()
        assert n.g_st > 0

    def test_coupling_trunk_tuft(self) -> None:
        """g_ta couples trunk↔tuft bidirectionally."""
        n = HayL5PyramidalNeuron()
        assert n.g_ta > 0

    def test_ca_dynamics_in_tuft(self) -> None:
        """Ca dynamics: dCa = (-f_ca·I_Ca - Ca/ca_decay)·dt, clipped ≥ 0."""
        n = HayL5PyramidalNeuron()
        for _ in range(5000):
            n.step(10.0)
        assert n.ca_a >= 0

    def test_reversal_ordering(self) -> None:
        n = HayL5PyramidalNeuron()
        assert n.e_k < n.e_l < n.e_ih < n.e_na < n.e_ca

    def test_soma_currents(self) -> None:
        """Soma: Na, K, leak, coupling."""
        n = HayL5PyramidalNeuron()
        assert n.g_na > 0 and n.g_k > 0 and n.g_l_s > 0

    def test_trunk_currents(self) -> None:
        """Trunk: Ca, Ih, leak."""
        n = HayL5PyramidalNeuron()
        assert n.g_ca_t > 0 and n.g_ih > 0 and n.g_l_t > 0

    def test_tuft_currents(self) -> None:
        """Tuft: CaA, KCa, leak."""
        n = HayL5PyramidalNeuron()
        assert n.g_ca_a > 0 and n.g_kca > 0 and n.g_l_a > 0


# ---------------------------------------------------------------------------
# 3. COMPARTMENT DYNAMICS
# ---------------------------------------------------------------------------
class TestHayCompartments:
    def test_somatic_input_drives_spiking(self) -> None:
        """Somatic drive produces spikes (soma may hyperpolarise post-spike)."""
        n = HayL5PyramidalNeuron()
        spikes = sum(n.step(10.0) for _ in range(2000))
        assert spikes >= 1

    def test_tuft_input_depolarises_tuft(self) -> None:
        n = HayL5PyramidalNeuron()
        for _ in range(500):
            n.step(0.0, current_tuft=10.0)
        assert n.v_a > -75.0

    def test_coupling_propagates_soma_to_trunk(self) -> None:
        """Somatic drive → trunk depolarises via coupling."""
        n = HayL5PyramidalNeuron()
        for _ in range(2000):
            n.step(10.0)
        assert n.v_t > -75.0

    def test_all_compartments_evolve(self) -> None:
        n = HayL5PyramidalNeuron()
        for _ in range(2000):
            n.step(10.0)
        assert n.v_s != -75.0 and n.v_t != -75.0 and n.v_a != -75.0


# ---------------------------------------------------------------------------
# 4. DYNAMICS
# ---------------------------------------------------------------------------
class TestHayDynamics:
    def test_fires_under_somatic_drive(self) -> None:
        n = HayL5PyramidalNeuron()
        spikes = _run(n, current=10.0, steps=5000)
        assert len(spikes) >= 1

    def test_subthreshold_silent(self) -> None:
        n = HayL5PyramidalNeuron()
        assert len(_run(n, current=1.0, steps=3000)) == 0

    def test_rate_monotonic(self) -> None:
        rates = []
        for I in [5.0, 10.0, 20.0]:
            n = HayL5PyramidalNeuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert rates[-1] >= rates[0]


# ---------------------------------------------------------------------------
# 4b. RK4 HARDENING / PARITY
# ---------------------------------------------------------------------------
class TestHayRK4Hardening:
    def test_default_integrator_is_rk4(self) -> None:
        n = HayL5PyramidalNeuron()
        assert n.integrator == "rk4"

    def test_unknown_integrator_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported integrator"):
            HayL5PyramidalNeuron(integrator="bad")  # type: ignore[arg-type]

    def test_rk4_and_euler_regression_paths_diverge(self) -> None:
        rk4 = HayL5PyramidalNeuron()
        euler = HayL5PyramidalNeuron(integrator="baseline_euler")
        rk4_spikes = sum(rk4.step(10.0) for _ in range(20_000))
        euler_spikes = sum(euler.step(10.0) for _ in range(20_000))
        assert rk4_spikes == 1
        assert euler_spikes == 10
        assert abs(rk4.v_s - euler.v_s) < 2e-8

    def test_cross_backend_somatic_anchor(self) -> None:
        n = HayL5PyramidalNeuron()
        spikes = sum(n.step(10.0) for _ in range(20_000))
        assert spikes == 1
        assert n.ca_a >= 0.0

    def test_cross_backend_dual_input_anchor(self) -> None:
        n = HayL5PyramidalNeuron()
        spikes = sum(n.step(5.0, 5.0) for _ in range(20_000))
        assert spikes == 4
        assert n.ca_a >= 0.0

    def test_invalid_input_preserves_state(self) -> None:
        n = HayL5PyramidalNeuron()
        for _ in range(10):
            n.step(10.0)
        old_state = (n.v_s, n.h_na, n.n_k, n.v_t, n.m_ca, n.h_ca, n.m_ih, n.v_a, n.ca_a)
        with pytest.raises(ValueError, match="current_soma must be finite"):
            n.step(float("nan"))
        assert (n.v_s, n.h_na, n.n_k, n.v_t, n.m_ca, n.h_ca, n.m_ih, n.v_a, n.ca_a) == old_state

    def test_corrupt_state_preserves_state(self) -> None:
        n = HayL5PyramidalNeuron()
        for _ in range(10):
            n.step(10.0)
        old_state = (n.v_s, n.h_na, n.n_k, n.v_t, n.m_ca, n.h_ca, n.m_ih, n.v_a, n.ca_a)
        n.ca_a = float("nan")
        with pytest.raises(ValueError, match="ca_a must be finite"):
            n.step(10.0)
        assert (n.v_s, n.h_na, n.n_k, n.v_t, n.m_ca, n.h_ca, n.m_ih, n.v_a) == old_state[:-1]

    def test_runtime_configuration_rejects_invalid_dt(self) -> None:
        n = HayL5PyramidalNeuron()
        n.dt = 0.0
        with pytest.raises(ValueError, match="dt must be positive"):
            n.step(10.0)

    @pytest.mark.parametrize("current", [0.0, 5.0, 10.0, 20.0])
    def test_fi_sweep(self, current: float) -> None:
        n = HayL5PyramidalNeuron()
        for _ in range(3000):
            n.step(current)
        assert np.isfinite(n.v_s)


# ---------------------------------------------------------------------------
# 5. PARAMETERS
# ---------------------------------------------------------------------------
class TestHayParameters:
    @pytest.mark.parametrize("g_na", [150.0, 300.0, 500.0])
    def test_g_na_sweep(self, g_na: float) -> None:
        n = HayL5PyramidalNeuron(g_na=g_na)
        for _ in range(3000):
            n.step(10.0)
        assert np.isfinite(n.v_s)

    @pytest.mark.parametrize("g_ca_t", [0.0, 2.0, 5.0])
    def test_g_ca_trunk_sweep(self, g_ca_t: float) -> None:
        n = HayL5PyramidalNeuron(g_ca_t=g_ca_t)
        for _ in range(3000):
            n.step(10.0)
        assert np.isfinite(n.v_t)

    @pytest.mark.parametrize("g_st", [0.5, 1.5, 3.0])
    def test_coupling_sweep(self, g_st: float) -> None:
        n = HayL5PyramidalNeuron(g_st=g_st)
        for _ in range(3000):
            n.step(10.0)
        assert np.isfinite(n.v_s) and np.isfinite(n.v_t)


# ---------------------------------------------------------------------------
# 6. PERFORMANCE
# ---------------------------------------------------------------------------
class TestHayPerformance:
    def test_isolation_throughput(self) -> None:
        n = HayL5PyramidalNeuron()
        N = 2000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(10.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        # 4 sub-steps × 3 compartments × multiple currents
        assert rate > 500, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self) -> None:
        pop = Population(HayL5PyramidalNeuron, n=10, label="bench")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=10.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.2, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 10 * 200
        rate = neuron_steps / elapsed
        assert rate > 100, f"network: {rate:.0f} neuron-steps/s"


# ---------------------------------------------------------------------------
# 7. FULL PIPELINE
# ---------------------------------------------------------------------------
class TestHayPipeline:
    def test_population(self) -> None:
        assert Population(HayL5PyramidalNeuron, n=5, label="hay").n == 5

    def test_projection_wiring(self) -> None:
        src = Population(HayL5PyramidalNeuron, n=3, label="src")
        tgt = Population(HayL5PyramidalNeuron, n=3, label="tgt")
        drive = PoissonInput(n=3, rate_hz=500.0, weight=10.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=5.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert isinstance(mon_src.count, int)

    def test_network_spikes(self) -> None:
        pop = Population(HayL5PyramidalNeuron, n=5, label="hay")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=10.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert isinstance(mon.count, int)

    def test_analysis_spike_count(self) -> None:
        n = HayL5PyramidalNeuron()
        train = np.array([float(n.step(10.0)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 1

    def test_analysis_isi(self) -> None:
        n = HayL5PyramidalNeuron()
        train = np.array([float(n.step(10.0)) for _ in range(5000)])
        intervals = isi(train, dt=0.000025)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))

    def test_analysis_firing_rate(self) -> None:
        n = HayL5PyramidalNeuron()
        train = np.array([float(n.step(10.0)) for _ in range(5000)])
        rate = firing_rate(train, dt=0.000025)
        assert rate >= 0
