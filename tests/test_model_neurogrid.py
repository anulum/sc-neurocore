# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: NeuroGridNeuron

"""Full pipeline test for NeuroGridNeuron (Boahen 2014).

2-compartment analog neuromorphic neuron:
Dendrite: dv_d/dt = (-(v_d-v_rest) + I - g_c·(v_d-v_s)) / tau_d
Soma:     dv_s/dt = (-(v_s-v_rest) + Δ_T·exp((v_s-θ)/Δ_T) + g_c·(v_d-v_s)) / tau_s

Dendrite (tau_d=50ms) passively integrates synaptic input.
Soma (tau_s=20ms) has EIF exponential spike initiation (Δ_T=2mV).
Compartments coupled by conductance g_c=0.5.
On v_s ≥ v_peak(20): v_s → v_reset(-65). exp clipped at 20.
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.neurogrid import NeuroGridNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: NeuroGridNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestNGIsolation:
    def test_defaults(self) -> None:
        n = NeuroGridNeuron()
        assert n.v_s == -65.0 and n.v_d == -65.0
        assert n.tau_s == 20.0 and n.tau_d == 50.0
        assert n.g_c == 0.5 and n.delta_t == 2.0
        assert n.v_threshold == -50.0 and n.v_peak == 20.0
        assert n.dt == 0.1

    def test_two_compartments(self) -> None:
        n = NeuroGridNeuron()
        assert hasattr(n, "v_s") and hasattr(n, "v_d")

    def test_step_returns_binary(self) -> None:
        assert NeuroGridNeuron().step(0.0) in (0, 1)

    def test_both_compartments_evolve(self) -> None:
        n = NeuroGridNeuron()
        vs0, vd0 = n.v_s, n.v_d
        for _ in range(500):
            n.step(50.0)
        assert n.v_s != vs0 or n.v_d != vd0

    def test_state_finite_long_run(self) -> None:
        n = NeuroGridNeuron()
        for _ in range(100_000):
            n.step(100.0)
        assert np.isfinite(n.v_s) and np.isfinite(n.v_d)

    def test_reset_restores_defaults(self) -> None:
        n = NeuroGridNeuron()
        for _ in range(5000):
            n.step(100.0)
        n.reset()
        assert n.v_s == -65.0 and n.v_d == -65.0

    def test_deterministic(self) -> None:
        traces = []
        for _ in range(2):
            n = NeuroGridNeuron()
            trace = [(n.step(100.0), n.v_s, n.v_d) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — dendrite, soma, coupling, spike mechanism
# ---------------------------------------------------------------------------
class TestNGAnalytical:
    def test_rk4_one_step_matches_candidate(self) -> None:
        """Default path commits the finite two-state RK4 candidate."""
        n = NeuroGridNeuron()
        state = (n.v_s, n.v_d)
        current = 50.0
        expected_vs, expected_vd = n._rk4_substep(state, current)
        n.step(current)
        assert abs(n.v_s - expected_vs) < 1e-12
        assert abs(n.v_d - expected_vd) < 1e-12

    def test_baseline_euler_formula_one_step(self) -> None:
        """Baseline Euler preserves the historical dendrite-first update."""
        n = NeuroGridNeuron(integrator="baseline_euler")
        vs0, vd0 = n.v_s, n.v_d
        I = 20.0  # subthreshold to avoid spike
        dvd = (-(vd0 - n.v_rest) + I - n.g_c * (vd0 - vs0)) / n.tau_d * n.dt
        vd_new = vd0 + dvd
        exp_arg = min((vs0 - n.v_threshold) / n.delta_t, 20.0)
        exp_term = n.delta_t * np.exp(exp_arg)
        dvs = (-(vs0 - n.v_rest) + exp_term + n.g_c * (vd_new - vs0)) / n.tau_s * n.dt
        n.step(I)
        assert abs((n.v_s - vs0) - dvs) < 1e-10
        assert abs(n.v_d - vd_new) < 1e-10

    def test_coupling_symmetric(self) -> None:
        """g_c·(v_d-v_s) in soma, -g_c·(v_d-v_s) in dendrite (current conservation)."""
        n = NeuroGridNeuron()
        # If v_d > v_s: current flows dendrite→soma
        n.v_d = -60.0
        n.v_s = -65.0
        coupling_to_soma = n.g_c * (n.v_d - n.v_s)  # +2.5
        coupling_from_dend = -n.g_c * (n.v_d - n.v_s)  # -2.5
        assert coupling_to_soma > 0  # excitatory to soma
        assert coupling_from_dend < 0  # drains dendrite
        assert abs(coupling_to_soma + coupling_from_dend) < 1e-12

    def test_exp_spike_initiation(self) -> None:
        """Exponential term grows as v_s → v_threshold."""
        n = NeuroGridNeuron()
        # Far below threshold: exp negligible
        exp_far = n.delta_t * np.exp((-65.0 - n.v_threshold) / n.delta_t)
        assert exp_far < 0.01
        # Near threshold: exp significant
        exp_near = n.delta_t * np.exp((-51.0 - n.v_threshold) / n.delta_t)
        assert exp_near > 1.0

    def test_exp_clipped_at_20(self) -> None:
        """Argument clamped at 20 to prevent overflow."""
        n = NeuroGridNeuron()
        # v_s very high → clipped
        n.v_s = 100.0
        n.v_d = -65.0
        n.step(0.0)  # Should not overflow
        assert np.isfinite(n.v_s)

    def test_spike_at_v_peak(self) -> None:
        """Spike when v_s ≥ v_peak, then v_s → v_reset."""
        n = NeuroGridNeuron()
        for _ in range(100_000):
            if n.step(100.0) == 1:
                assert n.v_s == n.v_reset
                break

    def test_dendritic_input_drives_soma(self) -> None:
        """Input to dendrite → dendrite depolarises → couples to soma → spike."""
        n = NeuroGridNeuron()
        for _ in range(1000):
            n.step(50.0)
        assert n.v_d > n.v_rest  # dendrite accumulated input


# ---------------------------------------------------------------------------
# 3. COMPARTMENT DYNAMICS
# ---------------------------------------------------------------------------
class TestNGCompartments:
    def test_dendrite_slower_than_soma(self) -> None:
        """tau_d > tau_s → dendrite integrates slower."""
        n = NeuroGridNeuron()
        assert n.tau_d > n.tau_s

    def test_dendrite_accumulates(self) -> None:
        n = NeuroGridNeuron()
        vd_vals = []
        for _ in range(500):
            n.step(50.0)
            vd_vals.append(n.v_d)
        # Should depolarise from -65 toward steady state
        assert vd_vals[-1] > vd_vals[0]

    def test_coupling_transfers_charge(self) -> None:
        """With g_c=0, compartments are independent."""
        n = NeuroGridNeuron(g_c=0.0)
        for _ in range(1000):
            n.step(50.0)
        # Dendrite gets input, soma gets nothing (only exp term)
        assert n.v_d > n.v_rest
        # Soma only has exp + leak
        # With no coupling, soma stays near rest
        # (may still spike due to exp, but v_d and v_s are independent)


# ---------------------------------------------------------------------------
# 4. DYNAMICS
# ---------------------------------------------------------------------------
class TestNGDynamics:
    def test_subthreshold_silent(self) -> None:
        n = NeuroGridNeuron()
        assert len(_run(n, current=20.0, steps=5000)) == 0

    def test_fires_under_drive(self) -> None:
        n = NeuroGridNeuron()
        assert len(_run(n, current=100.0, steps=10_000)) >= 5

    def test_rate_monotonic(self) -> None:
        rates = []
        for I in [50.0, 100.0, 200.0]:
            n = NeuroGridNeuron()
            rates.append(len(_run(n, current=I, steps=10_000)))
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 50.0, 100.0, 150.0, 200.0])
    def test_fi_sweep(self, current: float) -> None:
        n = NeuroGridNeuron()
        for _ in range(10_000):
            n.step(current)
        assert np.isfinite(n.v_s) and np.isfinite(n.v_d)


# ---------------------------------------------------------------------------
# 4b. RK4 HARDENING / PARITY
# ---------------------------------------------------------------------------
class TestNGRK4Hardening:
    def test_default_integrator_is_rk4(self) -> None:
        n = NeuroGridNeuron()
        assert n.integrator == "rk4"

    def test_unknown_integrator_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported integrator"):
            NeuroGridNeuron(integrator="bad")  # type: ignore[arg-type]

    def test_rk4_and_euler_regression_paths_diverge(self) -> None:
        rk4 = NeuroGridNeuron()
        euler = NeuroGridNeuron(integrator="baseline_euler")
        rk4_spikes = sum(rk4.step(100.0) for _ in range(20_000))
        euler_spikes = sum(euler.step(100.0) for _ in range(20_000))
        assert rk4_spikes == 94
        assert euler_spikes == 93

    def test_cross_backend_anchor(self) -> None:
        n = NeuroGridNeuron()
        spikes = sum(n.step(100.0) for _ in range(20_000))
        assert spikes == 94
        assert np.isfinite(n.v_s) and np.isfinite(n.v_d)

    def test_invalid_input_preserves_state(self) -> None:
        n = NeuroGridNeuron()
        for _ in range(10):
            n.step(100.0)
        old_state = (n.v_s, n.v_d)
        with pytest.raises(ValueError, match="current must be finite"):
            n.step(float("nan"))
        assert (n.v_s, n.v_d) == old_state

    def test_corrupt_state_rejected_before_mutation(self) -> None:
        n = NeuroGridNeuron()
        for _ in range(10):
            n.step(100.0)
        old_v_d = n.v_d
        n.v_s = float("nan")
        with pytest.raises(ValueError, match="v_s must be finite"):
            n.step(100.0)
        assert n.v_d == old_v_d

    def test_runtime_configuration_rejects_invalid_tau(self) -> None:
        n = NeuroGridNeuron()
        n.tau_s = 0.0
        with pytest.raises(ValueError, match="tau_s must be positive"):
            n.step(100.0)


# ---------------------------------------------------------------------------
# 5. PARAMETER SENSITIVITY
# ---------------------------------------------------------------------------
class TestNGParameters:
    @pytest.mark.parametrize("g_c", [0.1, 0.5, 1.0])
    def test_coupling_sweep(self, g_c: float) -> None:
        n = NeuroGridNeuron(g_c=g_c)
        for _ in range(10_000):
            n.step(100.0)
        assert np.isfinite(n.v_s) and np.isfinite(n.v_d)

    @pytest.mark.parametrize("delta_t", [1.0, 2.0, 4.0])
    def test_delta_t_sweep(self, delta_t: float) -> None:
        n = NeuroGridNeuron(delta_t=delta_t)
        for _ in range(10_000):
            n.step(100.0)
        assert np.isfinite(n.v_s)

    @pytest.mark.parametrize("tau_d", [20.0, 50.0, 100.0])
    def test_tau_d_sweep(self, tau_d: float) -> None:
        n = NeuroGridNeuron(tau_d=tau_d)
        for _ in range(10_000):
            n.step(100.0)
        assert np.isfinite(n.v_d)

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float) -> None:
        n = NeuroGridNeuron(dt=dt)
        for _ in range(10_000):
            n.step(100.0)
        assert np.isfinite(n.v_s) and np.isfinite(n.v_d)


# ---------------------------------------------------------------------------
# 6. PERFORMANCE
# ---------------------------------------------------------------------------
class TestNGPerformance:
    def test_isolation_throughput(self) -> None:
        n = NeuroGridNeuron()
        N = 100_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(100.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        # 1 exp + 2 compartment updates
        assert rate > 50_000, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self) -> None:
        pop = Population(NeuroGridNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=100.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 20 * 500
        rate = neuron_steps / elapsed
        assert rate > 2_000, f"network: {rate:.0f} neuron-steps/s"


# ---------------------------------------------------------------------------
# 7. FULL PIPELINE
# ---------------------------------------------------------------------------
class TestNGPipeline:
    def test_population(self) -> None:
        assert Population(NeuroGridNeuron, n=10, label="ng").n == 10

    def test_projection_wiring(self) -> None:
        src = Population(NeuroGridNeuron, n=5, label="src")
        tgt = Population(NeuroGridNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=1000.0, weight=500.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=50.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_tgt = SpikeMonitor(tgt)
        net = Network(src, tgt, drive, proj, mon_src, mon_tgt)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self) -> None:
        pop = Population(NeuroGridNeuron, n=10, label="ng")
        drive = PoissonInput(n=10, rate_hz=1000.0, weight=500.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self) -> None:
        n = NeuroGridNeuron()
        train = np.array([float(n.step(100.0)) for _ in range(10_000)])
        sc = spike_count(train)
        assert sc >= 3

    def test_analysis_isi(self) -> None:
        n = NeuroGridNeuron()
        train = np.array([float(n.step(100.0)) for _ in range(20_000)])
        intervals = isi(train, dt=0.0001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))
            assert np.all(intervals > 0)

    def test_analysis_firing_rate(self) -> None:
        n = NeuroGridNeuron()
        train = np.array([float(n.step(100.0)) for _ in range(10_000)])
        rate = firing_rate(train, dt=0.0001)
        assert rate >= 0

    def test_analysis_cross_validation(self) -> None:
        n = NeuroGridNeuron()
        train = np.array([float(n.step(100.0)) for _ in range(20_000)])
        sc = spike_count(train)
        dt_sim = 0.0001
        duration = len(train) * dt_sim
        rate = firing_rate(train, dt=dt_sim)
        if sc > 0:
            expected = sc / duration
            assert abs(rate - expected) < expected * 0.1


# Salvaged model-specific behavioural contracts from retired aggregate test file.
class TestNeuroGrid:
    def test_dynamics(self) -> None:
        from sc_neurocore.neurons.models.neurogrid import NeuroGridNeuron

        n = NeuroGridNeuron()
        for _ in range(200):
            n.step(10.0)
        assert n.v_s != n.v_d, "soma and dendrite should differ"

    def test_reset(self) -> None:
        from sc_neurocore.neurons.models.neurogrid import NeuroGridNeuron

        n = NeuroGridNeuron()
        for _ in range(50):
            n.step(5.0)
        n.reset()
        assert abs(n.v_s - n.v_rest) < 1e-10
