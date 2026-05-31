# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: LarterBreakspearNeuron

"""Full pipeline test for LarterBreakspearNeuron (Breakspear et al. 2003).

Neural mass with conductance-based ion channels (TVB model):
dV = -I_Ca - I_Na - I_K - I_L + I_ext + coupling + a_ee·V
dw = φ·(m_K(V) - w) / τ_K
dz = b·(V + 0.5 - z)

4 currents with tanh activations: m_Ca, m_Na, m_K.
Returns V (float), not binary spike. Used in whole-brain modelling.
Pipeline and performance contract tests live in this module-specific file."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.larter_breakspear import LarterBreakspearNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestLBIsolation:
    def test_defaults(self):
        n = LarterBreakspearNeuron()
        assert n.v == -0.5 and n.w == 0.0 and n.z == 0.0
        assert n.g_ca == 1.1 and n.g_na == 6.7
        assert n.dt == 0.01
        assert n.integrator == "rk4"

    def test_step_returns_float(self):
        n = LarterBreakspearNeuron()
        result = n.step(0.0)
        assert isinstance(result, (float, np.floating))

    def test_state_finite_long_run(self):
        n = LarterBreakspearNeuron()
        for _ in range(50_000):
            n.step(0.0)
        assert np.isfinite(n.v) and np.isfinite(n.w) and np.isfinite(n.z)

    def test_reset_restores_defaults(self):
        n = LarterBreakspearNeuron()
        for _ in range(5000):
            n.step(0.0)
        n.reset()
        assert n.v == -0.5 and n.w == 0.0 and n.z == 0.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = LarterBreakspearNeuron()
            trace = [n.step(0.0) for _ in range(500)]
            traces.append(trace)
        assert traces[0] == traces[1]

    def test_rejects_nonfinite_coupling(self):
        n = LarterBreakspearNeuron()
        with pytest.raises(ValueError, match="coupling"):
            n.step(float("inf"))


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — tanh activations, 4 currents, recurrent excitation
# ---------------------------------------------------------------------------
class TestLBAnalytical:
    def test_m_ca_tanh(self):
        """m_Ca = 0.5·(1 + tanh((V+0.01)/0.15))."""
        n = LarterBreakspearNeuron()
        # At V=-0.01: m_Ca = 0.5
        assert abs(n._m_ca(-0.01) - 0.5) < 1e-10

    def test_m_na_tanh(self):
        """m_Na = 0.5·(1 + tanh((V-0.12)/0.15))."""
        n = LarterBreakspearNeuron()
        assert abs(n._m_na(0.12) - 0.5) < 1e-10

    def test_m_k_tanh(self):
        """m_K = 0.5·(1 + tanh((V-v0)/0.3))."""
        n = LarterBreakspearNeuron()
        assert abs(n._m_k(n.v0) - 0.5) < 1e-10

    def test_four_currents_positive_conductances(self):
        n = LarterBreakspearNeuron()
        assert n.g_ca > 0 and n.g_na > 0 and n.g_k > 0 and n.g_l > 0

    def test_recurrent_excitation(self):
        """a_ee·V term provides recurrent self-excitation."""
        n = LarterBreakspearNeuron()
        assert n.a_ee > 0

    def test_output_is_voltage(self):
        """step() returns V directly."""
        n = LarterBreakspearNeuron()
        result = n.step(0.0)
        assert result == n.v

    def test_three_state_variables(self):
        n = LarterBreakspearNeuron()
        for attr in ["v", "w", "z"]:
            assert hasattr(n, attr)


# ---------------------------------------------------------------------------
# 3. DYNAMICS
# ---------------------------------------------------------------------------
class TestLBDynamics:
    def test_v_oscillates(self):
        n = LarterBreakspearNeuron()
        vs = [n.step(0.0) for _ in range(10_000)]
        assert np.std(vs) > 0.01

    def test_runge_kutta_tracks_substepped_reference_better_than_euler(self):
        horizon = 0.5
        coupling = 0.15
        reference = LarterBreakspearNeuron(dt=0.0005, integrator="rk4")
        coarse_rk4 = LarterBreakspearNeuron(dt=0.05, integrator="rk4")
        coarse_euler = LarterBreakspearNeuron(dt=0.05, integrator="euler")

        for _ in range(int(horizon / reference.dt)):
            reference.step(coupling)
        for _ in range(int(horizon / coarse_rk4.dt)):
            coarse_rk4.step(coupling)
            coarse_euler.step(coupling)

        rk4_error = abs(coarse_rk4.v - reference.v)
        euler_error = abs(coarse_euler.v - reference.v)

        assert rk4_error < euler_error
        assert rk4_error < 1e-3

    def test_coupling_affects_dynamics(self):
        n1 = LarterBreakspearNeuron()
        n2 = LarterBreakspearNeuron()
        for _ in range(5000):
            n1.step(0.0)
            n2.step(1.0)
        assert n1.v != n2.v

    @pytest.mark.parametrize("coupling", [0.0, 0.5, 1.0, 2.0])
    def test_coupling_sweep(self, coupling: float):
        n = LarterBreakspearNeuron()
        for _ in range(5000):
            n.step(coupling)
        assert np.isfinite(n.v)


# ---------------------------------------------------------------------------
# 4. PARAMETERS
# ---------------------------------------------------------------------------
class TestLBParameters:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("dt", 0.0),
            ("dt", float("nan")),
            ("tau_k", 0.0),
            ("phi", -0.1),
            ("b", -0.1),
            ("w", -0.01),
            ("w", 1.01),
            ("g_ca", 0.0),
            ("g_na", 0.0),
            ("g_k", 0.0),
            ("g_l", 0.0),
        ],
    )
    def test_rejects_nonphysical_parameters(self, field: str, value: float):
        with pytest.raises(ValueError, match=field):
            LarterBreakspearNeuron(**{field: value})

    @pytest.mark.parametrize("integrator", ["rk4", "euler"])
    def test_runtime_parameter_corruption_fails_before_mutation(self, integrator: str):
        n = LarterBreakspearNeuron(integrator=integrator)
        n.tau_k = float("nan")
        before = (n.v, n.w, n.z)

        with pytest.raises(ValueError):
            n.step(0.0)

        assert (n.v, n.w, n.z) == before

    @pytest.mark.parametrize("integrator", ["rk4", "euler"])
    def test_potassium_gate_bounds_fail_before_mutation(self, integrator: str):
        n = LarterBreakspearNeuron(w=0.0, dt=100.0, integrator=integrator)
        before = (n.v, n.w, n.z)

        with pytest.raises(FloatingPointError, match="potassium gate"):
            n.step(-100.0)

        assert (n.v, n.w, n.z) == before

    def test_rejects_unknown_integrator(self):
        with pytest.raises(ValueError, match="integrator"):
            LarterBreakspearNeuron(integrator="verlet")

    @pytest.mark.parametrize("g_ca", [0.5, 1.1, 2.0])
    def test_g_ca_sweep(self, g_ca: float):
        n = LarterBreakspearNeuron(g_ca=g_ca)
        for _ in range(5000):
            n.step(0.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("i_ext", [0.0, 0.3, 1.0])
    def test_i_ext_sweep(self, i_ext: float):
        n = LarterBreakspearNeuron(i_ext=i_ext)
        for _ in range(5000):
            n.step(0.0)
        assert np.isfinite(n.v)

    @pytest.mark.parametrize("a_ee", [0.0, 0.36, 0.5])
    def test_a_ee_sweep(self, a_ee: float):
        n = LarterBreakspearNeuron(a_ee=a_ee)
        for _ in range(5000):
            n.step(0.0)
        assert np.isfinite(n.v)


# ---------------------------------------------------------------------------
# 5. PERFORMANCE
# ---------------------------------------------------------------------------
class TestLBPerformance:
    def test_isolation_throughput(self):
        n = LarterBreakspearNeuron()
        N = 50_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(0.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        assert rate > 10_000, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(LarterBreakspearNeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 20 * 500
        rate = neuron_steps / elapsed
        assert rate > 1_000, f"network: {rate:.0f} neuron-steps/s"


# ---------------------------------------------------------------------------
# 6. PIPELINE
# ---------------------------------------------------------------------------
class TestLBPipeline:
    def test_population(self):
        assert Population(LarterBreakspearNeuron, n=5, label="lb").n == 5

    def test_projection_wiring(self):
        src = Population(LarterBreakspearNeuron, n=5, label="src")
        tgt = Population(LarterBreakspearNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=0.3, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert isinstance(mon_src.count, int)

    def test_network_runs(self):
        pop = Population(LarterBreakspearNeuron, n=10, label="lb")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.5, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert isinstance(mon.count, int)
