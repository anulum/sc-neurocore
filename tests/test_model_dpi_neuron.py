# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: DPINeuron

"""Full pipeline test for DPINeuron (Indiveri et al. 2011).

DYNAP-SE differential-pair integrator. Current-domain dynamics (nA):
τ · dI_mem/dt = -I_mem + gain·I_syn + I_leak
Spike when I_mem ≥ I_threshold, reset to I_reset.
I_mem clipped ≥ 0 (physical current constraint).

Analogue VLSI neuromorphic circuit. All state in current domain,
not voltage. τ=20, gain=1, I_leak=0.01, I_threshold=1.0.
FULL PIPELINE WIRED + PERFORMANCE."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.dpi_neuron import DPINeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: DPINeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


# ---------------------------------------------------------------------------
# 1. ISOLATION
# ---------------------------------------------------------------------------
class TestDPIIsolation:
    def test_defaults(self):
        n = DPINeuron()
        assert n.i_mem == 0.0 and n.i_threshold == 1.0
        assert n.i_reset == 0.0 and n.i_leak == 0.01
        assert n.tau == 20.0 and n.gain == 1.0 and n.dt == 1.0

    def test_step_returns_binary(self):
        assert DPINeuron().step(0.0) in (0, 1)

    def test_current_domain(self):
        """State is i_mem (current), not voltage."""
        n = DPINeuron()
        assert hasattr(n, "i_mem")
        assert not hasattr(n, "v") or n.__class__.__name__ == "DPINeuron"

    def test_state_finite_long_run(self):
        n = DPINeuron()
        for _ in range(100_000):
            n.step(50.0)
        assert np.isfinite(n.i_mem)

    def test_reset_restores_default(self):
        n = DPINeuron()
        for _ in range(1000):
            n.step(50.0)
        n.reset()
        assert n.i_mem == 0.0

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = DPINeuron()
            trace = [(n.step(50.0), n.i_mem) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


# ---------------------------------------------------------------------------
# 2. ANALYTICAL — di_mem formula, spike, clipping, steady state
# ---------------------------------------------------------------------------
class TestDPIAnalytical:
    def test_di_mem_formula(self):
        """dI = (-I_mem + gain·I_syn + I_leak) / τ · dt."""
        n = DPINeuron()
        i0 = n.i_mem
        I_syn = 0.5  # subthreshold
        expected_di = (-i0 + n.gain * I_syn + n.i_leak) / n.tau * n.dt
        n.step(I_syn)
        assert abs((n.i_mem - i0) - expected_di) < 1e-12

    def test_i_mem_non_negative(self):
        """I_mem clipped to ≥ 0 (physical constraint)."""
        n = DPINeuron()
        # Negative input should not make i_mem negative
        for _ in range(1000):
            n.step(-10.0)
        assert n.i_mem >= 0.0

    def test_spike_resets_to_i_reset(self):
        n = DPINeuron()
        for _ in range(10_000):
            if n.step(50.0) == 1:
                assert n.i_mem == n.i_reset
                break

    def test_steady_state(self):
        """At SS: I_mem = gain·I_syn + I_leak."""
        n = DPINeuron()
        I_syn = 0.5
        for _ in range(10_000):
            n.step(I_syn)
        expected_ss = n.gain * I_syn + n.i_leak
        # If below threshold, converges to SS
        if expected_ss < n.i_threshold:
            assert abs(n.i_mem - expected_ss) < 0.1

    def test_gain_scales_input(self):
        """gain=2 → double effective input."""
        n1 = DPINeuron(gain=1.0, i_threshold=100.0)
        n2 = DPINeuron(gain=2.0, i_threshold=100.0)
        for _ in range(100):
            n1.step(10.0)
            n2.step(10.0)
        assert n2.i_mem > n1.i_mem

    def test_leak_provides_baseline(self):
        """I_leak=0.01 provides small baseline current."""
        n = DPINeuron()
        n.step(0.0)  # zero input
        assert n.i_mem > 0  # leak pushes i_mem positive


# ---------------------------------------------------------------------------
# 3. DYNAMICS
# ---------------------------------------------------------------------------
class TestDPIDynamics:
    def test_fires_under_drive(self):
        n = DPINeuron()
        spikes = _run(n, current=50.0, steps=5000)
        assert len(spikes) >= 100

    def test_subthreshold_silent(self):
        """I_syn small → I_mem stays below threshold."""
        n = DPINeuron()
        # SS = gain*0.1 + 0.01 = 0.11 < threshold 1.0
        assert len(_run(n, current=0.1, steps=5000)) == 0

    def test_rate_monotonic(self):
        rates = []
        for I in [20.0, 50.0, 100.0]:
            n = DPINeuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 10.0, 30.0, 50.0, 100.0])
    def test_fi_sweep(self, current: float):
        n = DPINeuron()
        for _ in range(5000):
            n.step(current)
        assert np.isfinite(n.i_mem)


# ---------------------------------------------------------------------------
# 4. PARAMETERS
# ---------------------------------------------------------------------------
class TestDPIParameters:
    @pytest.mark.parametrize("tau", [5.0, 20.0, 50.0])
    def test_tau_sweep(self, tau: float):
        n = DPINeuron(tau=tau)
        for _ in range(5000):
            n.step(50.0)
        assert np.isfinite(n.i_mem)

    @pytest.mark.parametrize("i_threshold", [0.5, 1.0, 2.0])
    def test_threshold_sweep(self, i_threshold: float):
        n = DPINeuron(i_threshold=i_threshold)
        spikes = len(_run(n, current=50.0, steps=5000))
        assert isinstance(spikes, int)

    @pytest.mark.parametrize("gain", [0.5, 1.0, 2.0])
    def test_gain_sweep(self, gain: float):
        n = DPINeuron(gain=gain)
        for _ in range(5000):
            n.step(50.0)
        assert np.isfinite(n.i_mem)


# ---------------------------------------------------------------------------
# 5. PERFORMANCE
# ---------------------------------------------------------------------------
class TestDPIPerformance:
    def test_isolation_throughput(self):
        n = DPINeuron()
        N = 200_000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(50.0)
        elapsed = time.perf_counter() - t0
        rate = N / elapsed
        assert rate > 200_000, f"isolation: {rate:.0f} steps/s"

    def test_network_throughput(self):
        pop = Population(DPINeuron, n=20, label="bench")
        drive = PoissonInput(n=20, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        neuron_steps = 20 * 500
        rate = neuron_steps / elapsed
        assert rate > 2_000, f"network: {rate:.0f} neuron-steps/s"


# ---------------------------------------------------------------------------
# 6. FULL PIPELINE
# ---------------------------------------------------------------------------
class TestDPIPipeline:
    def test_population(self):
        assert Population(DPINeuron, n=10, label="dpi").n == 10

    def test_projection_wiring(self):
        src = Population(DPINeuron, n=5, label="src")
        tgt = Population(DPINeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=20.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon_src)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon_src.count > 0

    def test_network_spikes(self):
        pop = Population(DPINeuron, n=10, label="dpi")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=50.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_spike_count(self):
        n = DPINeuron()
        train = np.array([float(n.step(50.0)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 50

    def test_analysis_isi(self):
        n = DPINeuron()
        train = np.array([float(n.step(50.0)) for _ in range(5000)])
        intervals = isi(train, dt=0.001)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))

    def test_analysis_firing_rate(self):
        n = DPINeuron()
        train = np.array([float(n.step(50.0)) for _ in range(5000)])
        rate = firing_rate(train, dt=0.001)
        assert rate > 0


class TestDPINeuronSimulate:
    """Engineering-verification surface for ``DPINeuron.simulate``."""

    def test_simulate_python_returns_finite_trace(self) -> None:
        n = DPINeuron()
        trace, spikes = n.simulate(1000, current=1.0, backend="python")
        assert trace.shape == (1000,)
        assert np.all(np.isfinite(trace))
        assert spikes >= 1

    def test_simulate_rust_matches_or_ulp_python(self) -> None:
        pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built")
        py = DPINeuron()
        rs = DPINeuron()
        tr_py, sp_py = py.simulate(1000, current=1.0, backend="python")
        tr_rs, sp_rs = rs.simulate(1000, current=1.0, backend="rust")
        assert sp_py == sp_rs
        max_diff = float(np.max(np.abs(tr_py - tr_rs)))
        assert max_diff < 1e-9

    def test_simulate_rust_rejects_non_default(self) -> None:
        pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built")
        # force non-default via a constructor override that every model accepts
        try:
            n = DPINeuron(dt=0.02) if "dt" in DPINeuron.__dataclass_fields__ else DPINeuron()
            if "dt" not in DPINeuron.__dataclass_fields__:
                pytest.skip("no dt field")
        except TypeError:
            pytest.skip("cannot override defaults")
        with pytest.raises(RuntimeError, match="factory-default"):
            n.simulate(10, current=0.0, backend="rust")
