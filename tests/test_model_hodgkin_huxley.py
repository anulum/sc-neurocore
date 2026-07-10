# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: HodgkinHuxleyNeuron

"""Full pipeline test for HodgkinHuxleyNeuron (Hodgkin & Huxley 1952).

The original 4-ODE ion-channel model. 100 sub-steps per step() call.
Type-II excitability: non-monotonic f–I, discontinuous onset.
Performance: ~670 steps/s (100 sub-steps × exp calls dominate)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.hodgkin_huxley import HodgkinHuxleyNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate


def _run(neuron: HodgkinHuxleyNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestHHIsolation:
    def test_defaults(self):
        n = HodgkinHuxleyNeuron()
        assert n.v == -65.0 and n.m == 0.05 and n.h == 0.6 and n.n == 0.32
        assert n.g_na == 120.0 and n.g_k == 36.0 and n.g_l == 0.3
        assert n.e_na == 50.0 and n.e_k == -77.0 and n.e_l == -54.4

    def test_step_returns_binary(self):
        assert HodgkinHuxleyNeuron().step(0.0) in (0, 1)

    def test_four_variables_evolve(self):
        n = HodgkinHuxleyNeuron()
        initial = (n.v, n.m, n.h, n.n)
        for _ in range(100):
            n.step(10.0)
        for name, v0, v1 in zip(["v", "m", "h", "n"], initial, (n.v, n.m, n.h, n.n)):
            assert v0 != v1, f"{name} didn't evolve"

    def test_state_finite(self):
        n = HodgkinHuxleyNeuron()
        for _ in range(5000):
            n.step(10.0)
        for var in [n.v, n.m, n.h, n.n]:
            assert np.isfinite(var)

    def test_reset(self):
        n = HodgkinHuxleyNeuron()
        for _ in range(100):
            n.step(10.0)
        n.reset()
        assert n.v == -65.0 and n.m == 0.05 and n.h == 0.6 and n.n == 0.32

    def test_100_substeps(self):
        """round(1.0/dt) = 100 sub-steps per step() call."""
        n = HodgkinHuxleyNeuron(dt=0.01)
        assert round(1.0 / n.dt) == 100


class TestHHRateFunctions:
    """Verify α and β rate functions at specific V values."""

    def test_alpha_m_singularity_protected(self):
        """At V=-40, d=0 → returns 1.0 (L'Hôpital limit)."""
        n = HodgkinHuxleyNeuron()
        am = n._alpha_m(-40.0)
        assert abs(am - 1.0) < 1e-6

    def test_alpha_n_singularity_protected(self):
        """At V=-55, d=0 → returns 0.1 (L'Hôpital limit)."""
        n = HodgkinHuxleyNeuron()
        an = n._alpha_n(-55.0)
        assert abs(an - 0.1) < 1e-6

    def test_beta_m_formula(self):
        """β_m(V) = 4·exp(-(V+65)/18). At V=-65: β_m = 4."""
        n = HodgkinHuxleyNeuron()
        bm = n._beta_m(-65.0)
        assert abs(bm - 4.0) < 1e-10

    def test_alpha_h_formula(self):
        """α_h(V) = 0.07·exp(-(V+65)/20). At V=-65: α_h = 0.07."""
        n = HodgkinHuxleyNeuron()
        ah = n._alpha_h(-65.0)
        assert abs(ah - 0.07) < 1e-10

    def test_gating_bounded(self):
        """m, h, n should stay in [0, 1]."""
        n = HodgkinHuxleyNeuron()
        for _ in range(5000):
            n.step(10.0)
        for name, val in [("m", n.m), ("h", n.h), ("n", n.n)]:
            assert -0.01 <= val <= 1.01, f"{name} = {val:.6f}"


class TestHHCurrentBalance:
    def test_i_na_inward_at_rest(self):
        """I_Na at rest: g_Na·m³·h·(V-E_Na). V=-65 < E_Na=50 → negative (inward)."""
        n = HodgkinHuxleyNeuron()
        i_na = n.g_na * n.m**3 * n.h * (n.v - n.e_na)
        assert i_na < 0

    def test_i_k_outward_at_rest(self):
        """I_K: g_K·n⁴·(V-E_K). V=-65 > E_K=-77 → positive (outward)."""
        n = HodgkinHuxleyNeuron()
        i_k = n.g_k * n.n**4 * (n.v - n.e_k)
        assert i_k > 0


class TestHHTypeIIExcitability:
    """HH is the canonical Type-II excitable model."""

    def test_subthreshold_silent(self):
        n = HodgkinHuxleyNeuron()
        assert len(_run(n, current=5.0, steps=5000)) <= 2

    def test_suprathreshold_fires(self):
        n = HodgkinHuxleyNeuron()
        assert len(_run(n, current=10.0, steps=5000)) >= 100

    def test_non_monotonic_fi(self):
        """Type-II: f–I peaks then declines at high current."""
        rates = {}
        for I in [10.0, 20.0, 50.0]:
            n = HodgkinHuxleyNeuron()
            rates[I] = len(_run(n, current=I, steps=5000))
        # Rate should peak at moderate I, decline at I=50
        assert rates[20.0] > rates[50.0], (
            f"f(20)={rates[20.0]}, f(50)={rates[50.0]} — expected non-monotonic"
        )

    def test_isi_regularity(self):
        """HH ISI has moderate variability from the 4-variable interaction.

        CV at I=10 measured ~0.26 — higher than simple IF models but
        still structured (not random). The variability comes from the
        interplay between fast Na (m,h) and slow K (n) gating.
        """
        n = HodgkinHuxleyNeuron()
        spikes = _run(n, current=10.0, steps=5000)
        if len(spikes) >= 10:
            isis = np.diff(spikes[3:]).astype(float)
            cv = np.std(isis) / np.mean(isis)
            assert cv < 0.5, f"CV(ISI) = {cv:.4f}"


class TestHHParameters:
    @pytest.mark.parametrize("dt", [0.005, 0.01, 0.02])
    def test_dt_stability(self, dt: float):
        n = HodgkinHuxleyNeuron(dt=dt)
        for _ in range(2000):
            n.step(10.0)
        assert np.isfinite(n.v)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = HodgkinHuxleyNeuron()
            trace = [(n.step(10.0), n.v) for _ in range(100)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestHHPerformance:
    def test_isolation_throughput(self):
        """HH is slow due to 100 sub-steps + exp() per step."""
        n = HodgkinHuxleyNeuron()
        N = 500
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(10.0)
        elapsed = time.perf_counter() - t0
        steps_per_s = N / elapsed
        # Expected ~670 steps/s; just verify it's > 100
        assert steps_per_s > 100, f"{steps_per_s:.0f} steps/s"


class TestHHPipeline:
    def test_population(self):
        assert Population(HodgkinHuxleyNeuron, n=5, label="hh").n == 5

    def test_network_spikes(self):
        pop = Population(HodgkinHuxleyNeuron, n=3, label="hh")
        drive = PoissonInput(n=3, rate_hz=500.0, weight=10.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(HodgkinHuxleyNeuron, n=3, label="src")
        tgt = Population(HodgkinHuxleyNeuron, n=3, label="tgt")
        drive = PoissonInput(n=3, rate_hz=500.0, weight=10.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=5.0, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self):
        n = HodgkinHuxleyNeuron()
        train = np.array([float(n.step(10.0)) for _ in range(5000)])
        sc = spike_count(train)
        assert sc >= 50
        rate = firing_rate(train, dt=0.001)  # 1 ms per step (100 substeps × 0.01)
        assert rate > 0


class TestHodgkinHuxleyNeuronSimulate:
    """Engineering-verification surface for ``HodgkinHuxleyNeuron.simulate``."""

    def test_simulate_python_returns_finite_trace(self) -> None:
        n = HodgkinHuxleyNeuron()
        trace, spikes = n.simulate(1000, current=10.0, backend="python")
        assert trace.shape == (1000,)
        assert np.all(np.isfinite(trace))
        assert spikes >= 1

    def test_simulate_rust_matches_or_ulp_python(self) -> None:
        pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built")
        py = HodgkinHuxleyNeuron()
        rs = HodgkinHuxleyNeuron()
        tr_py, sp_py = py.simulate(1000, current=10.0, backend="python")
        tr_rs, sp_rs = rs.simulate(1000, current=10.0, backend="rust")
        assert sp_py == sp_rs
        max_diff = float(np.max(np.abs(tr_py - tr_rs)))
        assert max_diff < 1e-9

    def test_simulate_rust_rejects_non_default(self) -> None:
        pytest.importorskip("sc_neurocore_engine", reason="Rust engine not built")
        # force non-default via a constructor override that every model accepts
        try:
            n = (
                HodgkinHuxleyNeuron(dt=0.02)
                if "dt" in HodgkinHuxleyNeuron.__dataclass_fields__
                else HodgkinHuxleyNeuron()
            )
            if "dt" not in HodgkinHuxleyNeuron.__dataclass_fields__:
                pytest.skip("no dt field")
        except TypeError:
            pytest.skip("cannot override defaults")
        with pytest.raises(RuntimeError, match="factory-default"):
            n.simulate(10, current=0.0, backend="rust")
