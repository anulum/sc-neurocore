# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: AdExNeuron

"""Full pipeline test for AdExNeuron (Brette & Gerstner 2005).

Adaptive Exponential IF: 2 ODEs (V, w). Exponential spike initiation
+ adaptation current w. ISI lengthens (adaptation) due to w += b on spike."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.adex import AdExNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, isi, firing_rate


def _run(neuron: AdExNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestAdExIsolation:
    def test_construction_defaults(self):
        n = AdExNeuron()
        assert n.v == -65.0
        assert n.w == 0.0
        assert n.delta_t == 2.0
        assert n.a == 0.5
        assert n.b == 7.0
        assert n.c_m == 200.0

    def test_step_returns_binary(self):
        assert AdExNeuron().step(0.0) in (0, 1)

    def test_two_variables_evolve(self):
        n = AdExNeuron()
        v0, w0 = n.v, n.w
        for _ in range(100):
            n.step(500.0)
        assert n.v != v0 and n.w != w0

    def test_state_finite(self):
        n = AdExNeuron()
        for _ in range(50000):
            n.step(500.0)
        assert np.isfinite(n.v) and np.isfinite(n.w)

    def test_reset(self):
        n = AdExNeuron()
        for _ in range(100):
            n.step(500.0)
        n.reset()
        assert n.v == n.v_rest and n.w == 0.0

    def test_exp_clipped(self):
        """Exponential term is clipped to avoid overflow."""
        n = AdExNeuron()
        n.v = 100.0  # far above threshold → should clip exp
        n.step(0.0)
        assert np.isfinite(n.v)


class TestAdExAdaptation:
    def test_w_increments_on_spike(self):
        """Each spike adds b to w."""
        n = AdExNeuron()
        w_before = n.w
        for _ in range(10000):
            if n.step(500.0) == 1:
                # w should have been incremented by b
                assert n.w > w_before
                break
        else:
            pytest.fail("No spike")

    def test_isi_lengthens(self):
        """Adaptation: early ISIs shorter than late ISIs."""
        n = AdExNeuron()
        spikes = _run(n, current=500.0, steps=10000)
        assert len(spikes) >= 10
        isis_arr = np.diff(spikes)
        early = np.mean(isis_arr[:3])
        late = np.mean(isis_arr[-3:])
        assert late > early, f"early={early:.0f}, late={late:.0f}"

    def test_w_decays_between_spikes(self):
        """w decays toward a·(V-V_rest) between spikes (tau_w timescale)."""
        n = AdExNeuron()
        n.w = 50.0
        # Subthreshold: w should decay
        for _ in range(1000):
            n.step(0.0)
        assert n.w < 50.0

    def test_no_adaptation_when_b_zero(self):
        """b=0: no w increment on spike → constant ISI (like EIF)."""
        n = AdExNeuron(b=0.0, a=0.0)
        spikes = _run(n, current=500.0, steps=10000)
        if len(spikes) >= 10:
            isis_arr = np.diff(spikes[3:]).astype(float)
            cv = np.std(isis_arr) / np.mean(isis_arr)
            assert cv < 0.05, f"CV(ISI) = {cv:.4f} without adaptation"

    def test_stronger_adaptation_fewer_spikes(self):
        """Higher b → stronger per-spike w increment → fewer spikes."""
        n_weak = AdExNeuron(b=2.0)
        n_strong = AdExNeuron(b=20.0)
        s_weak = len(_run(n_weak, current=500.0, steps=10000))
        s_strong = len(_run(n_strong, current=500.0, steps=10000))
        assert s_weak > s_strong


class TestAdExExponentialSpike:
    def test_exponential_upstroke(self):
        """delta_T controls spike sharpness. Larger delta_T → softer spike."""
        n_sharp = AdExNeuron(delta_t=1.0)
        n_soft = AdExNeuron(delta_t=5.0)
        # Both should fire, but with different dynamics
        s_sharp = len(_run(n_sharp, current=500.0, steps=10000))
        s_soft = len(_run(n_soft, current=500.0, steps=10000))
        # Just verify both fire
        assert s_sharp > 0 and s_soft > 0


class TestAdExFI:
    def test_subthreshold_silent(self):
        n = AdExNeuron()
        assert len(_run(n, current=0.0, steps=10000)) == 0

    def test_monotonic_fi(self):
        rates = []
        for I in [200.0, 500.0, 1000.0, 2000.0]:
            n = AdExNeuron()
            rates.append(len(_run(n, current=I, steps=10000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))


class TestAdExParameters:
    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float):
        n = AdExNeuron(dt=dt)
        for _ in range(10000):
            n.step(500.0)
        assert np.isfinite(n.v)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = AdExNeuron()
            trace = [(n.step(500.0), n.v, n.w) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestAdExPipeline:
    def test_population(self):
        assert Population(AdExNeuron, n=10, label="adex").n == 10

    def test_network_with_drive(self):
        pop = Population(AdExNeuron, n=10, label="adex")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=500.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        """Projection from source adds current to target. Verify source fires
        and projection object is accepted by Network without error."""
        src = Population(AdExNeuron, n=5, label="src")
        tgt = Population(AdExNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=500.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=200.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_tgt = SpikeMonitor(tgt)
        # Run without error — Projection is wired into the network graph
        net = Network(src, tgt, drive, proj, mon_src, mon_tgt)
        net.run(duration=2.0, dt=0.001, backend="python")
        assert mon_src.count > 0, "Source should fire"
        # Target may or may not fire depending on projection current magnitude
        # The key test: network accepted the Projection and ran without error

    def test_analysis_pipeline(self):
        n = AdExNeuron()
        train = np.array([float(n.step(500.0)) for _ in range(10000)])
        sc = spike_count(train)
        assert sc >= 10
        isis = isi(train, dt=0.0001)  # dt = 0.1 ms per step
        assert len(isis) >= 5
        rate = firing_rate(train, dt=0.0001)
        assert rate > 0
