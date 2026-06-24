# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: DurstewitzDopamineNeuron

"""Full pipeline test for DurstewitzDopamineNeuron.

Fires spontaneously at I=0 (~20 spikes/10k). Rate increases with I.
Performance: ~54K steps/s. Full pipeline wired."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.durstewitz_dopamine import DurstewitzDopamineNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate


def _run(neuron: DurstewitzDopamineNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestDurstewitzIsolation:
    def test_step_returns_binary(self):
        assert DurstewitzDopamineNeuron().step(0.0) in (0, 1)

    def test_state_finite(self):
        n = DurstewitzDopamineNeuron()
        for _ in range(50000):
            n.step(10.0)
        assert np.isfinite(n.v)

    def test_reset(self):
        n = DurstewitzDopamineNeuron()
        for _ in range(100):
            n.step(10.0)
        n.reset()
        assert np.isfinite(n.v)


class TestDurstewitzDynamics:
    def test_spontaneous_firing(self):
        """Fires at I=0 (dopaminergic tonic activity)."""
        n = DurstewitzDopamineNeuron()
        spikes = _run(n, current=0.0, steps=10000)
        assert len(spikes) >= 5

    def test_rate_increases_with_current(self):
        n0 = DurstewitzDopamineNeuron()
        n50 = DurstewitzDopamineNeuron()
        s0 = len(_run(n0, current=0.0, steps=10000))
        s50 = len(_run(n50, current=50.0, steps=10000))
        assert s50 > s0

    def test_monotonic_fi(self):
        rates = []
        for I in [0.0, 10.0, 30.0, 50.0]:
            n = DurstewitzDopamineNeuron()
            rates.append(len(_run(n, current=I, steps=10000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = DurstewitzDopamineNeuron()
            trace = [(n.step(10.0), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestDurstewitzPerformance:
    def test_isolation_throughput(self):
        n = DurstewitzDopamineNeuron()
        N = 10000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(10.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 10000


class TestDurstewitzPipeline:
    def test_population(self):
        assert Population(DurstewitzDopamineNeuron, n=5, label="dd").n == 5

    def test_network_spikes(self):
        pop = Population(DurstewitzDopamineNeuron, n=5, label="dd")
        drive = PoissonInput(n=5, rate_hz=200.0, weight=10.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(DurstewitzDopamineNeuron, n=5, label="src")
        tgt = Population(DurstewitzDopamineNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=200.0, weight=10.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=5.0, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self):
        n = DurstewitzDopamineNeuron()
        train = np.array([float(n.step(10.0)) for _ in range(10000)])
        sc = spike_count(train)
        assert sc >= 10
        rate = firing_rate(train, dt=0.001)
        assert rate > 0


# Salvaged model-specific behavioural contracts from retired aggregate test file.
class TestDurstewitzDopamine:
    def test_fires(self):
        from sc_neurocore.neurons.models.durstewitz_dopamine import DurstewitzDopamineNeuron

        n = DurstewitzDopamineNeuron()
        assert sum(n.step(10.0) for _ in range(300)) > 0

    def test_d1_modulation(self):
        from sc_neurocore.neurons.models.durstewitz_dopamine import DurstewitzDopamineNeuron

        n = DurstewitzDopamineNeuron(d1_level=0.8)
        for _ in range(100):
            n.step(8.0)
        assert n.v != -65.0


class TestDurstewitzRK4:
    """Guards the candidate-first RK4 integrator and its cross-backend parity.

    The historical hard-coded forward Euler advanced the gates from the old
    voltage and then the voltage from the freshly updated gates, mixing two
    inconsistent states. The production path is RK4 over ``(v, h_na, n_k)`` with
    one consistent right-hand side per stage; the staggered baseline survives
    only behind ``integrator="baseline_euler"``.
    """

    def test_default_integrator_is_rk4(self):
        assert DurstewitzDopamineNeuron().integrator == "rk4"

    def test_unknown_integrator_rejected(self):
        with pytest.raises(ValueError, match="integrator"):
            DurstewitzDopamineNeuron(integrator="midpoint")  # type: ignore[arg-type]

    def test_baseline_euler_path_runs_and_fires(self):
        n = DurstewitzDopamineNeuron(integrator="baseline_euler")
        assert sum(n.step(10.0) for _ in range(10000)) > 0

    def test_rk4_and_baseline_differ(self):
        rk4 = DurstewitzDopamineNeuron()
        euler = DurstewitzDopamineNeuron(integrator="baseline_euler")
        rk4_v = [round(rk4.step(10.0) or rk4.v, 9) for _ in range(500)]
        euler_v = [round(euler.step(10.0) or euler.v, 9) for _ in range(500)]
        assert rk4_v != euler_v

    def test_cross_backend_spike_anchor(self):
        # Pins the Python reference the Rust/Julia/Go/Mojo kernels reproduce
        # bit-for-bit (verified by benchmarks/bench_model_durstewitz_dopamine.py).
        n = DurstewitzDopamineNeuron()
        assert sum(n.step(10.0) for _ in range(200000)) == 925

    def test_monotone_fi_anchors(self):
        counts = []
        for current in (0.0, 10.0, 30.0, 50.0):
            n = DurstewitzDopamineNeuron()
            counts.append(sum(n.step(current) for _ in range(10000)))
        assert counts == sorted(counts)
        assert counts[0] >= 5  # spontaneous dopaminergic tone

    def test_non_finite_current_rejected(self):
        n = DurstewitzDopamineNeuron()
        with pytest.raises(ValueError, match="current"):
            n.step(math.inf)

    def test_non_finite_state_rejected_on_step(self):
        n = DurstewitzDopamineNeuron()
        n.v = math.nan
        with pytest.raises(ValueError, match="v"):
            n.step(10.0)

    def test_negative_conductance_rejected(self):
        with pytest.raises(ValueError, match="g_na"):
            DurstewitzDopamineNeuron(g_na=-1.0)

    def test_non_positive_dt_rejected(self):
        with pytest.raises(ValueError, match="dt"):
            DurstewitzDopamineNeuron(dt=0.0)

    def test_mg_block_high_at_rest(self):
        n = DurstewitzDopamineNeuron()
        assert n.mg_block(-65.0) < 0.1

    def test_mg_block_relieved_when_depolarised(self):
        n = DurstewitzDopamineNeuron()
        assert n.mg_block(0.0) > n.mg_block(-65.0)
