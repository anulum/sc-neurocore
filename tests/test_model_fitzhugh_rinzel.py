# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: FitzHughRinzelNeuron

"""Full pipeline test for FitzHughRinzelNeuron (FitzHugh 1976 / Rinzel 1987).

3D FHN + ultra-slow y for bursting. Three timescales: v (fast), w (delta=0.08),
y (mu=0.0001). Performance: ~447K isolation steps/s. FULL PIPELINE."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.fitzhugh_rinzel import FitzHughRinzelNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate


def _run(neuron: FitzHughRinzelNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestFHRIsolation:
    def test_defaults(self):
        n = FitzHughRinzelNeuron()
        assert n.v == -1.0 and n.w == -0.5 and n.y == 0.0
        assert n.delta == 0.08 and n.mu == 0.0001

    def test_step_returns_binary(self):
        assert FitzHughRinzelNeuron().step(0.0) in (0, 1)

    def test_three_variables_evolve(self):
        n = FitzHughRinzelNeuron()
        initial = (n.v, n.w, n.y)
        for _ in range(1000):
            n.step(0.5)
        for name, v0, v1 in zip(["v", "w", "y"], initial, (n.v, n.w, n.y)):
            assert v0 != v1, f"{name} didn't evolve"

    def test_state_finite(self):
        n = FitzHughRinzelNeuron()
        for _ in range(100000):
            n.step(0.5)
        assert np.isfinite(n.v) and np.isfinite(n.w) and np.isfinite(n.y)

    def test_reset(self):
        n = FitzHughRinzelNeuron()
        for _ in range(500):
            n.step(0.5)
        n.reset()
        assert n.v == -1.0 and n.w == -0.5 and n.y == 0.0


class TestFHRThreeTimescales:
    def test_y_ultra_slow(self):
        """mu=0.0001 → y changes ~800× slower than w (delta=0.08)."""
        n = FitzHughRinzelNeuron()
        w0, y0 = n.w, n.y
        for _ in range(100):
            n.step(0.5)
        dw = abs(n.w - w0)
        dy = abs(n.y - y0)
        assert dw > 100 * dy, f"dw={dw:.6f}, dy={dy:.6f}"

    def test_y_modulates_oscillation(self):
        """Different c values (y-nullcline offset) change dynamics."""
        n1 = FitzHughRinzelNeuron(c=-0.5)
        n2 = FitzHughRinzelNeuron(c=-1.0)
        s1 = len(_run(n1, current=0.5, steps=10000))
        s2 = len(_run(n2, current=0.5, steps=10000))
        assert s1 != s2


class TestFHRDynamics:
    def test_dv_formula(self):
        """dv = (v - v³/3 - w + y + I) · dt."""
        n = FitzHughRinzelNeuron()
        v0, w0, y0 = n.v, n.w, n.y
        I = 0.5
        expected_dv = (v0 - v0**3 / 3 - w0 + y0 + I) * n.dt
        n.step(I)
        actual_dv = n.v - v0
        assert abs(actual_dv - expected_dv) < 0.01

    def test_oscillates_at_moderate_I(self):
        n = FitzHughRinzelNeuron()
        spikes = _run(n, current=0.5, steps=10000)
        assert len(spikes) >= 5

    def test_v_bounded(self):
        n = FitzHughRinzelNeuron()
        vs = [n.v]
        for _ in range(10000):
            n.step(0.5)
            vs.append(n.v)
        assert min(vs) > -3 and max(vs) < 3

    def test_isi_regularity(self):
        n = FitzHughRinzelNeuron()
        spikes = _run(n, current=0.5, steps=10000)
        if len(spikes) >= 5:
            isis = np.diff(spikes[2:]).astype(float)
            cv = np.std(isis) / np.mean(isis)
            assert cv < 0.3


class TestFHRParameters:
    def test_mu_controls_y_speed(self):
        n_fast = FitzHughRinzelNeuron(mu=0.01)
        n_slow = FitzHughRinzelNeuron(mu=0.00001)
        for _ in range(5000):
            n_fast.step(0.5)
            n_slow.step(0.5)
        assert abs(n_fast.y) > abs(n_slow.y)

    @pytest.mark.parametrize("dt", [0.05, 0.1, 0.2])
    def test_dt_stability(self, dt: float):
        n = FitzHughRinzelNeuron(dt=dt)
        for _ in range(10000):
            n.step(0.5)
        assert np.isfinite(n.v)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = FitzHughRinzelNeuron()
            trace = [(n.step(0.5), n.v, n.w, n.y) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestFHRPerformance:
    def test_isolation_throughput(self):
        samples = []
        for _ in range(3):
            n = FitzHughRinzelNeuron()
            steps = 50_000
            t0 = time.perf_counter()
            for _ in range(steps):
                n.step(0.5)
            samples.append(time.perf_counter() - t0)

        best_seconds_per_step = min(samples) / steps
        assert best_seconds_per_step < 20e-6

    def test_network_throughput(self):
        pop = Population(FitzHughRinzelNeuron, n=50, label="bench")
        drive = PoissonInput(n=50, rate_hz=500.0, weight=0.8, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        t0 = time.perf_counter()
        net.run(duration=0.5, dt=0.001, backend="python")
        elapsed = time.perf_counter() - t0
        assert 50 * 500 / elapsed > 5000


class TestFHRPipeline:
    def test_population(self):
        assert Population(FitzHughRinzelNeuron, n=10, label="fhr").n == 10

    def test_network_spikes(self):
        pop = Population(FitzHughRinzelNeuron, n=10, label="fhr")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=0.8, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(FitzHughRinzelNeuron, n=5, label="src")
        tgt = Population(FitzHughRinzelNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=0.8, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=0.5, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=5.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis_pipeline(self):
        n = FitzHughRinzelNeuron()
        train = np.array([float(n.step(0.5)) for _ in range(10000)])
        sc = spike_count(train)
        assert sc >= 3
        rate = firing_rate(train, dt=0.0001)
        assert rate > 0


def test_fitzhugh_rinzel_rejects_invalid_numeric_configuration() -> None:
    with pytest.raises(ValueError, match="dt.*positive"):
        FitzHughRinzelNeuron(dt=0.0)

    with pytest.raises(ValueError, match="delta.*positive"):
        FitzHughRinzelNeuron(delta=-0.1)


def test_fitzhugh_rinzel_rejects_nonfinite_current_without_mutation() -> None:
    neuron = FitzHughRinzelNeuron(v=-1.0, w=0.2, y=0.1)
    before = (neuron.v, neuron.w, neuron.y)

    with pytest.raises(FloatingPointError, match="runtime state and current"):
        neuron.step(float("nan"))

    assert (neuron.v, neuron.w, neuron.y) == before


def test_fitzhugh_rinzel_rejects_corrupted_runtime_parameter_without_mutation() -> None:
    neuron = FitzHughRinzelNeuron(v=-1.0, w=0.2, y=0.1)
    before = (neuron.v, neuron.w, neuron.y)
    neuron.mu = float("nan")

    with pytest.raises(ValueError, match="mu.*finite"):
        neuron.step(0.5)

    assert (neuron.v, neuron.w, neuron.y) == before


def test_fitzhugh_rinzel_rejects_overflow_candidate_without_mutation() -> None:
    neuron = FitzHughRinzelNeuron(v=1.0e155, w=0.2, y=0.1)
    before = (neuron.v, neuron.w, neuron.y)

    with pytest.raises(FloatingPointError, match="derivative overflow"):
        neuron.step(0.5)

    assert (neuron.v, neuron.w, neuron.y) == before
