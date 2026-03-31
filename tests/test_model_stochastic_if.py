# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: StochasticIFNeuron

"""Full pipeline test for StochasticIFNeuron (Brunel & Hakim 1999).

LIF with Ornstein-Uhlenbeck noise: dV/dt = (-(V-V_rest) + mu + I)/tau_m + sigma·ξ.
sigma=0 → deterministic LIF (CV=0). sigma>0 → stochastic ISI variability.
Noise enables subthreshold spike triggering."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.stochastic_if import StochasticIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, isi, firing_rate


def _run(neuron: StochasticIFNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestStochasticIFIsolation:
    def test_construction_defaults(self):
        n = StochasticIFNeuron()
        assert n.v == -70.0
        assert n.sigma == 3.0
        assert n.tau_m == 20.0
        assert n.v_threshold == -50.0
        assert n.dt == 1.0

    def test_step_returns_binary(self):
        assert StochasticIFNeuron().step(0.0) in (0, 1)

    def test_v_evolves_with_noise(self):
        n = StochasticIFNeuron()
        v0 = n.v
        n.step(10.0)
        assert n.v != v0

    def test_state_finite_long_run(self):
        n = StochasticIFNeuron()
        for _ in range(100000):
            n.step(20.0)
        assert np.isfinite(n.v)

    def test_reset(self):
        n = StochasticIFNeuron()
        for _ in range(100):
            n.step(30.0)
        n.reset()
        assert n.v == n.v_rest


class TestStochasticIFNoiseMechanism:
    """Core: OU noise with amplitude sigma·sqrt(dt/tau_m)."""

    def test_sigma_zero_is_deterministic(self):
        """sigma=0 → identical runs (no RNG)."""
        np.random.seed(42)
        n1 = StochasticIFNeuron(sigma=0.0)
        t1 = [(n1.step(25.0), n1.v) for _ in range(200)]
        np.random.seed(42)
        n2 = StochasticIFNeuron(sigma=0.0)
        t2 = [(n2.step(25.0), n2.v) for _ in range(200)]
        assert t1 == t2

    def test_sigma_zero_constant_isi(self):
        """sigma=0: deterministic LIF → perfectly constant ISI."""
        n = StochasticIFNeuron(sigma=0.0)
        spikes = _run(n, current=25.0, steps=10000)
        if len(spikes) > 10:
            isis = np.diff(spikes[2:]).astype(float)
            cv = np.std(isis) / np.mean(isis)
            assert cv < 0.001, f"CV(ISI) = {cv:.4f} with sigma=0"

    def test_sigma_nonzero_variable_isi(self):
        """sigma>0: ISI has variability (CV > 0)."""
        n = StochasticIFNeuron(sigma=3.0)
        spikes = _run(n, current=25.0, steps=50000)
        assert len(spikes) >= 100
        isis = np.diff(spikes).astype(float)
        cv = np.std(isis) / np.mean(isis)
        assert cv > 0.05, f"CV(ISI) = {cv:.4f}, expected > 0.05 with noise"

    def test_noise_amplitude_scales_with_sigma(self):
        """Higher sigma → more ISI variability."""
        cv_low = _measure_cv(StochasticIFNeuron(sigma=1.0), 25.0, 50000)
        cv_high = _measure_cv(StochasticIFNeuron(sigma=5.0), 25.0, 50000)
        if cv_low is not None and cv_high is not None:
            assert cv_high > cv_low

    def test_noise_enables_subthreshold_spiking(self):
        """At I=15 (subthreshold for deterministic), noise (sigma=10) triggers spikes."""
        n_det = StochasticIFNeuron(sigma=0.0)
        n_noisy = StochasticIFNeuron(sigma=10.0)
        s_det = len(_run(n_det, current=15.0, steps=10000))
        s_noisy = len(_run(n_noisy, current=15.0, steps=10000))
        assert s_det == 0, "Deterministic should not spike at I=15"
        assert s_noisy > 10, "Noise should trigger spikes at subthreshold I"

    def test_two_runs_differ(self):
        """Two neurons with same params produce different spike trains."""
        n1 = StochasticIFNeuron()
        n2 = StochasticIFNeuron()
        t1 = [n1.step(25.0) for _ in range(1000)]
        t2 = [n2.step(25.0) for _ in range(1000)]
        assert t1 != t2


class TestStochasticIFFI:
    def test_subthreshold_deterministic_silent(self):
        """I=10 with sigma=0 → no spikes (V_ss = V_rest + I = -60 < -50)."""
        n = StochasticIFNeuron(sigma=0.0)
        assert len(_run(n, current=10.0, steps=10000)) == 0

    def test_suprathreshold_fires(self):
        n = StochasticIFNeuron()
        assert len(_run(n, current=25.0, steps=10000)) >= 50

    def test_rate_increases_with_current(self):
        n20 = StochasticIFNeuron()
        n50 = StochasticIFNeuron()
        s20 = len(_run(n20, current=20.0, steps=50000))
        s50 = len(_run(n50, current=50.0, steps=50000))
        assert s50 > s20

    def test_rate_increases_with_sigma(self):
        """More noise → more noise-driven spikes → higher rate (near threshold)."""
        n_low = StochasticIFNeuron(sigma=1.0)
        n_high = StochasticIFNeuron(sigma=10.0)
        s_low = len(_run(n_low, current=18.0, steps=50000))
        s_high = len(_run(n_high, current=18.0, steps=50000))
        assert s_high > s_low


class TestStochasticIFLIFLimit:
    """At sigma=0, model reduces to standard LIF. Verify LIF properties."""

    def test_lif_membrane_equation(self):
        """dV/dt = (-(V-V_rest) + I) / tau_m. Verify one step."""
        n = StochasticIFNeuron(sigma=0.0, mu=0.0)
        v0 = n.v
        I = 15.0
        n.step(I)
        expected = v0 + (-(v0 - n.v_rest) + I) / n.tau_m * n.dt
        assert abs(n.v - expected) < 1e-10

    def test_lif_steady_state(self):
        """V_ss = V_rest + mu + I. At I=15, sigma=0: V_ss = -55 < threshold."""
        n = StochasticIFNeuron(sigma=0.0, mu=0.0)
        for _ in range(10000):
            n.step(15.0)
        expected_vss = n.v_rest + 15.0  # -55
        assert abs(n.v - expected_vss) < 0.1


class TestStochasticIFParameters:
    def test_tau_m_affects_dynamics(self):
        n_fast = StochasticIFNeuron(tau_m=5.0, sigma=0.0)
        n_slow = StochasticIFNeuron(tau_m=40.0, sigma=0.0)
        s_fast = len(_run(n_fast, current=25.0, steps=10000))
        s_slow = len(_run(n_slow, current=25.0, steps=10000))
        assert s_fast > s_slow

    def test_mu_shifts_baseline(self):
        """mu adds constant offset to the input."""
        n = StochasticIFNeuron(sigma=0.0, mu=10.0)
        # Effective input = mu + I = 10 + 15 = 25, same as I=25 with mu=0
        n2 = StochasticIFNeuron(sigma=0.0, mu=0.0)
        s1 = len(_run(n, current=15.0, steps=10000))
        s2 = len(_run(n2, current=25.0, steps=10000))
        assert s1 == s2

    @pytest.mark.parametrize("dt", [0.5, 1.0, 2.0])
    def test_dt_stability(self, dt: float):
        n = StochasticIFNeuron(dt=dt)
        for _ in range(10000):
            n.step(25.0)
        assert np.isfinite(n.v)


class TestStochasticIFPipeline:
    def test_population(self):
        assert Population(StochasticIFNeuron, n=10, label="sif").n == 10

    def test_network_with_drive(self):
        pop = Population(StochasticIFNeuron, n=10, label="sif")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=25.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_affects_target(self):
        """Verify projection wiring by comparing target spikes with/without projection.

        Both target populations get the same subthreshold drive.
        Only one gets a projection from a firing source.
        """
        src = Population(StochasticIFNeuron, n=20, label="src")
        tgt_with = Population(StochasticIFNeuron, n=20, label="tgt_proj")
        tgt_without = Population(StochasticIFNeuron, n=20, label="tgt_noproj")
        drive_src = PoissonInput(n=20, rate_hz=500.0, weight=30.0, dt=0.001, seed=42)
        drive_tgt1 = PoissonInput(n=20, rate_hz=200.0, weight=15.0, dt=0.001, seed=99)
        drive_tgt2 = PoissonInput(n=20, rate_hz=200.0, weight=15.0, dt=0.001, seed=99)
        proj = Projection(src, tgt_with, weight=50.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_with = SpikeMonitor(tgt_with)
        mon_without = SpikeMonitor(tgt_without)
        net_with = Network(src, tgt_with, drive_src, drive_tgt1, proj, mon_src, mon_with)
        net_without = Network(tgt_without, drive_tgt2, mon_without)
        net_with.run(duration=2.0, dt=0.001, backend="python")
        net_without.run(duration=2.0, dt=0.001, backend="python")
        assert mon_src.count > 0, "Source should fire"
        # Target with projection should fire at least as much as without
        # (projection adds excitatory input)
        assert mon_with.count >= mon_without.count

    def test_analysis_pipeline(self):
        n = StochasticIFNeuron()
        train = np.array([float(n.step(25.0)) for _ in range(50000)])
        sc = spike_count(train)
        assert sc >= 50
        isis = isi(train, dt=0.001)
        assert len(isis) >= 10
        rate = firing_rate(train, dt=0.001)
        assert rate > 0
        duration = 50000 * 0.001
        assert abs(rate - sc / duration) < 10.0


def _measure_cv(neuron: StochasticIFNeuron, current: float, steps: int) -> float | None:
    spikes = _run(neuron, current=current, steps=steps)
    if len(spikes) < 20:
        return None
    isis = np.diff(spikes).astype(float)
    return float(np.std(isis) / np.mean(isis))
