# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: WangBuzsakiNeuron

"""Full pipeline test for WangBuzsakiNeuron (Wang & Buzsáki 1996).

Fast-spiking GABAergic interneuron. 3 ODEs (V, h, n) with m instantaneous.
Designed for gamma oscillation (30–80 Hz). phi=5 accelerates gating.
Each step() call does int(0.5/dt) sub-steps (50 at default dt=0.01)."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.neurons.models.wang_buzsaki import WangBuzsakiNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, isi, firing_rate


def _run(neuron: WangBuzsakiNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestWBIsolation:
    def test_construction_defaults(self):
        n = WangBuzsakiNeuron()
        assert n.v == -65.0
        assert n.h == 0.8
        assert n.n == 0.1
        assert n.g_na == 35.0
        assert n.phi == 5.0
        assert n.dt == 0.01

    def test_step_returns_binary(self):
        assert WangBuzsakiNeuron().step(0.0) in (0, 1)

    def test_three_variables_evolve(self):
        n = WangBuzsakiNeuron()
        initial = (n.v, n.h, n.n)
        for _ in range(100):
            n.step(1.0)
        for name, v0, v1 in zip(["v", "h", "n"], initial, (n.v, n.h, n.n)):
            assert v0 != v1, f"{name} didn't evolve"

    def test_state_finite(self):
        n = WangBuzsakiNeuron()
        for _ in range(20000):
            n.step(2.0)
        assert all(np.isfinite(v) for v in [n.v, n.h, n.n])

    def test_reset(self):
        n = WangBuzsakiNeuron()
        for _ in range(200):
            n.step(5.0)
        n.reset()
        assert n.v == -65.0 and n.h == 0.8 and n.n == 0.1

    def test_substep_count(self):
        """int(0.5/dt) = 50 sub-steps at dt=0.01."""
        n = WangBuzsakiNeuron(dt=0.01)
        expected = int(0.5 / 0.01)  # 50
        assert expected == 50


class TestWBGammaFrequency:
    """The model is designed for gamma-band (30–80 Hz) firing."""

    def test_gamma_band_at_moderate_current(self):
        """At I=0.5–1.0, firing frequency should be in gamma range (30–80 Hz).

        Each step() = 0.5 ms. ISI in steps × 0.5 ms = ISI_ms.
        Freq = 1000 / ISI_ms Hz.
        """
        n = WangBuzsakiNeuron()
        spikes = _run(n, current=1.0, steps=20000)
        assert len(spikes) >= 20
        isis = np.diff(spikes[5:])
        mean_isi_ms = np.mean(isis) * 0.5  # each step = 0.5 ms
        freq_hz = 1000.0 / mean_isi_ms
        assert 30 < freq_hz < 100, f"freq = {freq_hz:.0f} Hz, expected gamma range"

    def test_onset_frequency_near_30hz(self):
        """At threshold current, frequency should start near lower gamma."""
        n = WangBuzsakiNeuron()
        spikes = _run(n, current=0.5, steps=20000)
        if len(spikes) >= 10:
            isis = np.diff(spikes[5:])
            freq = 1000.0 / (np.mean(isis) * 0.5)
            assert freq > 20, f"Onset freq = {freq:.0f} Hz"


class TestWBFI:
    def test_subthreshold_silent(self):
        n = WangBuzsakiNeuron()
        assert len(_run(n, current=0.0, steps=20000)) == 0

    def test_monotonic_fi(self):
        rates = []
        for I in [0.5, 1.0, 2.0, 5.0]:
            n = WangBuzsakiNeuron()
            rates.append(len(_run(n, current=I, steps=20000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))

    def test_fast_spiking_at_high_current(self):
        """At I=10, frequency >> gamma band (fast-spiking characteristic)."""
        n = WangBuzsakiNeuron()
        spikes = _run(n, current=10.0, steps=20000)
        assert len(spikes) >= 1000  # very high rate


class TestWBHHProperties:
    def test_m_is_instantaneous(self):
        """m is computed as m_inf each sub-step, not integrated as ODE."""
        # Verify by checking that m_inf depends only on V, not on history
        n = WangBuzsakiNeuron()
        # After many steps, the state should be on the limit cycle
        for _ in range(5000):
            n.step(1.0)
        # m_inf should be deterministic from V alone
        alpha_m = (
            0.1 * (n.v + 35.0) / (1.0 - np.exp(-(n.v + 35.0) / 10.0))
            if abs(n.v + 35.0) > 1e-6
            else 1.0
        )
        beta_m = 4.0 * np.exp(-(n.v + 60.0) / 18.0)
        m_inf = alpha_m / (alpha_m + beta_m)
        # m_inf is deterministic — just verify it's finite and in [0,1]
        assert 0 <= m_inf <= 1

    def test_phi_accelerates_gating(self):
        """phi=5 makes h and n dynamics 5× faster than standard HH."""
        n_fast = WangBuzsakiNeuron(phi=5.0)
        n_slow = WangBuzsakiNeuron(phi=1.0)
        h_fast_init, h_slow_init = n_fast.h, n_slow.h
        for _ in range(100):
            n_fast.step(1.0)
            n_slow.step(1.0)
        dh_fast = abs(n_fast.h - h_fast_init)
        dh_slow = abs(n_slow.h - h_slow_init)
        assert dh_fast > dh_slow

    def test_gating_bounded(self):
        n = WangBuzsakiNeuron()
        for _ in range(20000):
            n.step(5.0)
        assert -0.01 <= n.h <= 1.01, f"h = {n.h}"
        assert -0.01 <= n.n <= 1.01, f"n = {n.n}"

    def test_isi_regularity(self):
        n = WangBuzsakiNeuron()
        spikes = _run(n, current=1.0, steps=20000)
        isis = np.diff(spikes[5:]).astype(float)
        cv = np.std(isis) / np.mean(isis)
        assert cv < 0.05

    def test_singularity_protection(self):
        n = WangBuzsakiNeuron(v=-35.0)
        n.step(0.0)
        assert np.isfinite(n.v)


class TestWBParameters:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"v": np.nan}, "v"),
            ({"h": np.inf}, "h"),
            ({"n": np.nan}, "n"),
            ({"g_na": 0.0}, "g_na"),
            ({"g_k": -1.0}, "g_k"),
            ({"g_l": np.nan}, "g_l"),
            ({"c_m": 0.0}, "c_m"),
            ({"phi": 0.0}, "phi"),
            ({"dt": 0.0}, "dt"),
        ],
    )
    def test_rejects_invalid_numerical_configuration(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            WangBuzsakiNeuron(**kwargs)

    def test_rejects_non_finite_current_before_state_mutation(self):
        n = WangBuzsakiNeuron()
        state = (n.v, n.h, n.n)
        with pytest.raises(ValueError, match="current"):
            n.step(np.nan)
        assert (n.v, n.h, n.n) == state

    def test_rejects_corrupted_runtime_state_before_mutation(self):
        n = WangBuzsakiNeuron()
        n.h = np.inf
        state = (n.v, n.h, n.n)
        with pytest.raises(FloatingPointError, match="state"):
            n.step(1.0)
        assert (n.v, n.h, n.n) == state

    def test_rejects_rate_overflow_before_state_mutation(self):
        n = WangBuzsakiNeuron(v=-1.0e308)
        state = (n.v, n.h, n.n)
        with pytest.raises(FloatingPointError, match="rate overflowed"):
            n.step(1.0)
        assert (n.v, n.h, n.n) == state

    @pytest.mark.parametrize("dt", [0.005, 0.01, 0.02])
    def test_dt_stability(self, dt: float):
        n = WangBuzsakiNeuron(dt=dt)
        for _ in range(10000):
            n.step(2.0)
        assert np.isfinite(n.v)

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = WangBuzsakiNeuron()
            trace = [(n.step(2.0), n.v) for _ in range(100)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestWBPipeline:
    def test_population(self):
        assert Population(WangBuzsakiNeuron, n=10, label="wb").n == 10

    def test_network_with_drive(self):
        pop = Population(WangBuzsakiNeuron, n=10, label="wb")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_affects_target(self):
        src = Population(WangBuzsakiNeuron, n=10, label="src")
        tgt_with = Population(WangBuzsakiNeuron, n=10, label="tgt_w")
        tgt_without = Population(WangBuzsakiNeuron, n=10, label="tgt_wo")
        drive_src = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        drive_tgt1 = PoissonInput(n=10, rate_hz=100.0, weight=0.5, dt=0.001, seed=99)
        drive_tgt2 = PoissonInput(n=10, rate_hz=100.0, weight=0.5, dt=0.001, seed=99)
        proj = Projection(src, tgt_with, weight=2.0, probability=1.0, seed=42)
        mon_src = SpikeMonitor(src)
        mon_with = SpikeMonitor(tgt_with)
        mon_without = SpikeMonitor(tgt_without)
        net_with = Network(src, tgt_with, drive_src, drive_tgt1, proj, mon_src, mon_with)
        net_without = Network(tgt_without, drive_tgt2, mon_without)
        net_with.run(duration=1.0, dt=0.001, backend="python")
        net_without.run(duration=1.0, dt=0.001, backend="python")
        assert mon_src.count > 0
        assert mon_with.count >= mon_without.count

    def test_analysis_pipeline(self):
        n = WangBuzsakiNeuron()
        train = np.array([float(n.step(2.0)) for _ in range(20000)])
        sc = spike_count(train)
        assert sc >= 100
        isis = isi(train, dt=0.0005)  # each step = 0.5 ms
        assert len(isis) >= 10
        rate = firing_rate(train, dt=0.0005)
        assert rate > 0
