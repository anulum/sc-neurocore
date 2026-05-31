# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: BrunelWangNeuron

"""Full pipeline test for BrunelWangNeuron.

Performance: ~333K steps/s. Fires at I≥1 (~477 spikes/10k).
Full pipeline wired."""

from __future__ import annotations

import time

import numpy as np
import pytest

from sc_neurocore.neurons.models.brunel_wang import BrunelWangNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate


def _run(neuron: BrunelWangNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


class TestBrunelWangIsolation:
    def test_step_returns_binary(self):
        assert BrunelWangNeuron().step(0.0) in (0, 1)

    def test_state_finite(self):
        n = BrunelWangNeuron()
        for _ in range(10000):
            n.step(1.0)
        assert np.isfinite(n.v)

    def test_reset(self):
        n = BrunelWangNeuron()
        for _ in range(100):
            n.step(1.0)
        n.reset()
        assert np.isfinite(n.v)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("v", np.nan),
            ("tau_m", 0.0),
            ("tau_ref", 0.0),
            ("tau_ampa", 0.0),
            ("tau_nmda_rise", 0.0),
            ("tau_nmda_decay", 0.0),
            ("tau_gaba", 0.0),
            ("g_ampa_ext", -1.0),
            ("g_nmda", -1.0),
            ("C_m", 0.0),
            ("mg_conc", -1.0),
            ("dt", 0.0),
        ],
    )
    def test_rejects_invalid_numerical_configuration(self, field: str, value: float):
        with pytest.raises((ValueError, FloatingPointError)):
            BrunelWangNeuron(**{field: value})

    def test_rejects_invalid_synaptic_input_before_state_mutation(self):
        n = BrunelWangNeuron()
        before = (n.v, n.get_state()["ref_remaining"])
        with pytest.raises(ValueError, match="s_nmda_rec"):
            n.step(1.0, s_nmda_rec=np.inf)
        assert (n.v, n.get_state()["ref_remaining"]) == before

    def test_rejects_corrupted_runtime_state_before_mutation(self):
        n = BrunelWangNeuron()
        n.v = np.inf
        before = (n.v, n.get_state()["ref_remaining"])
        with pytest.raises(FloatingPointError, match="voltage state"):
            n.step(1.0)
        assert (n.v, n.get_state()["ref_remaining"]) == before

    def test_nmda_voltage_factor_saturates_for_extreme_negative_voltage(self):
        n = BrunelWangNeuron()
        assert n._nmda_voltage_dep(-1.0e6) == 0.0


class TestBrunelWangDynamics:
    def test_subthreshold_silent(self):
        n = BrunelWangNeuron()
        assert len(_run(n, current=0.0, steps=10000)) == 0

    def test_suprathreshold_fires(self):
        n = BrunelWangNeuron()
        assert len(_run(n, current=1.0, steps=10000)) >= 100

    def test_isi_regularity(self):
        n = BrunelWangNeuron()
        spikes = _run(n, current=1.0, steps=10000)
        if len(spikes) >= 20:
            isis = np.diff(spikes[5:]).astype(float)
            cv = np.std(isis) / np.mean(isis)
            assert cv < 0.3

    def test_deterministic(self):
        traces = []
        for _ in range(2):
            n = BrunelWangNeuron()
            trace = [(n.step(1.0), n.v) for _ in range(200)]
            traces.append(trace)
        assert traces[0] == traces[1]


class TestBrunelWangPerformance:
    def test_isolation_throughput(self):
        n = BrunelWangNeuron()
        N = 50000
        t0 = time.perf_counter()
        for _ in range(N):
            n.step(1.0)
        elapsed = time.perf_counter() - t0
        assert N / elapsed > 50000


class TestBrunelWangPipeline:
    def test_population(self):
        assert Population(BrunelWangNeuron, n=10, label="bw").n == 10

    def test_network_spikes(self):
        pop = Population(BrunelWangNeuron, n=10, label="bw")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_projection_wiring(self):
        src = Population(BrunelWangNeuron, n=5, label="src")
        tgt = Population(BrunelWangNeuron, n=5, label="tgt")
        drive = PoissonInput(n=5, rate_hz=500.0, weight=2.0, dt=0.001, seed=42)
        proj = Projection(src, tgt, weight=1.0, probability=1.0, seed=42)
        mon = SpikeMonitor(src)
        net = Network(src, tgt, drive, proj, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0

    def test_analysis(self):
        n = BrunelWangNeuron()
        train = np.array([float(n.step(1.0)) for _ in range(10000)])
        sc = spike_count(train)
        assert sc >= 50
        rate = firing_rate(train, dt=0.001)
        assert rate > 0
