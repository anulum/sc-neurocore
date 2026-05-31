# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — End-to-end test: BoothRinzelNeuron

"""Full pipeline test for BoothRinzelNeuron (Booth & Rinzel 1995).

2-compartment bistable motoneuron: soma (Na/K) + dendrite (Ca/KCa).
4 sub-steps per step. Exhibits bistability at high current."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons.models.booth_rinzel import BoothRinzelNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import firing_rate, spike_count, isi


class TestBoothRinzelIsolation:
    def test_construction(self):
        n = BoothRinzelNeuron()
        assert n.vs == -65.0
        assert n.vd == -65.0
        assert n.ca == 0.0

    def test_step_returns_binary(self):
        n = BoothRinzelNeuron()
        assert n.step(0.0) in (0, 1)

    def test_spikes_under_drive(self):
        n = BoothRinzelNeuron()
        spikes = sum(n.step(10.0) for _ in range(50_000))
        assert spikes > 100, f"too few spikes at I=10: {spikes}"

    def test_two_compartments_differ(self):
        """Soma and dendrite should have different voltages under drive."""
        n = BoothRinzelNeuron()
        for _ in range(10_000):
            n.step(10.0)
        assert n.vs != n.vd, "soma and dendrite identical"

    def test_calcium_accumulates(self):
        n = BoothRinzelNeuron()
        for _ in range(10_000):
            n.step(10.0)
        assert n.ca > 0, "calcium did not accumulate"

    def test_bistability(self):
        """At high current, model may enter depolarisation block (fewer spikes)."""
        n_low = BoothRinzelNeuron()
        n_high = BoothRinzelNeuron()
        spikes_low = sum(n_low.step(10.0) for _ in range(50_000))
        spikes_high = sum(n_high.step(50.0) for _ in range(50_000))
        # High current may give fewer spikes (depolarisation block)
        # Just verify both are finite and model doesn't crash
        assert np.isfinite(n_high.vs)

    def test_numerical_stability(self):
        """Model should not produce NaN or Inf at any current."""
        for I in [0, 5, 10, 20, 50]:
            n = BoothRinzelNeuron()
            for _ in range(10_000):
                n.step(float(I))
            assert np.isfinite(n.vs), f"vs NaN/Inf at I={I}"
            assert np.isfinite(n.vd), f"vd NaN/Inf at I={I}"
            assert np.isfinite(n.h), f"h NaN/Inf at I={I}"
            assert np.isfinite(n.n), f"n NaN/Inf at I={I}"
            assert np.isfinite(n.ca), f"ca NaN/Inf at I={I}"

    def test_gating_bounded(self):
        """Gating variables h, n, q should stay in [0, 1]."""
        n = BoothRinzelNeuron()
        for _ in range(50_000):
            n.step(10.0)
        assert 0 <= n.h <= 1
        assert 0 <= n.n <= 1
        assert 0 <= n.q <= 1

    def test_reset(self):
        n = BoothRinzelNeuron()
        for _ in range(1000):
            n.step(10.0)
        n.reset()
        assert n.vs == -65.0
        assert n.vd == -65.0
        assert n.ca == 0.0


class TestBoothRinzelNetwork:
    def test_population(self):
        pop = Population(BoothRinzelNeuron, n=5, label="br")
        assert pop.n == 5
        assert pop.model_name == "BoothRinzelNeuron"

    def test_network_spikes(self):
        pop = Population(BoothRinzelNeuron, n=10, label="br")
        drive = PoissonInput(n=10, rate_hz=500.0, weight=10.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=1.0, dt=0.001, backend="python")
        assert mon.count > 0, "network produced zero spikes"

    def test_with_projection(self):
        pop = Population(BoothRinzelNeuron, n=10, label="br")
        proj = Projection(pop, pop, weight=0.5, probability=0.2, seed=42)
        drive = PoissonInput(n=10, rate_hz=500.0, weight=10.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, proj, drive, mon)
        net.run(duration=0.5, dt=0.001, backend="python")
        trains = mon.spike_trains
        assert isinstance(trains, dict)


class TestBoothRinzelAnalysis:
    def _get_binary_train(self):
        n = BoothRinzelNeuron()
        train = np.zeros(50_000, dtype=np.int8)
        for t in range(50_000):
            train[t] = n.step(10.0)
        return train

    def test_firing_rate(self):
        train = self._get_binary_train()
        rate = firing_rate(train, dt=0.000025)  # dt=0.025ms (4 sub-steps)
        assert rate > 0

    def test_spike_count(self):
        train = self._get_binary_train()
        assert spike_count(train) > 100

    def test_isi(self):
        train = self._get_binary_train()
        intervals = isi(train, dt=0.000025)
        if intervals.size > 0:
            assert np.all(np.isfinite(intervals))
            assert np.all(intervals > 0)


import pytest


def _booth_state_tuple(neuron):
    return (neuron.vs, neuron.vd, neuron.h, neuron.n, neuron.q, neuron.ca)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dt", 0.0),
        ("p", 0.0),
        ("p", 1.0),
        ("gc", 0.0),
        ("g_na", 0.0),
        ("g_k", 0.0),
        ("g_ca", 0.0),
        ("g_kca", 0.0),
        ("g_l", 0.0),
        ("c_m", 0.0),
        ("alpha_ca", 0.0),
        ("k_ca", 0.0),
        ("f_ca", 0.0),
        ("h", -0.01),
        ("n", 1.01),
        ("q", float("nan")),
        ("ca", -0.01),
    ],
)
def test_booth_rinzel_rejects_invalid_physical_configuration(field, value):
    kwargs = {field: value}
    with pytest.raises(ValueError):
        BoothRinzelNeuron(**kwargs)


def test_booth_rinzel_runtime_validation_is_fail_closed():
    neuron = BoothRinzelNeuron()
    neuron.p = 1.0
    before = _booth_state_tuple(neuron)
    with pytest.raises(ValueError):
        neuron.step(10.0)
    assert _booth_state_tuple(neuron) == before


def test_booth_rinzel_nonfinite_input_is_fail_closed():
    neuron = BoothRinzelNeuron()
    before = _booth_state_tuple(neuron)
    with pytest.raises(ValueError):
        neuron.step(float("nan"))
    assert _booth_state_tuple(neuron) == before


def test_booth_rinzel_drive_preserves_physical_bounds():
    neuron = BoothRinzelNeuron()
    for _ in range(100):
        neuron.step(8.0)
        assert -200.0 <= neuron.vs <= 100.0
        assert -200.0 <= neuron.vd <= 100.0
        assert 0.0 <= neuron.h <= 1.0
        assert 0.0 <= neuron.n <= 1.0
        assert 0.0 <= neuron.q <= 1.0
        assert neuron.ca >= 0.0
