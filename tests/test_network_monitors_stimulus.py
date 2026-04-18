# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for network monitors (Spike/State/Rate) and stimuli

"""Unit tests for SpikeMonitor, StateMonitor, RateMonitor,
TimedArray, StepCurrent, PoissonInput."""

from __future__ import annotations

import numpy as np

from sc_neurocore import StochasticLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.monitor import SpikeMonitor, StateMonitor, RateMonitor
from sc_neurocore.network.stimulus import TimedArray, StepCurrent, PoissonInput


# ---------- SpikeMonitor ----------


class TestSpikeMonitor:
    def test_empty_after_init(self):
        pop = Population(StochasticLIFNeuron, n=10, label="exc")
        mon = SpikeMonitor(pop)
        assert mon.count == 0

    def test_record_single_spike(self):
        pop = Population(StochasticLIFNeuron, n=5, label="test")
        mon = SpikeMonitor(pop)
        spikes = np.array([0, 1, 0, 0, 0], dtype=np.int8)
        mon.record(spikes, t_step=10)
        assert mon.count == 1

    def test_record_multiple_spikes(self):
        pop = Population(StochasticLIFNeuron, n=4, label="test")
        mon = SpikeMonitor(pop)
        mon.record(np.array([1, 1, 0, 1]), t_step=0)
        mon.record(np.array([0, 0, 1, 0]), t_step=1)
        assert mon.count == 4

    def test_spike_trains_dict(self):
        pop = Population(StochasticLIFNeuron, n=3, label="test")
        mon = SpikeMonitor(pop)
        mon.record(np.array([1, 0, 0]), t_step=5)
        mon.record(np.array([1, 0, 1]), t_step=10)
        trains = mon.spike_trains
        assert isinstance(trains, dict)
        assert 0 in trains
        assert len(trains[0]) == 2  # neuron 0 spiked at step 5 and 10

    def test_label(self):
        pop = Population(StochasticLIFNeuron, n=5, label="my_pop")
        mon = SpikeMonitor(pop, label="custom_label")
        assert mon.label == "custom_label"

    def test_no_spikes_empty_trains(self):
        pop = Population(StochasticLIFNeuron, n=3, label="quiet")
        mon = SpikeMonitor(pop)
        for t in range(50):
            mon.record(np.zeros(3, dtype=np.int8), t_step=t)
        assert mon.count == 0


# ---------- StateMonitor ----------


class TestStateMonitor:
    def test_records_voltage(self):
        pop = Population(StochasticLIFNeuron, n=5, label="test")
        mon = StateMonitor(pop, variables=["v"])
        assert "v" in mon._data

    def test_default_variable_is_v(self):
        pop = Population(StochasticLIFNeuron, n=3, label="test")
        mon = StateMonitor(pop)
        assert mon.variables == ["v"]


# ---------- RateMonitor ----------


class TestRateMonitor:
    def test_empty_after_init(self):
        pop = Population(StochasticLIFNeuron, n=10, label="test")
        mon = RateMonitor(pop, bin_ms=10)
        assert len(mon._spike_counts) == 0

    def test_bin_accumulation(self):
        pop = Population(StochasticLIFNeuron, n=10, label="test")
        mon = RateMonitor(pop, bin_ms=10)
        # 10 ms bin at dt=1ms = 10 steps
        for t in range(20):
            spikes = np.array([1 if t % 5 == 0 else 0] * 10)
            mon.record(spikes, t_step=t, dt=0.001)
        # After 20 steps (2 bins of 10 ms)
        assert len(mon._spike_counts) >= 1


# ---------- TimedArray ----------


class TestTimedArray:
    def test_returns_value_at_step(self):
        ta = TimedArray([0.0, 1.0, 2.0, 3.0], dt=0.001)
        assert ta.get_current(0) == 0.0
        assert ta.get_current(2) == 2.0

    def test_clamps_past_end(self):
        ta = TimedArray([5.0, 10.0], dt=0.001)
        assert ta.get_current(100) == 10.0

    def test_accepts_numpy_array(self):
        arr = np.linspace(0, 1, 50)
        ta = TimedArray(arr, dt=0.001)
        np.testing.assert_allclose(ta.get_current(25), arr[25])

    def test_single_value(self):
        ta = TimedArray([42.0])
        assert ta.get_current(0) == 42.0
        assert ta.get_current(999) == 42.0


# ---------- StepCurrent ----------


class TestStepCurrent:
    def test_zero_outside_window(self):
        sc = StepCurrent(onset=100, offset=200, amplitude=5.0)
        assert sc.get_current(50) == 0.0
        assert sc.get_current(250) == 0.0

    def test_amplitude_inside_window(self):
        sc = StepCurrent(onset=100, offset=200, amplitude=5.0)
        assert sc.get_current(150) == 5.0

    def test_onset_inclusive(self):
        sc = StepCurrent(onset=10, offset=20, amplitude=1.0)
        assert sc.get_current(10) == 1.0

    def test_offset_exclusive(self):
        sc = StepCurrent(onset=10, offset=20, amplitude=1.0)
        assert sc.get_current(20) == 0.0

    def test_negative_amplitude(self):
        sc = StepCurrent(onset=0, offset=100, amplitude=-3.0)
        assert sc.get_current(50) == -3.0


# ---------- PoissonInput ----------


class TestPoissonInput:
    def test_creation(self):
        pi = PoissonInput(n=20, rate_hz=100.0, weight=1.0, dt=0.001, seed=42)
        assert pi.n == 20
        assert pi.rate_hz == 100.0

    def test_rate_stored(self):
        pi = PoissonInput(n=5, rate_hz=50.0, weight=2.0, dt=0.001, seed=42)
        assert pi.rate_hz == 50.0
        assert pi.weight == 2.0
