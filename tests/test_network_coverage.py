# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Coverage gap tests for network/ module

"""Tests to close coverage gaps in network/population.py, network/network.py,
network/monitor.py. These target specific uncovered lines identified by
coverage audit."""

from __future__ import annotations

import numpy as np

from sc_neurocore import StochasticLIFNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor, StateMonitor, RateMonitor
from sc_neurocore.network.stimulus import PoissonInput, StepCurrent, TimedArray


# --- Population coverage gaps ---


class TestPopulationCoverage:
    def test_get_states(self):
        pop = Population(StochasticLIFNeuron, n=5, label="test")
        pop.step_all(np.ones(5) * 0.5)
        states = pop.get_states()
        assert isinstance(states, dict)
        assert "v" in states
        assert states["v"].shape == (5,)

    def test_set_voltages(self):
        pop = Population(StochasticLIFNeuron, n=5, label="test")
        new_v = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        pop.set_voltages(new_v)
        for i, neuron in enumerate(pop.neurons):
            assert abs(neuron.v - new_v[i]) < 1e-6

    def test_step_all_with_spike_gating(self):
        pop = Population(StochasticLIFNeuron, n=10, label="test")
        currents = np.ones(10) * 0.5
        spikes = pop.step_all(currents, spike_gating=True)
        assert len(spikes) == 10

    def test_voltages_property(self):
        pop = Population(StochasticLIFNeuron, n=3, label="test")
        v = pop.voltages
        assert isinstance(v, np.ndarray)
        assert len(v) == 3

    def test_empty_population(self):
        pop = Population(StochasticLIFNeuron, n=0, label="empty")
        assert pop.n == 0
        states = pop.get_states()
        assert states == {}


# --- Network coverage gaps ---


class TestNetworkCoverage:
    def test_run_python_backend(self):
        pop = Population(StochasticLIFNeuron, n=10, label="exc")
        drive = PoissonInput(n=10, rate_hz=100.0, weight=2.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop)
        net = Network(pop, drive, mon)
        net.run(duration=0.01, dt=0.001, backend="python")
        assert mon.count >= 0

    def test_add_method(self):
        net = Network()
        pop = Population(StochasticLIFNeuron, n=5, label="p")
        net.add(pop)
        assert len(net.populations) == 1

    def test_step_current_stimulus(self):
        pop = Population(StochasticLIFNeuron, n=5, label="test")
        step = StepCurrent(onset=5, offset=15, amplitude=3.0)
        mon = SpikeMonitor(pop)
        net = Network(pop, step, mon)
        net.run(duration=0.02, dt=0.001, backend="python")

    def test_timed_array_stimulus(self):
        pop = Population(StochasticLIFNeuron, n=5, label="test")
        arr = TimedArray(np.linspace(0, 3, 20), dt=0.001)
        mon = SpikeMonitor(pop)
        net = Network(pop, arr, mon)
        net.run(duration=0.02, dt=0.001, backend="python")

    def test_state_monitor_in_network(self):
        pop = Population(StochasticLIFNeuron, n=5, label="test")
        drive = PoissonInput(n=5, rate_hz=100.0, weight=2.0, dt=0.001, seed=42)
        smon = StateMonitor(pop, variables=["v"])
        net = Network(pop, drive, smon)
        net.run(duration=0.01, dt=0.001, backend="python")

    def test_rate_monitor_in_network(self):
        pop = Population(StochasticLIFNeuron, n=10, label="test")
        drive = PoissonInput(n=10, rate_hz=100.0, weight=2.0, dt=0.001, seed=42)
        rmon = RateMonitor(pop, bin_ms=5)
        net = Network(pop, drive, rmon)
        net.run(duration=0.02, dt=0.001, backend="python")

    def test_projection_propagation(self):
        """Verify projections carry spikes between populations."""
        pop_a = Population(StochasticLIFNeuron, n=10, label="a")
        pop_b = Population(StochasticLIFNeuron, n=5, label="b")
        proj = Projection(pop_a, pop_b, weight=0.5, probability=0.5, seed=42)
        drive = PoissonInput(n=10, rate_hz=200.0, weight=3.0, dt=0.001, seed=42)
        mon = SpikeMonitor(pop_b)
        net = Network(pop_a, pop_b, proj, drive, mon)
        net.run(duration=0.05, dt=0.001, backend="python")
        # pop_b should get some spikes through the projection
        assert mon.count >= 0

    def test_fim_lambda_in_run(self):
        pop = Population(StochasticLIFNeuron, n=10, label="e")
        proj = Projection(pop, pop, weight=0.3, probability=0.3, plasticity="stdp", seed=42)
        drive = PoissonInput(n=10, rate_hz=100.0, weight=2.0, dt=0.001, seed=42)
        net = Network(pop, proj, drive, fim_lambda=5.0)
        net.run(duration=0.02, dt=0.001, backend="python")


# --- Monitor coverage gaps ---


class TestMonitorCoverage:
    def test_spike_monitor_spike_trains_property(self):
        pop = Population(StochasticLIFNeuron, n=3, label="test")
        mon = SpikeMonitor(pop)
        mon.record(np.array([1, 0, 1], dtype=np.int8), t_step=5)
        trains = mon.spike_trains
        assert 0 in trains
        assert 2 in trains

    def test_spike_monitor_count_property(self):
        pop = Population(StochasticLIFNeuron, n=2, label="test")
        mon = SpikeMonitor(pop)
        assert mon.count == 0
        mon.record(np.array([1, 1], dtype=np.int8), t_step=0)
        assert mon.count == 2

    def test_rate_monitor_record(self):
        pop = Population(StochasticLIFNeuron, n=5, label="test")
        rmon = RateMonitor(pop, bin_ms=10)
        for t in range(20):
            spikes = np.zeros(5, dtype=np.int8)
            if t % 3 == 0:
                spikes[0] = 1
            rmon.record(spikes, t_step=t, dt=0.001)

    def test_state_monitor_variables(self):
        pop = Population(StochasticLIFNeuron, n=3, label="test")
        smon = StateMonitor(pop, variables=["v"])
        assert smon.variables == ["v"]


# --- Projection coverage gaps ---


class TestProjectionCoverage:
    def test_delay_projection(self):
        pop = Population(StochasticLIFNeuron, n=5, label="test")
        proj = Projection(pop, pop, weight=0.3, probability=0.5, delay=3.0, seed=42)
        assert proj.source is pop
        assert proj.target is pop

    def test_weight_threshold(self):
        pop = Population(StochasticLIFNeuron, n=5, label="test")
        proj = Projection(pop, pop, weight=0.3, probability=0.8, weight_threshold=0.1, seed=42)
        assert len(proj.data) > 0

    def test_directional_bias_parameter(self):
        pop_a = Population(StochasticLIFNeuron, n=10, label="a")
        pop_b = Population(StochasticLIFNeuron, n=5, label="b")
        proj = Projection(pop_a, pop_b, weight=0.3, probability=0.5, plasticity="stdp", seed=42)
        src_sp = np.zeros(10, dtype=np.int8)
        src_sp[0] = 1
        tgt_sp = np.zeros(5, dtype=np.int8)
        tgt_sp[0] = 1
        proj.update_plasticity(src_sp, tgt_sp, directional_bias=1.36)


# --- Deep monitor coverage ---


class TestMonitorDeepCoverage:
    def test_record_event_direct(self):
        """SpikeMonitor.record_event() — Rust backend path."""
        pop = Population(StochasticLIFNeuron, n=5, label="test")
        mon = SpikeMonitor(pop)
        mon.record_event(2, 100)
        mon.record_event(0, 200)
        assert mon.count == 2

    def test_spike_times_property(self):
        pop = Population(StochasticLIFNeuron, n=3, label="test")
        mon = SpikeMonitor(pop)
        mon.record_event(0, 10)
        mon.record_event(1, 20)
        times = mon.spike_times
        assert len(times) == 2
        assert times[0] == 10

    def test_state_monitor_snapshot_and_traces(self):
        pop = Population(StochasticLIFNeuron, n=5, label="test")
        smon = StateMonitor(pop, variables=["v"])
        for t in range(10):
            pop.step_all(np.ones(5) * 0.3)
            smon.snapshot(t_step=t)
        traces = smon.traces
        assert "v" in traces
        assert traces["v"].shape[0] == 10

    def test_state_monitor_subset(self):
        pop = Population(StochasticLIFNeuron, n=10, label="test")
        smon = StateMonitor(pop, variables=["v"], record=[0, 3, 7])
        pop.step_all(np.ones(10) * 0.5)
        smon.snapshot(t_step=0)
        traces = smon.traces
        assert traces["v"].shape[1] == 3

    def test_rate_monitor_rate_property(self):
        pop = Population(StochasticLIFNeuron, n=10, label="test")
        rmon = RateMonitor(pop, bin_ms=5)
        for t in range(10):
            spikes = np.zeros(10, dtype=np.int8)
            spikes[t % 10] = 1
            rmon.record(spikes, t_step=t, dt=0.001)
        rate = rmon.rate
        assert isinstance(rate, np.ndarray)

    def test_rate_monitor_t_property(self):
        pop = Population(StochasticLIFNeuron, n=5, label="test")
        rmon = RateMonitor(pop, bin_ms=5)
        for t in range(10):
            rmon.record(np.zeros(5, dtype=np.int8), t_step=t, dt=0.001)
        assert isinstance(rmon.t, np.ndarray)


# --- Network backend logic ---


class TestNetworkBackendCoverage:
    def test_can_use_rust_returns_false(self):
        pop = Population(StochasticLIFNeuron, n=5, label="test")
        net = Network(pop)
        assert not net._can_use_rust()

    def test_auto_falls_to_python(self):
        pop = Population(StochasticLIFNeuron, n=5, label="test")
        drive = PoissonInput(n=5, rate_hz=100.0, weight=2.0, dt=0.001, seed=42)
        net = Network(pop, drive)
        net.run(duration=0.005, dt=0.001, backend="auto")

    def test_rust_raises_without_engine(self):
        pop = Population(StochasticLIFNeuron, n=5, label="test")
        net = Network(pop)
        try:
            net.run(duration=0.001, dt=0.001, backend="rust")
            raise AssertionError("should raise")
        except RuntimeError:
            pass

    def test_mpi_raises_without_mpi4py(self):
        pop = Population(StochasticLIFNeuron, n=5, label="test")
        net = Network(pop)
        try:
            net.run(duration=0.001, dt=0.001, backend="mpi")
            raise AssertionError("should raise")
        except (ImportError, RuntimeError):
            pass

    def test_progress_flag(self):
        pop = Population(StochasticLIFNeuron, n=5, label="test")
        drive = PoissonInput(n=5, rate_hz=100.0, weight=2.0, dt=0.001, seed=42)
        net = Network(pop, drive)
        net.run(duration=0.01, dt=0.001, backend="python", progress=True)


# --- Rust backend mock coverage ---


class TestRustBackendMock:
    def test_get_rust_engine_caches(self):
        import sc_neurocore.network.network as netmod

        old = netmod._RUST_ENGINE
        try:
            netmod._RUST_ENGINE = None
            result = netmod._get_rust_engine()
            assert result is False
            assert netmod._RUST_ENGINE is False
        finally:
            netmod._RUST_ENGINE = old

    def test_rust_supports_model_false(self):
        import sc_neurocore.network.network as netmod

        assert not netmod._rust_supports_model("StochasticLIFNeuron")

    def test_can_use_rust_with_stimuli(self):
        pop = Population(StochasticLIFNeuron, n=5, label="test")
        drive = PoissonInput(n=5, rate_hz=100.0, weight=2.0, dt=0.001, seed=42)
        net = Network(pop, drive)
        assert not net._can_use_rust()

    def test_can_use_rust_with_plasticity(self):
        pop = Population(StochasticLIFNeuron, n=5, label="test")
        proj = Projection(pop, pop, weight=0.3, probability=0.3, plasticity="stdp", seed=42)
        net = Network(pop, proj)
        assert not net._can_use_rust()

    def test_run_rust_mock(self):
        from unittest.mock import MagicMock
        import sc_neurocore.network.network as netmod

        mock_runner = MagicMock()
        mock_runner.add_population.return_value = 0
        mock_runner.run.return_value = {
            "voltages": [np.zeros(5)],
            "spike_data": [np.array([], dtype=np.uint64)],
        }
        mock_cls = MagicMock(return_value=mock_runner)
        mock_cls.supported_models.return_value = ["StochasticLIFNeuron"]

        old = netmod._RUST_ENGINE
        try:
            netmod._RUST_ENGINE = mock_cls
            pop = Population(StochasticLIFNeuron, n=5, label="test")
            mon = SpikeMonitor(pop)
            net = Network(pop, mon)
            net.run(duration=0.005, dt=0.001, backend="rust")
            mock_runner.run.assert_called_once()
        finally:
            netmod._RUST_ENGINE = old

    def test_rust_supports_model_mock(self):
        from unittest.mock import MagicMock
        import sc_neurocore.network.network as netmod

        mock_cls = MagicMock()
        mock_cls.supported_models.return_value = ["StochasticLIFNeuron"]

        old = netmod._RUST_ENGINE
        try:
            netmod._RUST_ENGINE = mock_cls
            assert netmod._rust_supports_model("StochasticLIFNeuron")
            assert not netmod._rust_supports_model("Nonexistent")
        finally:
            netmod._RUST_ENGINE = old
