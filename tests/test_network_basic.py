# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for declarative network simulation engine

"""Tests for the declarative network simulation engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.monitor import SpikeMonitor, StateMonitor, RateMonitor
from sc_neurocore.network.network import Network
from sc_neurocore.network.stimulus import TimedArray, PoissonInput, StepCurrent
from sc_neurocore.network import topology
from sc_neurocore.network.export import export_verilog
from sc_neurocore.exceptions import SCHardwareError

# --- Population ---


class TestPopulation:
    def test_create_by_string(self):
        pop = Population("LapicqueNeuron", 5)
        assert pop.n == 5
        assert pop.label == "LapicqueNeuron"

    def test_create_by_class(self):
        from sc_neurocore.neurons.models import AdExNeuron

        pop = Population(AdExNeuron, 3, label="exc")
        assert pop.n == 3
        assert pop.label == "exc"

    def test_unknown_model_string(self):
        with pytest.raises(ValueError, match="Unknown model"):
            Population("NonexistentNeuron", 2)

    def test_step_all_returns_spike_vector(self):
        pop = Population("LapicqueNeuron", 4)
        currents = np.array([10.0, 0.0, 10.0, 0.0])
        spikes = pop.step_all(currents)
        assert spikes.shape == (4,)
        assert spikes.dtype == np.int8

    def test_reset_all(self):
        pop = Population("LapicqueNeuron", 3)
        pop.step_all(np.array([100.0, 100.0, 100.0]))
        pop.reset_all()
        assert np.allclose(pop.voltages, 0.0)

    def test_get_states(self):
        pop = Population("LapicqueNeuron", 3)
        states = pop.get_states()
        assert "v" in states
        assert states["v"].shape == (3,)

    def test_get_states_uses_neuron_get_state_when_available(self):
        # Neuron models exposing get_state() drive the state keys directly,
        # rather than the dataclass-field fallback used for plain dataclasses.
        from sc_neurocore.neurons.models import Izhikevich2007Neuron

        pop = Population(Izhikevich2007Neuron, 3)
        states = pop.get_states()
        assert "v" in states and "u" in states
        assert states["v"].shape == (3,)

    def test_params_override(self):
        pop = Population("LapicqueNeuron", 2, params={"v_threshold": 0.5})
        assert pop.neurons[0].v_threshold == 0.5

    def test_spike_gating_skips_resting_silent_neuron_and_steps_active_neuron(self):
        class GatedNeuron:
            def __init__(self):
                self.v = 0.0
                self.v_rest = 0.0
                self.v_threshold = 1.0
                self.step_calls = 0

            def step(self, current):
                self.step_calls += 1
                self.v += current
                return self.v >= self.v_threshold

        pop = Population(GatedNeuron, n=2, label="gated")

        spikes = pop.step_all(np.array([0.0, 1.25]), spike_gating=True)

        assert pop.neurons[0].step_calls == 0
        assert pop.neurons[0].v == 0.0
        assert pop.voltages[0] == 0.0
        assert pop.neurons[1].step_calls == 1
        assert pop.neurons[1].v == 1.25
        assert pop.voltages[1] == 1.25
        np.testing.assert_array_equal(spikes, np.array([0, 1], dtype=np.int8))

    def test_get_states_falls_back_to_voltage_for_minimal_neuron(self):
        class MinimalVoltageNeuron:
            def __init__(self):
                self.v = -0.25

        pop = Population(MinimalVoltageNeuron, n=3, label="minimal")
        for neuron, voltage in zip(pop.neurons, [-0.5, 0.0, 0.75], strict=True):
            neuron.v = voltage

        states = pop.get_states()

        assert set(states) == {"v"}
        np.testing.assert_allclose(states["v"], np.array([-0.5, 0.0, 0.75]))

    def test_reset_all_prefers_reset_and_updates_voltage_cache(self):
        class ResetNeuron:
            def __init__(self):
                self.v = 1.0
                self.reset_calls = 0

            def reset(self):
                self.reset_calls += 1
                self.v = -0.125

        pop = Population(ResetNeuron, n=2, label="reset")
        pop.neurons[0].v = 0.5
        pop.neurons[1].v = 0.75

        pop.reset_all()

        assert [neuron.reset_calls for neuron in pop.neurons] == [1, 1]
        np.testing.assert_allclose(pop.voltages, np.array([-0.125, -0.125]))

    def test_reset_all_uses_reset_state_when_reset_is_unavailable(self):
        class ResetStateNeuron:
            def __init__(self):
                self.v = 1.0
                self.reset_state_calls = 0

            def reset_state(self):
                self.reset_state_calls += 1
                self.v = -0.375

        pop = Population(ResetStateNeuron, n=2, label="reset_state")
        pop.neurons[0].v = 0.5
        pop.neurons[1].v = 0.75

        pop.reset_all()

        assert [neuron.reset_state_calls for neuron in pop.neurons] == [1, 1]
        np.testing.assert_allclose(pop.voltages, np.array([-0.375, -0.375]))

    def test_get_states_uses_dataclass_fields_without_timestep_parameter(self):
        @dataclass
        class DataclassStateNeuron:
            v: float = -0.5
            adaptation: float = 0.25
            dt: float = 0.001

        pop = Population(
            DataclassStateNeuron,
            n=2,
            params={"v": -0.4, "adaptation": 0.125, "dt": 0.002},
            label="dataclass",
        )
        pop.neurons[1].v = 0.6
        pop.neurons[1].adaptation = 0.5

        states = pop.get_states()

        assert set(states) == {"v", "adaptation"}
        np.testing.assert_allclose(states["v"], np.array([-0.4, 0.6]))
        np.testing.assert_allclose(states["adaptation"], np.array([0.125, 0.5]))

    def test_empty_population_exposes_empty_state_mapping(self):
        pop = Population("LapicqueNeuron", n=0, label="empty")

        assert pop.n == 0
        assert pop.get_states() == {}


# --- Topology ---


class TestTopology:
    def test_random_connectivity(self):
        indptr, indices, data = topology.random_connectivity(5, 5, 0.5, 1.0, seed=0)
        assert indptr.shape == (6,)
        assert len(indices) == len(data)
        assert np.all(data == 1.0)

    def test_all_to_all(self):
        indptr, indices, data = topology.all_to_all(3, 4, 2.0)
        assert indptr[-1] == 12
        assert np.all(data == 2.0)

    def test_ring_topology(self):
        indptr, indices, data = topology.ring_topology(6, 1, 0.5)
        assert indptr[-1] == 12  # 6 nodes * 2 connections each

    def test_small_world(self):
        indptr, indices, data = topology.small_world(10, 4, 0.1, 1.0, seed=7)
        assert indptr.shape == (11,)
        assert len(indices) > 0

    def test_scale_free(self):
        indptr, indices, data = topology.scale_free(10, 2, 1.0, seed=7)
        assert indptr.shape == (11,)
        assert len(indices) > 0

    def test_grid_topology(self):
        indptr, indices, data = topology.grid_topology(3, 3, 1, 1.0)
        assert indptr.shape == (10,)  # 9 nodes + 1


# --- Projection ---


class TestProjection:
    def test_propagate_basic(self):
        src = Population("LapicqueNeuron", 3)
        tgt = Population("LapicqueNeuron", 3)
        proj = Projection(src, tgt, weight=1.0, probability=1.0, topology="all_to_all")
        spikes = np.array([1, 0, 1], dtype=np.int8)
        current = proj.propagate(spikes)
        assert current.shape == (3,)
        assert current.sum() > 0

    def test_delay_buffer(self):
        src = Population("LapicqueNeuron", 2)
        tgt = Population("LapicqueNeuron", 2)
        proj = Projection(src, tgt, weight=1.0, delay=2.0, topology="all_to_all")
        spikes = np.array([1, 0], dtype=np.int8)
        c1 = proj.propagate(spikes)
        assert np.allclose(c1, 0.0)  # delay=2: nothing yet
        c2 = proj.propagate(np.zeros(2, dtype=np.int8))
        assert np.allclose(c2, 0.0)  # still buffered
        c3 = proj.propagate(np.zeros(2, dtype=np.int8))
        assert c3.sum() > 0  # delayed current arrives after 2 steps

    def test_per_synapse_delay(self):
        src = Population("LapicqueNeuron", 3)
        tgt = Population("LapicqueNeuron", 3)
        proj = Projection(src, tgt, weight=1.0, topology="all_to_all")
        n_syn = proj.n_synapses
        delays = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=np.float64)[:n_syn]
        proj_d = Projection(src, tgt, weight=1.0, delay=delays, topology="all_to_all")
        assert proj_d.delay_mode == "per_synapse"
        assert proj_d.max_delay == 3

        spikes = np.array([1, 0, 0], dtype=np.float64)
        # Step 1: inject spikes
        c1 = proj_d.propagate(spikes)
        # Step 2-4: delayed arrivals
        arrivals = [c1.sum()]
        for _ in range(4):
            c = proj_d.propagate(np.zeros(3))
            arrivals.append(c.sum())
        # Some current should arrive at steps 2, 3, 4 (delays 1, 2, 3)
        assert sum(arrivals) > 0, "Per-synapse delay produced no output"
        assert arrivals[0] == 0.0 or arrivals[1] > 0 or arrivals[2] > 0

    def test_per_synapse_delay_validates_length(self):
        src = Population("LapicqueNeuron", 2)
        tgt = Population("LapicqueNeuron", 2)
        with pytest.raises(ValueError, match="must match"):
            Projection(src, tgt, weight=1.0, delay=np.array([1, 2, 3]), topology="all_to_all")

    def test_delay_mode_property(self):
        src = Population("LapicqueNeuron", 2)
        tgt = Population("LapicqueNeuron", 2)
        p0 = Projection(src, tgt, weight=1.0, delay=0.0, topology="all_to_all")
        assert p0.delay_mode == "none"
        p1 = Projection(src, tgt, weight=1.0, delay=3.0, topology="all_to_all")
        assert p1.delay_mode == "uniform"

    def test_stdp_modifies_weights(self):
        src = Population("LapicqueNeuron", 2)
        tgt = Population("LapicqueNeuron", 2)
        proj = Projection(src, tgt, weight=0.5, probability=1.0, plasticity="stdp")
        w_before = proj.data.copy()
        src_sp = np.array([1, 0], dtype=np.int8)
        tgt_sp = np.array([0, 1], dtype=np.int8)
        for _ in range(20):
            proj.update_plasticity(src_sp, tgt_sp)
        assert not np.array_equal(proj.data, w_before)


# --- Monitors ---


class TestMonitors:
    def test_spike_monitor_count(self):
        pop = Population("LapicqueNeuron", 3)
        mon = SpikeMonitor(pop)
        mon.record(np.array([1, 0, 1], dtype=np.int8), 0)
        mon.record(np.array([0, 1, 0], dtype=np.int8), 1)
        assert mon.count == 3

    def test_spike_monitor_trains(self):
        pop = Population("LapicqueNeuron", 2)
        mon = SpikeMonitor(pop)
        mon.record(np.array([1, 0], dtype=np.int8), 0)
        mon.record(np.array([1, 1], dtype=np.int8), 5)
        trains = mon.spike_trains
        assert len(trains[0]) == 2
        assert len(trains[1]) == 1

    def test_spike_monitor_raster(self):
        pop = Population("LapicqueNeuron", 2)
        mon = SpikeMonitor(pop)
        mon.record(np.array([1, 0], dtype=np.int8), 0)
        ts, ids = mon.raster_data()
        assert len(ts) == 1
        assert ids[0] == 0

    def test_spike_monitor_firing_rates(self):
        pop = Population("LapicqueNeuron", 2)
        mon = SpikeMonitor(pop)
        mon.record(np.array([1, 0], dtype=np.int8), 0)
        mon.record(np.array([1, 0], dtype=np.int8), 1)
        rates = mon.firing_rates(n_steps=100, dt=0.001)
        assert rates[0] > 0
        assert rates[1] == 0.0

    def test_spike_monitor_isi(self):
        pop = Population("LapicqueNeuron", 1)
        mon = SpikeMonitor(pop)
        mon.record(np.array([1], dtype=np.int8), 10)
        mon.record(np.array([1], dtype=np.int8), 20)
        mon.record(np.array([1], dtype=np.int8), 35)
        intervals = mon.isi(0)
        np.testing.assert_array_equal(intervals, [10, 15])

    def test_state_monitor_traces(self):
        pop = Population("LapicqueNeuron", 2)
        mon = StateMonitor(pop, variables=["v"])
        mon.snapshot(0)
        mon.snapshot(1)
        assert mon.traces["v"].shape == (2, 2)
        assert len(mon.t) == 2

    def test_rate_monitor(self):
        pop = Population("LapicqueNeuron", 4)
        mon = RateMonitor(pop, bin_ms=10)
        for t in range(100):
            spikes = np.array([1, 0, 0, 0], dtype=np.int8)
            mon.record(spikes, t, dt=0.001)
        assert len(mon.rate) > 0
        assert mon.rate[0] > 0


# --- Stimulus ---


class TestStimulus:
    def test_timed_array(self):
        ta = TimedArray([0.0, 1.0, 2.0, 3.0], dt=0.001)
        assert ta.get_current(0) == 0.0
        assert ta.get_current(2) == 2.0
        assert ta.get_current(100) == 3.0  # clamp

    def test_poisson_input(self):
        pi = PoissonInput(n=10, rate_hz=1000.0, weight=0.5, dt=0.001, seed=0)
        c = pi.get_current(0)
        assert c.shape == (10,)

    def test_step_current(self):
        sc = StepCurrent(onset=10, offset=20, amplitude=5.0)
        assert sc.get_current(5) == 0.0
        assert sc.get_current(15) == 5.0
        assert sc.get_current(20) == 0.0


# --- Network integration ---


class TestNetwork:
    def test_run_basic(self):
        pop = Population("LapicqueNeuron", 5)
        stim = StepCurrent(onset=0, offset=50, amplitude=2.0)
        stim.target = pop
        mon = SpikeMonitor(pop)
        net = Network(pop, stim, mon, seed=0)
        net.run(duration=0.05, dt=0.001)
        assert mon.count >= 0  # simulation ran without error

    def test_two_populations(self):
        exc = Population("LapicqueNeuron", 10, label="exc")
        inh = Population("LapicqueNeuron", 5, label="inh")
        proj = Projection(exc, inh, weight=0.5, probability=0.5)
        stim = StepCurrent(onset=0, offset=100, amplitude=3.0)
        stim.target = exc
        mon_exc = SpikeMonitor(exc)
        mon_inh = SpikeMonitor(inh)
        net = Network(exc, inh, proj, stim, mon_exc, mon_inh, seed=42)
        net.run(duration=0.1, dt=0.001)
        assert mon_exc.count + mon_inh.count >= 0

    def test_network_with_state_monitor(self):
        pop = Population("LapicqueNeuron", 3)
        stim = StepCurrent(onset=0, offset=50, amplitude=2.0)
        stim.target = pop
        smon = StateMonitor(pop, variables=["v"])
        net = Network(pop, stim, smon, seed=0)
        net.run(duration=0.01, dt=0.001)
        assert smon.traces["v"].shape[0] == 10
        assert smon.traces["v"].shape[1] == 3

    def test_network_with_rate_monitor(self):
        pop = Population("LapicqueNeuron", 5)
        stim = StepCurrent(onset=0, offset=500, amplitude=3.0)
        stim.target = pop
        rmon = RateMonitor(pop, bin_ms=10)
        net = Network(pop, stim, rmon, seed=0)
        net.run(duration=0.05, dt=0.001)
        assert len(rmon.rate) > 0

    def test_network_add_rejects_bad_type(self):
        net = Network()
        with pytest.raises(TypeError, match="Unknown object type"):
            net.add("not_a_valid_object")

    def test_poisson_stimulus_in_network(self):
        pop = Population("LapicqueNeuron", 10)
        pi = PoissonInput(n=10, rate_hz=500.0, weight=2.0, seed=7)
        pi.target = pop
        mon = SpikeMonitor(pop)
        net = Network(pop, pi, mon, seed=7)
        net.run(duration=0.1, dt=0.001)
        assert mon.count >= 0


# --- Export ---


class TestExport:
    def test_export_lif_network(self, tmp_path):
        pop = Population("LapicqueNeuron", 4)
        net = Network(pop)
        path = export_verilog(net, str(tmp_path / "verilog"))
        assert path.endswith(".v")
        with open(path) as f:
            content = f.read()
        assert "sc_lif_array" in content
        assert "sc_network_top" in content

    def test_export_rejects_unsupported(self, tmp_path):
        pop = Population("HodgkinHuxleyNeuron", 2)
        net = Network(pop)
        with pytest.raises(SCHardwareError, match="cannot be exported"):
            export_verilog(net, str(tmp_path / "verilog"))

    def test_export_creates_params_file(self, tmp_path):
        pop = Population("LapicqueNeuron", 8, label="layer0")
        net = Network(pop)
        export_verilog(net, str(tmp_path / "out"))
        params = (tmp_path / "out" / "params.vh").read_text()
        assert "POP_0_SIZE 8" in params
