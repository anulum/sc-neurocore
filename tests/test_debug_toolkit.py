# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

# Tests for sc_neurocore.debug (tracer + analyzer)

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.debug.tracer import ExecutionTrace
from sc_neurocore.debug.analyzer import (
    find_divergence,
    spike_diff,
    causal_chain,
    DivergencePoint,
    CausalEvent,
)


def _make_trace(n_neurons=5, n_steps=10, spikes=None, voltages=None, currents=None):
    if spikes is None:
        spikes = np.zeros((n_steps, n_neurons), dtype=np.int8)
    if voltages is None:
        voltages = np.random.randn(n_steps, n_neurons) * 0.1
    if currents is None:
        currents = np.random.randn(n_steps, n_neurons) * 0.05
    return ExecutionTrace(
        n_neurons=n_neurons,
        n_steps=n_steps,
        spikes=spikes,
        voltages=voltages,
        currents=currents,
        population_labels=["pop_a", "pop_b"],
        population_ranges=[(0, 3), (3, n_neurons)],
    )


class TestExecutionTrace:
    def test_spike_count_zero(self):
        trace = _make_trace()
        assert trace.spike_count == 0

    def test_spike_count_nonzero(self):
        spikes = np.zeros((10, 5), dtype=np.int8)
        spikes[3, 1] = 1
        spikes[7, 4] = 1
        trace = _make_trace(spikes=spikes)
        assert trace.spike_count == 2

    def test_firing_rates(self):
        spikes = np.zeros((10, 5), dtype=np.int8)
        spikes[0, 0] = 1
        spikes[5, 0] = 1
        trace = _make_trace(spikes=spikes)
        rates = trace.firing_rates
        assert rates[0] == pytest.approx(0.2)
        assert rates[1] == pytest.approx(0.0)

    def test_neuron_trace(self):
        spikes = np.zeros((10, 5), dtype=np.int8)
        spikes[2, 1] = 1
        spikes[7, 1] = 1
        voltages = np.random.randn(10, 5)
        currents = np.random.randn(10, 5)
        trace = _make_trace(spikes=spikes, voltages=voltages, currents=currents)
        nt = trace.neuron_trace(1)
        assert len(nt["spike_times"]) == 2
        assert nt["spike_times"][0] == 2
        assert nt["spike_times"][1] == 7

    def test_spike_times(self):
        spikes = np.zeros((10, 5), dtype=np.int8)
        spikes[4, 3] = 1
        trace = _make_trace(spikes=spikes)
        times = trace.spike_times(3)
        assert len(times) == 1
        assert times[0] == 4

    def test_population_spikes(self):
        spikes = np.zeros((10, 5), dtype=np.int8)
        spikes[0, 0] = 1
        spikes[0, 4] = 1
        trace = _make_trace(spikes=spikes)
        pop_a = trace.population_spikes("pop_a")
        assert pop_a.shape == (10, 3)
        assert pop_a[0, 0] == 1
        pop_b = trace.population_spikes("pop_b")
        assert pop_b.shape == (10, 2)
        assert pop_b[0, 1] == 1

    def test_population_spikes_not_found(self):
        trace = _make_trace()
        with pytest.raises(ValueError, match="not found"):
            trace.population_spikes("nonexistent")


class TestFindDivergence:
    def test_identical_traces(self):
        spikes = np.zeros((10, 5), dtype=np.int8)
        spikes[3, 2] = 1
        t1 = _make_trace(spikes=spikes.copy())
        t2 = _make_trace(spikes=spikes.copy())
        assert find_divergence(t1, t2) is None

    def test_divergent_traces(self):
        s1 = np.zeros((10, 5), dtype=np.int8)
        s2 = np.zeros((10, 5), dtype=np.int8)
        s1[3, 2] = 1
        s2[3, 2] = 0
        v = np.random.randn(10, 5)
        t1 = _make_trace(spikes=s1, voltages=v.copy())
        t2 = _make_trace(spikes=s2, voltages=v.copy())
        dp = find_divergence(t1, t2)
        assert isinstance(dp, DivergencePoint)
        assert dp.timestep == 3
        assert dp.neuron_id == 2
        assert dp.trace_a_spike == 1
        assert dp.trace_b_spike == 0

    def test_different_sizes(self):
        s1 = np.zeros((10, 5), dtype=np.int8)
        s2 = np.zeros((8, 3), dtype=np.int8)
        s1[0, 0] = 1
        t1 = _make_trace(n_neurons=5, n_steps=10, spikes=s1)
        t2 = ExecutionTrace(
            n_neurons=3,
            n_steps=8,
            spikes=s2,
            voltages=np.zeros((8, 3)),
            currents=np.zeros((8, 3)),
        )
        dp = find_divergence(t1, t2)
        assert dp is not None
        assert dp.timestep == 0


class TestSpikeDiff:
    def test_identical(self):
        spikes = np.zeros((10, 5), dtype=np.int8)
        t1 = _make_trace(spikes=spikes.copy())
        t2 = _make_trace(spikes=spikes.copy())
        d = spike_diff(t1, t2)
        assert d["total_mismatches"] == 0
        assert d["mismatch_rate"] == 0.0
        assert d["first_divergence"] is None

    def test_with_mismatches(self):
        s1 = np.zeros((10, 5), dtype=np.int8)
        s2 = np.zeros((10, 5), dtype=np.int8)
        s1[0, 0] = 1
        s1[5, 3] = 1
        t1 = _make_trace(spikes=s1)
        t2 = _make_trace(spikes=s2)
        d = spike_diff(t1, t2)
        assert d["total_mismatches"] == 2
        assert d["mismatch_rate"] == pytest.approx(2.0 / 50)
        assert d["first_divergence"] is not None
        assert d["per_neuron_mismatches"][0] == 1
        assert d["per_neuron_mismatches"][3] == 1


class TestCausalChain:
    def test_single_spike(self):
        spikes = np.zeros((10, 5), dtype=np.int8)
        spikes[5, 2] = 1
        currents = np.random.randn(10, 5) * 0.01
        voltages = np.random.randn(10, 5) * 0.1
        trace = _make_trace(spikes=spikes, currents=currents, voltages=voltages)
        chain = causal_chain(trace, neuron_id=2, timestep=5, max_depth=3)
        assert len(chain) >= 1
        assert isinstance(chain[0], CausalEvent)
        assert chain[0].timestep == 5
        assert chain[0].neuron_id == 2
        assert chain[0].spiked is True

    def test_causal_chain_with_predecessors(self):
        spikes = np.zeros((10, 5), dtype=np.int8)
        spikes[5, 2] = 1
        spikes[4, 0] = 1
        spikes[4, 1] = 1
        currents = np.ones((10, 5)) * 0.1
        voltages = np.ones((10, 5)) * 0.5
        trace = _make_trace(spikes=spikes, currents=currents, voltages=voltages)
        chain = causal_chain(trace, neuron_id=2, timestep=5, max_depth=5)
        assert len(chain) >= 3  # target + 2 predecessors

    def test_max_depth_respected(self):
        spikes = np.ones((10, 5), dtype=np.int8)
        trace = _make_trace(spikes=spikes)
        chain = causal_chain(trace, neuron_id=0, timestep=9, max_depth=2)
        timesteps_in_chain = {e.timestep for e in chain}
        assert min(timesteps_in_chain) >= 7

    def test_early_stop_at_time_zero(self):
        spikes = np.zeros((10, 5), dtype=np.int8)
        spikes[0, 0] = 1
        trace = _make_trace(spikes=spikes)
        chain = causal_chain(trace, neuron_id=0, timestep=0, max_depth=5)
        assert len(chain) == 1


class _MockPop:
    def __init__(self, label, n):
        self.label = label
        self.n = n
        self.voltages = np.zeros(n)

    def step_all(self, currents):
        self.voltages = currents * 0.1
        return (currents > 0.5).astype(np.int8)


class _MockNetwork:
    def __init__(self):
        self.populations = [_MockPop("exc", 3), _MockPop("inh", 2)]

    def _apply_stimuli(self, pop_currents, t, dt):
        for pid in pop_currents:
            pop_currents[pid] += 1.0

    def _apply_projections(self, pop_currents, last_spikes):
        pass

    def _record(self, pop, spikes, t, dt):
        pass

    def _update_plasticity(self, last_spikes):
        pass


class TestSpikeTracer:
    def test_run(self):
        from sc_neurocore.debug.tracer import SpikeTracer

        net = _MockNetwork()
        tracer = SpikeTracer(net)
        trace = tracer.run(duration=0.005, dt=0.001)
        assert isinstance(trace, ExecutionTrace)
        assert trace.n_neurons == 5
        assert trace.n_steps == 5
        assert trace.spikes.shape == (5, 5)
        assert trace.voltages.shape == (5, 5)
        assert trace.currents.shape == (5, 5)
        assert trace.population_labels == ["exc", "inh"]
        assert trace.spike_count >= 0
