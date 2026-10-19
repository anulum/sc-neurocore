# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNetwork from former test_network_basic.py

"""Focused suite: TestNetwork from former test_network_basic.py."""

from __future__ import annotations

from tests.network_basic_support import *  # noqa: F403


class TestNetwork:
    """End-to-end runs of a Network with populations, stimuli and monitors."""

    def test_run_basic(self) -> None:
        """A single stimulated population with a spike monitor runs without error."""
        pop = Population("LapicqueNeuron", 5)
        stim = StepCurrent(onset=0, offset=50, amplitude=2.0)
        stim.target = pop
        mon = SpikeMonitor(pop)
        net = Network(pop, stim, mon, seed=0)
        net.run(duration=0.05, dt=0.001)
        assert mon.count >= 0  # simulation ran without error

    def test_two_populations(self) -> None:
        """Two populations joined by a projection run together without error."""
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

    def test_network_with_state_monitor(self) -> None:
        """A state monitor records one voltage row per step for every neuron."""
        pop = Population("LapicqueNeuron", 3)
        stim = StepCurrent(onset=0, offset=50, amplitude=2.0)
        stim.target = pop
        smon = StateMonitor(pop, variables=["v"])
        net = Network(pop, stim, smon, seed=0)
        net.run(duration=0.01, dt=0.001)
        assert smon.traces["v"].shape[0] == 10
        assert smon.traces["v"].shape[1] == 3

    def test_network_with_rate_monitor(self) -> None:
        """A rate monitor in a running network produces rate bins."""
        pop = Population("LapicqueNeuron", 5)
        stim = StepCurrent(onset=0, offset=500, amplitude=3.0)
        stim.target = pop
        rmon = RateMonitor(pop, bin_ms=10)
        net = Network(pop, stim, rmon, seed=0)
        net.run(duration=0.05, dt=0.001)
        assert len(rmon.rate) > 0

    def test_network_add_rejects_bad_type(self) -> None:
        """Adding an object of an unknown type raises TypeError."""
        net = Network()
        with pytest.raises(TypeError, match="Unknown object type"):
            net.add("not_a_valid_object")

    def test_poisson_stimulus_in_network(self) -> None:
        """A Poisson input drives a population inside a network without error."""
        pop = Population("LapicqueNeuron", 10)
        pi = PoissonInput(n=10, rate_hz=500.0, weight=2.0, seed=7)
        pi.target = pop
        mon = SpikeMonitor(pop)
        net = Network(pop, pi, mon, seed=7)
        net.run(duration=0.1, dt=0.001)
        assert mon.count >= 0
