# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestProjection from former test_network_basic.py

"""Focused suite: TestProjection from former test_network_basic.py."""

from __future__ import annotations

from tests.network_basic_support import *  # noqa: F403

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
