# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCausalChain from former test_debug_toolkit.py

"""Focused suite: TestCausalChain from former test_debug_toolkit.py."""

from __future__ import annotations

from tests.debug_toolkit_support import *  # noqa: F403

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
