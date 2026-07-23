# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSpikeTracer from former test_debug_toolkit.py

"""Focused suite: TestSpikeTracer from former test_debug_toolkit.py."""

from __future__ import annotations

from tests.debug_toolkit_support import *  # noqa: F403

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
