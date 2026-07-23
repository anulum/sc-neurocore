# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_debug_toolkit.py

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

__all__ = ['np', 'pytest', 'ExecutionTrace', 'find_divergence', 'spike_diff', 'causal_chain', 'DivergencePoint', 'CausalEvent', '_make_trace', '_MockPop', '_MockNetwork']
