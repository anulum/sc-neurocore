# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_traub_miles.py

from __future__ import annotations

"""Full pipeline test for TraubMilesNeuron (Traub & Miles 1991).

Reduced hippocampal CA3 pyramidal cell. HH-type Na/K/leak with 10
sub-steps per step() call. High Na conductance (g_Na=100) drives fast
action potentials."""
import numpy as np
import pytest
from sc_neurocore.neurons.models.traub_miles import TraubMilesNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, isi, firing_rate
def _run(neuron: TraubMilesNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]
def _rk4_expected_after_call(
    neuron: TraubMilesNeuron, current: float
) -> tuple[float, float, float, float]:
    v, m, h, n = neuron.v, neuron.m, neuron.h, neuron.n

    def derivatives(
        vs: float, ms: float, hs: float, ns: float
    ) -> tuple[float, float, float, float]:
        am, bm, ah, bh, an, bn = neuron._rates(vs)
        dm = am * (1.0 - ms) - bm * ms
        dh = ah * (1.0 - hs) - bh * hs
        dn = an * (1.0 - ns) - bn * ns
        i_na = neuron.g_na * ms**3 * hs * (vs - neuron.e_na)
        i_k = neuron.g_k * ns**4 * (vs - neuron.e_k)
        i_l = neuron.g_l * (vs - neuron.e_l)
        dv = -i_na - i_k - i_l + current
        return dv, dm, dh, dn

    for _ in range(10):
        k1 = derivatives(v, m, h, n)
        k2 = derivatives(
            v + 0.5 * neuron.dt * k1[0],
            m + 0.5 * neuron.dt * k1[1],
            h + 0.5 * neuron.dt * k1[2],
            n + 0.5 * neuron.dt * k1[3],
        )
        k3 = derivatives(
            v + 0.5 * neuron.dt * k2[0],
            m + 0.5 * neuron.dt * k2[1],
            h + 0.5 * neuron.dt * k2[2],
            n + 0.5 * neuron.dt * k2[3],
        )
        k4 = derivatives(
            v + neuron.dt * k3[0],
            m + neuron.dt * k3[1],
            h + neuron.dt * k3[2],
            n + neuron.dt * k3[3],
        )
        v += neuron.dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        m += neuron.dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        h += neuron.dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
        n += neuron.dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0
    return v, m, h, n

__all__ = ['np', 'pytest', 'TraubMilesNeuron', 'Population', 'Projection', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count', 'isi', 'firing_rate', '_run', '_rk4_expected_after_call']
