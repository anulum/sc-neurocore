# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_courage_nekorkin_map.py

from __future__ import annotations

"""Full pipeline test for CourageNekorkinMapNeuron (Courbage-Nekorkin-Vdovin 2007).

Canonical discontinuous two-dimensional spiking map (Chaos 17:043109;
arXiv:0712.2097, eqs. 3-5):

    x(n+1) = x(n) + F(x(n)) - y(n) - beta*H(x(n) - d) + I
    y(n+1) = y(n) + eps*(x(n) - J)

    F(x) = -m0*x        for x <= Jmin
           m1*(x - a)   for Jmin < x < Jmax
           -m0*(x - 1)  for x >= Jmax
    H(z) = 1 for z >= 0, else 0
    Jmin = a*m1/(m0 + m1), Jmax = (m0 + a*m1)/(m0 + m1)

The default parameters (m0=0.0864, m1=0.65, a=0.2 from figure 1; d=0.235, J=0.2,
beta=0.085, eps=0.02 inside the B^+ invariant-region triangle) place the model in
the published chaotic spiking-bursting regime. The map has NO clip: it stays
bounded by its own invariant attractor.
"""
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.courage_nekorkin_map import CourageNekorkinMapNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi


def _run(neuron: CourageNekorkinMapNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


def _breakpoints(m0=0.0864, m1=0.65, a=0.2):
    am1 = a * m1
    den = m0 + m1
    return am1 / den, (m0 + am1) / den


__all__ = [
    "time",
    "np",
    "pytest",
    "CourageNekorkinMapNeuron",
    "Population",
    "Projection",
    "Network",
    "SpikeMonitor",
    "PoissonInput",
    "spike_count",
    "firing_rate",
    "isi",
    "_run",
    "_breakpoints",
]
