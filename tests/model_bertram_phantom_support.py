# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_bertram_phantom.py

from __future__ import annotations

"""Module-specific tests for BertramPhantomBurster (Bertram et al. 2008).

Dual slow variable phantom burster (pancreatic β-cell model).
C dV/dt = -(I_Ca + I_K + I_s1 + I_s2 + I_L) + I_ext
ds1/dt = (s1_inf(V) - s1) / tau_s1    (tau=20000)
ds2/dt = (s2_inf(V) - s2) / tau_s2    (tau=100000)

Boltzmann: σ(v, vh, k) = 1/(1+exp((vh-v)/k)).
Five ionic currents: I_Ca (m_inf-gated), I_K (n_inf-gated),
I_s1 (s1-gated, slow), I_s2 (s2-gated, ultra-slow), I_L (leak).
Phantom slow manifold: bursting can emerge from dual slow interaction in
appropriate parameter regimes. Current tests validate RK4 integration,
module wiring, and bounded runtime under CI instrumentation."""
import math
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.bertram_phantom import BertramPhantomBurster
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count, firing_rate, isi
def _run(neuron: BertramPhantomBurster, current: float, steps: int) -> list[int]:
    """Collect spike times from isolated neuron."""
    return [t for t in range(steps) if neuron.step(current) == 1]
def _boltz(v: float, vh: float, k: float) -> float:
    """Reference Boltzmann sigmoid for analytical cross-checks."""
    return 1.0 / (1.0 + np.exp((vh - v) / k))

__all__ = ['math', 'time', 'np', 'pytest', 'BertramPhantomBurster', 'Population', 'Projection', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count', 'firing_rate', 'isi', '_run', '_boltz']
