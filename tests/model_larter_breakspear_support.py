# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_larter_breakspear.py

from __future__ import annotations

"""Full pipeline test for LarterBreakspearNeuron (Breakspear et al. 2003).

Neural mass with conductance-based ion channels (TVB model):
dV = -I_Ca - I_Na - I_K - I_L + I_ext + coupling + a_ee·V
dw = φ·(m_K(V) - w) / τ_K
dz = b·(V + 0.5 - z)

4 currents with tanh activations: m_Ca, m_Na, m_K.
Returns V (float), not binary spike. Used in whole-brain modelling.
Pipeline and performance contract tests live in this module-specific file."""
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.larter_breakspear import LarterBreakspearNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput

__all__ = [
    "time",
    "np",
    "pytest",
    "LarterBreakspearNeuron",
    "Population",
    "Projection",
    "Network",
    "SpikeMonitor",
    "PoissonInput",
]
