# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_pospischil.py

from __future__ import annotations

"""Full pipeline test for PospischilNeuron (Pospischil et al. 2008).

Minimal HH model for cortical cell types. Default: RS pyramidal (g_m=0.07).
I_M (slow K⁺) provides spike-frequency adaptation.
Cell type variants: RS (g_m=0.07), FS (g_m=0), IB (g_m=0.03)."""
import numpy as np
import pytest
from sc_neurocore.neurons.models.pospischil import PospischilNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count
def _run(neuron: PospischilNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]

__all__ = ['np', 'pytest', 'PospischilNeuron', 'Population', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count', '_run', '__all__']
