# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_pinsky_rinzel.py

from __future__ import annotations

"""Full pipeline test for PinskyRinzelNeuron (Pinsky & Rinzel 1994).

Two-compartment CA3 pyramidal cell integrated with fourth-order Runge-Kutta:
soma (fast Na/K-DR) coupled to dendrite (Ca, K-AHP, K-C). Eight states
``(v_s, v_d, h, n, s, c, q, ca)``; ``step(current_soma, current_dend)`` has a
dual-input signature. The model fires repetitively at low somatic drive and
enters depolarisation block (Na inactivation) at high drive, giving a
non-monotonic f-I relation. Reference: PR1994 / ModelDB 35358.
"""
import numpy as np
import pytest
from sc_neurocore.neurons.models.pinsky_rinzel import PinskyRinzelNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count
def _run(
    neuron: PinskyRinzelNeuron, current_soma: float, steps: int, current_dend: float = 0.0
) -> list[int]:
    """Return the indices of steps on which a somatic spike was registered."""
    return [t for t in range(steps) if neuron.step(current_soma, current_dend) == 1]

__all__ = ['np', 'pytest', 'PinskyRinzelNeuron', 'Population', 'Network', 'SpikeMonitor', 'PoissonInput', 'spike_count', '_run', '__all__']
