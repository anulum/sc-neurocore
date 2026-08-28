# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared Rulkov map test support

from __future__ import annotations

"""Shared public-surface support for the Rulkov 2002 piecewise map.

The source event is entry into the rightmost reset branch. The state recurrence
uses ``I`` as the paper's fast control ``beta_n`` and ``sigma`` as its slow
control. No ODE integration is involved.
"""
import numpy as np
import pytest
from sc_neurocore.neurons.models.rulkov_map import RulkovMapNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.analysis.spike_stats.basic import spike_count


def _run(neuron: RulkovMapNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


__all__ = [
    "np",
    "pytest",
    "RulkovMapNeuron",
    "Population",
    "Network",
    "SpikeMonitor",
    "PoissonInput",
    "spike_count",
    "_run",
]
