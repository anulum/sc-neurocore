# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_de_schutter_purkinje.py

from __future__ import annotations

"""Full pipeline test for DeSchutterPurkinjeNeuron.

Complex Purkinje cell model. Needs very high current (I≥500) for even
1 transient spike at default params. Converges to stable fixed point.
Performance: ~4.8K steps/s (complex multi-current model)."""
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.de_schutter_purkinje import DeSchutterPurkinjeNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput


def _run(neuron: DeSchutterPurkinjeNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


__all__ = [
    "time",
    "np",
    "pytest",
    "DeSchutterPurkinjeNeuron",
    "Population",
    "Network",
    "SpikeMonitor",
    "PoissonInput",
    "_run",
]
