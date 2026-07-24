# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_cfc.py

from __future__ import annotations

"""Full pipeline test for ClosedFormContinuousNeuron (Hasani et al. 2022).

Analytical ODE solution: x = x·decay + f_target·(1-decay).
f_target = tanh(w_x·x + w_in·I) is bounded ∈ (-1, 1).
FINDING: default v_threshold=1.0 unreachable (tanh < 1). Lower
threshold (0.5–0.95) enables spiking. Performance: ~68K steps/s."""
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.cfc import ClosedFormContinuousNeuron
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput


def _run(neuron: ClosedFormContinuousNeuron, current: float, steps: int) -> list[int]:
    return [t for t in range(steps) if neuron.step(current) == 1]


__all__ = [
    "time",
    "np",
    "pytest",
    "ClosedFormContinuousNeuron",
    "Population",
    "Network",
    "SpikeMonitor",
    "PoissonInput",
    "_run",
]
