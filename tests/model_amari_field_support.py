# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_amari_field.py

from __future__ import annotations

"""Full pipeline test for AmariNeuralField (Amari 1977).

Continuous neural field discretised on a periodic N=64 grid. The declared
difference-of-exponentials kernel is locally excitatory and distally
inhibitory, and the source-level output is Amari's Heaviside firing rate.
``step()`` accepts a scalar broadcast or exact-length vector and returns the
active-site fraction.

Performance is matrix-vector dominated at the default grid size.
Network: Population works, Network.run produces spikes (float return
interpreted as non-zero → spike; that compatibility behavior is not a
biological single-neuron event claim)."""
import time
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from sc_neurocore.neurons.models.amari_field import AmariNeuralField
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput


def amari_state(neuron: AmariNeuralField) -> npt.NDArray[np.float64]:
    """Return the initialized state while narrowing the constructor type."""
    state = neuron.u
    assert state is not None
    return state


__all__ = [
    "time",
    "np",
    "pytest",
    "AmariNeuralField",
    "Population",
    "Network",
    "SpikeMonitor",
    "PoissonInput",
    "Any",
    "amari_state",
]
