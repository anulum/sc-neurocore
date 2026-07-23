# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_amari_field.py

from __future__ import annotations

"""Full pipeline test for AmariNeuralField (Amari 1977).

Continuous neural field discretised on N=64 nodes. Mexican-hat kernel
w(x) = A·exp(-a|x|) - B·exp(-b|x|). step() takes NDArray input,
returns float (mean activation). FFT-based convolution.

FINDING: default params (a_exc=1.5, b_inh=0.75) → kernel sum=4.5
→ unstable (field diverges under persistent input). Balanced params
(a_exc=0.5, b_inh=0.5) → kernel sum≈0.96 → stable bump.

Performance: ~19K isolation steps/s (FFT-dominated).
Network: Population works, Network.run produces spikes (float return
interpreted as non-zero → spike)."""
import time
import numpy as np
from sc_neurocore.neurons.models.amari_field import AmariNeuralField
from sc_neurocore.network.population import Population
from sc_neurocore.network.network import Network
from sc_neurocore.network.monitor import SpikeMonitor
from sc_neurocore.network.stimulus import PoissonInput

__all__ = ['time', 'np', 'AmariNeuralField', 'Population', 'Network', 'SpikeMonitor', 'PoissonInput']
