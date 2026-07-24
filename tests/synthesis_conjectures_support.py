# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_synthesis_conjectures.py

from __future__ import annotations

"""Tests for the 3 "worth testing" conjectures from SYNTHESIS_REALITY_CHECK:
1. SC-FIM analogy: longer L → lower encoding error (necessary condition)
2. STDP-FIM competition: both active simultaneously, weights diverge
3. Coherence restoration: FIM warm-up after population reset

Plus regression tests for the Lazarus phase gap fix.
All claims clearly scoped — no overclaiming.
"""
import os
import tempfile
import numpy as np
from sc_neurocore import StochasticLIFNeuron
from sc_neurocore import BitstreamEncoder, bitstream_to_probability
from sc_neurocore.network.population import Population
from sc_neurocore.network.projection import Projection
from sc_neurocore.network.network import Network
from sc_neurocore.network.stimulus import PoissonInput
from sc_neurocore.identity.substrate import IdentitySubstrate
from sc_neurocore.identity.checkpoint import Checkpoint

__all__ = [
    "os",
    "tempfile",
    "np",
    "StochasticLIFNeuron",
    "BitstreamEncoder",
    "bitstream_to_probability",
    "Population",
    "Projection",
    "Network",
    "PoissonInput",
    "IdentitySubstrate",
    "Checkpoint",
]
