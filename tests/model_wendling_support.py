# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_wendling.py

from __future__ import annotations

"""Full pipeline test for WendlingNeuron (Wendling et al. 2002).

Extended Jansen-Rit: 8 ODEs (4 populations × 2 states). Returns float
(EEG signal = y1 - y2 - y3), not spike. Reproduces epileptiform patterns.
Pipeline limited: float return. Performance: ~59K isolation steps/s."""
import math
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.wendling import WendlingNeuron
from sc_neurocore.network.population import Population

__all__ = ["math", "time", "np", "pytest", "WendlingNeuron", "Population"]
