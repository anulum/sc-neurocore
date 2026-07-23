# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_astrocyte.py

from __future__ import annotations

"""Full pipeline test for AstrocyteModel (Li & Bhatt 1994).

3 ODEs: Ca (cytosolic), h (IP3R de-inactivation), IP3.
Returns float (Ca concentration µM), not int spike.
Ca oscillates at I=0 (range 0.05–0.94 µM). IP3 input drives Ca high.
Performance: ~73K steps/s."""
import time
import numpy as np
import pytest
from sc_neurocore.neurons.models.astrocyte import AstrocyteModel
from sc_neurocore.network.population import Population

__all__ = ['time', 'np', 'pytest', 'AstrocyteModel', 'Population']
