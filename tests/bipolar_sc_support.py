# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_bipolar_sc.py

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.core.bipolar import (
    bipolar_decode,
    bipolar_encode,
    bipolar_mac,
    bipolar_multiply,
    bipolar_sc_layer,
    float_to_bipolar_weights,
)

__all__ = [
    "np",
    "pytest",
    "bipolar_decode",
    "bipolar_encode",
    "bipolar_mac",
    "bipolar_multiply",
    "bipolar_sc_layer",
    "float_to_bipolar_weights",
]
