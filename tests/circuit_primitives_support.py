# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_circuit_primitives.py

from __future__ import annotations

"""Tests for lateral inhibition and winner-take-all circuits."""
import numpy as np
from sc_neurocore.layers.circuit_primitives import LateralInhibition, WinnerTakeAll

__all__ = ['np', 'LateralInhibition', 'WinnerTakeAll']
