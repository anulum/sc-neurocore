# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_cordiv_division.py

from __future__ import annotations

"""Tests for sc_divide (Li et al. 2014) and Hoeffding adaptive_length."""
import numpy as np
from sc_neurocore.utils.bitstreams import (
    generate_bernoulli_bitstream,
    sc_divide,
    adaptive_length,
)
from sc_neurocore.utils.rng import RNG

__all__ = ['np', 'generate_bernoulli_bitstream', 'sc_divide', 'adaptive_length', 'RNG']
