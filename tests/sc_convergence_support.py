# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_sc_convergence.py

from __future__ import annotations

"""Property-based tests for stochastic computing convergence.

Verifies:
- AND multiplication converges O(1/sqrt(L))
- Sobol encoding converges faster than Bernoulli
- CORDIV quotient is monotonic in numerator
- Correlated inputs violate multiplication correctness
- BitstreamEncoder roundtrip stays within expected bounds
- Popcount is exact (no approximation error)
"""
import numpy as np
import pytest
from sc_neurocore import (
    BitstreamEncoder,
    bitstream_to_probability,
    generate_bernoulli_bitstream,
    generate_sobol_bitstream,
)
from sc_neurocore.utils.bitstreams import sc_divide
from sc_neurocore.utils.rng import RNG
N_TRIALS = 100

__all__ = ['np', 'pytest', 'BitstreamEncoder', 'bitstream_to_probability', 'generate_bernoulli_bitstream', 'generate_sobol_bitstream', 'sc_divide', 'RNG', 'N_TRIALS']
