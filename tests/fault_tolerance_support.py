# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_fault_tolerance.py

from __future__ import annotations

"""Fault tolerance benchmark: inject bit-flip errors and measure accuracy degradation.

SC's fundamental advantage: a single bit-flip in a bitstream changes the
encoded probability by only 1/L, whereas a bit-flip in a fixed-point register
can corrupt the MSB and cause catastrophic error. This test suite quantifies
that advantage.
"""
import numpy as np
import pytest
from sc_neurocore.utils.bitstreams import (
    generate_bernoulli_bitstream,
    bitstream_to_probability,
)
from sc_neurocore.utils.fault_injection import FaultInjector
from sc_neurocore.layers.hardware_aware import HardwareAwareSCLayer

__all__ = [
    "np",
    "pytest",
    "generate_bernoulli_bitstream",
    "bitstream_to_probability",
    "FaultInjector",
    "HardwareAwareSCLayer",
]
