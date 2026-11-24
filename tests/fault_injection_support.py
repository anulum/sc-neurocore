# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_fault_injection.py

"""Shared names and a TMR majority vote for the fault-injection tests."""

from __future__ import annotations

import numpy.typing as npt
import numpy as np
from sc_neurocore.utils.bitstreams import generate_bernoulli_bitstream
from sc_neurocore.utils.fault_injection import FaultInjector
from sc_neurocore.utils.rng import RNG


def _majority_vote(
    a: npt.NDArray[np.uint8], b: npt.NDArray[np.uint8], c: npt.NDArray[np.uint8]
) -> npt.NDArray[np.uint8]:
    voted: npt.NDArray[np.uint8] = ((a & b) | (a & c) | (b & c)).astype(np.uint8)
    return voted


__all__ = ["np", "generate_bernoulli_bitstream", "FaultInjector", "RNG", "_majority_vote"]
