# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Simulates hardware faults in Stochastic Computing bitstreams

from __future__ import annotations

from typing import Any

import numpy as np


class FaultInjector:
    """
    Simulates hardware faults in Stochastic Computing bitstreams.
    """

    @staticmethod
    def inject_bit_flips(
        bitstream: np.ndarray[Any, Any], error_rate: float
    ) -> np.ndarray[Any, Any]:
        """
        Randomly flips bits with probability 'error_rate'.
        """
        if error_rate <= 0:
            return bitstream

        # Generate error mask (1 where error occurs)
        # Using numpy for speed
        mask = np.random.random(bitstream.shape) < error_rate

        # XOR with mask flips the bits where mask is 1
        # bitstream is uint8 {0,1}
        # We need to ensure we don't go out of bounds (0/1)
        # 0 ^ 1 = 1
        # 1 ^ 1 = 0
        # 0 ^ 0 = 0
        # 1 ^ 0 = 1

        corrupted = np.bitwise_xor(bitstream.astype(bool), mask)
        flipped: np.ndarray[Any, Any] = corrupted.astype(np.uint8)
        return flipped

    @staticmethod
    def inject_stuck_at(
        bitstream: np.ndarray[Any, Any], fault_rate: float, value: int
    ) -> np.ndarray[Any, Any]:
        """
        Simulates Stuck-At-0 or Stuck-At-1 faults.
        """
        mask = np.random.random(bitstream.shape) < fault_rate
        corrupted = bitstream.copy()
        corrupted[mask] = value
        return corrupted
