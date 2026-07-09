# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class ReversibleLayer:
    """
    Simulates Reversible (Adiabatic) Logic.
    Uses Toffoli (CCNOT) gates which are universal and reversible.
    (a, b, c) -> (a, b, c XOR (a AND b))
    """

    def toffoli_gate(
        self, a: np.ndarray, b: np.ndarray, c: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Applies Toffoli gate to bitstreams.
        Input: Control a, Control b, Target c
        Output: a, b, c'
        """
        # c' = c XOR (a AND b)
        and_ab = np.bitwise_and(a, b)
        c_prime = np.bitwise_xor(c, and_ab)
        return a, b, c_prime

    def reverse_toffoli(
        self, a: np.ndarray, b: np.ndarray, c_prime: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reverses the Toffoli gate.
        Toffoli is its own inverse!
        """
        return self.toffoli_gate(a, b, c_prime)

    def forward(self, input_a: np.ndarray, input_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulates an AND gate reversibly.
        We need a 'c' ancilla initialized to 0.
        Output is (a, b, result). We return result and garbage (a,b).
        """
        # Ancilla 'c' is 0
        c = np.zeros_like(input_a)

        # Apply Toffoli
        a_out, b_out, res = self.toffoli_gate(input_a, input_b, c)

        # In reversible computing, we keep a_out and b_out to uncompute later
        # Here we just return the 'useful' result and 'garbage'
        return res, (a_out, b_out)
