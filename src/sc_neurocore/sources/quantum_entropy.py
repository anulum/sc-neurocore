# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Generates entropy based on simulated Quantum Measurement

"""Simulated quantum-measurement entropy source for stochastic inputs.

This module maintains a small classical state-vector simulation, applies
Hadamard-style mixing before measurement, and converts seeded measurement
outcomes into deterministic pseudo-random samples. It does not claim access to
physical quantum hardware or certified quantum random numbers.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class QuantumEntropySource:
    """Simulated quantum-measurement entropy source.

    Injects simulated quantum indeterminacy into neural models by maintaining
    a qubit state ``|psi>``, applying Hadamard superposition and phase
    rotations, and measuring (collapsing) the state to generate noise.
    """

    n_qubits: int = 1
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        """Initialise the RNG and reset the qubit register to ``|0>``."""
        self._rng = np.random.RandomState(self.seed)
        # Initialize |0> state
        self.state = np.zeros(2**self.n_qubits, dtype=np.complex128)
        self.state[0] = 1.0

    def _hadamard(self) -> None:
        """Apply Hadamard gate H = (1/√2)[[1,1],[1,-1]] to each qubit."""
        H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        result = self.state.copy()
        n = self.n_qubits
        dim = 2**n
        for q in range(n):
            new_result = np.zeros(dim, dtype=np.complex128)
            block = 2 ** (n - q)
            half = block // 2
            for start in range(0, dim, block):
                for i in range(half):
                    a = result[start + i]
                    b = result[start + half + i]
                    new_result[start + i] = H[0, 0] * a + H[0, 1] * b
                    new_result[start + half + i] = H[1, 0] * a + H[1, 1] * b
            result = new_result
        self.state = result

    def _measure(self) -> int:
        """Apply Hadamard, measure via Born rule, collapse state."""
        self._hadamard()
        probs = np.abs(self.state) ** 2
        idx = self._rng.choice(len(probs), p=probs)
        # Wavefunction collapse to measured basis state
        self.state = np.zeros_like(self.state)
        self.state[idx] = 1.0
        return int(idx)

    def sample_normal(self, mean: float = 0.0, std: float = 1.0) -> float:
        """
        Two independent measurements → Box-Muller → Gaussian sample.

        Discrete outcomes dithered with uniform jitter for continuous input.
        """
        N = len(self.state)

        u1 = (self._measure() + self._rng.uniform()) / N
        u1 = np.clip(u1, 1e-10, 1.0 - 1e-10)
        u2 = (self._measure() + self._rng.uniform()) / N

        z = np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)
        return float(mean + z * std)

    def sample(self) -> float:
        """Return one default normal sample from the simulated measurement source."""
        return self.sample_normal()
