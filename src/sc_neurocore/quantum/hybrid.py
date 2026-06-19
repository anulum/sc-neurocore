# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Simulates a Quantum-Classical Hybrid Layer

from typing import Any
from dataclasses import dataclass
import numpy as np


@dataclass
class QuantumStochasticLayer:
    """
    Simulates a Quantum-Classical Hybrid Layer.
    Input bitstream probability -> Qubit Rotation -> Measurement Probability.

    Mapping: p_in -> theta = p_in * pi
    P_out = |<0|Ry(theta)|0>|^2 = cos^2(theta/2)
    This non-linearity is useful for classification.
    """

    n_qubits: int
    length: int = 1024
    rng_seed: int | None = None

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.rng_seed)

    def forward(self, input_bitstreams: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        input_bitstreams: (n_qubits, length)
        """
        # 1. Decode inputs to probabilities
        p_in = np.mean(input_bitstreams, axis=1)

        # 2. Quantum Rotation (Simulated)
        theta = p_in * np.pi

        # 3. Measurement Probability
        # Probability of measuring |0>
        p_measure = np.cos(theta / 2.0) ** 2

        # 4. Re-encode to bitstream (Collapse)
        # (n_qubits, length)
        rands = self._rng.random((self.n_qubits, self.length))
        out_bits: np.ndarray[Any, Any] = (rands < p_measure[:, None]).astype(np.uint8)

        return out_bits
