# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header


import numpy as np
from dataclasses import dataclass


@dataclass
class EventHorizonLayer:
    """
    Simulates information scrambling at a Black Hole Event Horizon.
    Maps volume information to surface area bits (Holographic Principle).
    """

    n_inputs: int
    n_outputs: int  # Represents Surface Area (Entropy S = A/4G)

    def __post_init__(self):
        # Scrambling Matrix: Random Unitary (approximated by Orthogonal)
        # Fast Scrambler: spreading information maximally
        self.scrambler = np.random.normal(0, 1, (self.n_outputs, self.n_inputs))
        # Normalize to preserve 'energy' (information)
        u, s, vh = np.linalg.svd(self.scrambler, full_matrices=False)
        self.unitary_scrambler = u @ vh

    def scramble(self, input_bitstream: np.ndarray) -> np.ndarray:
        """
        Input: (n_inputs, length)
        Output: (n_outputs, length)

        Information is 'smeared' across the output surface.
        """
        # Decode to probabilities
        p_in = np.mean(input_bitstream, axis=1)

        # Unitary Mix
        p_out = np.dot(self.unitary_scrambler, p_in)

        # Softmax-like normalization to keep within [0, 1]
        p_out = np.abs(p_out)
        p_out = p_out / (np.max(p_out) + 1e-9)

        # Re-generate bitstream
        length = input_bitstream.shape[1]
        rands = np.random.random((self.n_outputs, length))
        out_bits = (rands < p_out[:, None]).astype(np.uint8)

        return out_bits
