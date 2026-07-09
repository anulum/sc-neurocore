# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header


from dataclasses import dataclass
import numpy as np


@dataclass
class SymbiosisProtocol:
    """
    Human-AI Symbiosis Interface.
    Translates Semantics <-> Bitstreams.
    """

    def encode_thought(self, semantic_vector: np.ndarray, urgency: float) -> np.ndarray:
        """
        Human -> Machine.
        Encodes a thought vector into a high-priority bitstream.
        """
        # Map vector [-1, 1] to prob [0, 1]
        probs = (semantic_vector + 1.0) / 2.0
        # Boost probability by urgency (Attention)
        probs = np.clip(probs * (1.0 + urgency), 0, 1)

        # Generate spike train
        rands = np.random.random(probs.shape)
        bits = (rands < probs).astype(np.uint8)
        return bits

    def decode_sensation(self, bitstream: np.ndarray) -> str:
        """
        Machine -> Human.
        Decodes a result bitstream into a sensation/concept.
        """
        # Calculate 'activation'
        mean_activity = np.mean(bitstream)

        if mean_activity > 0.8:
            return "Sensation: FLASH OF INSIGHT (High Confidence)"
        elif mean_activity > 0.5:
            return "Sensation: Vague Intuition"
        elif mean_activity > 0.2:
            return "Sensation: Uncertainty"
        else:
            return "Sensation: Silence"
