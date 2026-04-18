# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — A minimal token-level text generator for SC

from typing import Any
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class SCTextGenerator:
    """
    A minimal token-level text generator for SC.
    Maps probability distributions over vocabulary to tokens.
    """

    vocab: List[str]

    def generate_token(self, prob_dist: np.ndarray[Any, Any]) -> str:
        """
        Input: prob_dist (len(vocab),)
        Returns: selected token based on probability.
        """
        # Ensure it sums to 1
        dist = prob_dist / (np.sum(prob_dist) + 1e-9)
        idx = np.random.choice(len(self.vocab), p=dist)
        return self.vocab[idx]

    def generate_sequence(self, length: int) -> str:
        """
        Generate a random sequence of tokens.
        """
        tokens = [
            self.generate_token(np.random.dirichlet(np.ones(len(self.vocab))))
            for _ in range(length)
        ]
        return " ".join(tokens)
