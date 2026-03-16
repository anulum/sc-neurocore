# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from typing import Any
import numpy as np
from typing import Dict


class SpikeToConceptMapper:
    """
    XAI Module: Maps spike patterns to semantic concepts.
    """

    def __init__(self, concept_map: Dict[int, str]):
        self.concept_map = concept_map

    def explain(self, spikes: np.ndarray[Any, Any]) -> str:
        """
        Input: Spike vector (n_neurons,)
        Output: "The agent is thinking about [Concept1, Concept2]"
        """
        active_indices = np.where(spikes > 0)[0]

        concepts = []
        for idx in active_indices:
            if idx in self.concept_map:
                concepts.append(self.concept_map[idx])
            else:
                concepts.append(f"Unknown({idx})")

        if not concepts:
            return "The agent is idle."

        return f"The agent is active on: {', '.join(concepts)}"
