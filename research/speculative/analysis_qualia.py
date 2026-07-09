# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header


import logging
import numpy as np
from transcendent.noetic import SemioticTriad, Sign
from sc_neurocore.viz.neuro_art import NeuroArtGenerator

logger = logging.getLogger(__name__)


class QualiaTuringTest:
    """
    Test for subjective experience (or simulation thereof).
    Can the agent describe a novel internal state using meaningful metaphors?
    """

    def __init__(self, semiotics: SemioticTriad):
        self.semiotics = semiotics
        self.art_gen = NeuroArtGenerator()

    def administer_test(self, state_vector: np.ndarray) -> bool:
        """
        1. Generate Art from state (The 'Qualia').
        2. Ask agent to describe it (The 'Report').
        3. Check if report is a valid metaphor.
        """
        # 1. Internal Representation
        # img = self.art_gen.generate_visual(state_vector)
        # (We don't analyze the pixels, we analyze the mapping logic)

        # 2. Agent's Report (Simulated)
        # The agent must map the high-dimensional state to a Concept.
        # We simulate this by mapping the dominant state feature to a Sign.
        dominant_feature = np.argmax(state_vector)
        # Mock mapping: 0->Red/Fire, 1->Blue/Water, etc.
        concept_map = {0: "Fire", 1: "Ocean", 2: "Void"}
        base_concept = concept_map.get(dominant_feature, "Chaos")

        sign = Sign("InternalState", base_concept, "Emotion")

        # 3. Metaphorical Shift
        # The agent 'feels' the state and shifts the meaning
        description = self.semiotics.interpret(sign)

        logger.info("Qualia Test: State Peak %d -> Concept '%s'.", dominant_feature, base_concept)
        logger.info(
            "    Agent Description: '%s' (via %s)", description.signified, description.signifier
        )

        # 4. Evaluation
        # If the description is a valid association in the Semiotic Graph, pass.
        # Check if description.signified is linked to base_concept
        dist = self.semiotics.metaphor_distance(base_concept, description.signified)

        if dist >= 0 and description.signified != base_concept:
            logger.info("    Result: PASS. Agent generated valid metaphorical description.")
            return True
        elif description.signified == base_concept:
            logger.info("    Result: INCONCLUSIVE. Literal description (Zombie behavior).")
            return False
        else:
            logger.info("    Result: FAIL. Incoherent hallucination.")
            return False
