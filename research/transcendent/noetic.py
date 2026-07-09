# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header


from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Sign:
    signifier: str  # Word/Image
    signified: str  # Concept
    interpretant: str  # Context/Meaning


class SemioticTriad:
    """
    Simulates Noetic/Semiotic Computing.
    Processing via Meaning Shifts (Metaphor/Metonymy).
    """

    def __init__(self):
        # Knowledge Graph / Ontology
        self.associations: Dict[str, List[str]] = {}

    def learn_association(self, concept: str, related: str):
        if concept not in self.associations:
            self.associations[concept] = []
        self.associations[concept].append(related)

    def interpret(self, sign: Sign) -> Sign:
        """
        Shift meaning based on 'Interpretant'.
        Semiosis: The Interpretant becomes the new Signifier.
        """
        # Look up associations for the interpretant
        context = sign.interpretant
        if context in self.associations:
            # Shift meaning (Metaphor)
            new_concept = self.associations[context][0]  # Simple selection
            return Sign(signifier=context, signified=new_concept, interpretant=sign.signified)
        return sign

    def metaphor_distance(self, start: str, end: str, depth=5) -> int:
        """
        Distance in meaning space (Noetic distance).
        """
        # BFS
        frontier = [(start, 0)]
        visited = set()
        while frontier:
            curr, dist = frontier.pop(0)
            if curr == end:
                return dist
            if dist >= depth:
                continue

            if curr in self.associations:
                for neighbor in self.associations[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        frontier.append((neighbor, dist + 1))
        return -1
