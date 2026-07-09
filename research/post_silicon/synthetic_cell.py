# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header


from dataclasses import dataclass


@dataclass
class CellularComputer:
    """
    Simulates computing inside a Synthetic Cell.
    Logic is driven by Brownian motion collisions between molecules and enzymes.
    """

    n_molecules_a: int = 0
    n_molecules_b: int = 0
    reaction_rate: float = 0.1

    def step(self, inject_a: int, inject_b: int) -> int:
        """
        Inject reactants, simulate collisions, release product C.
        Returns amount of C produced (Spike output).
        """
        self.n_molecules_a += inject_a
        self.n_molecules_b += inject_b

        # Collision probability depends on concentrations
        # P_react ~ k * [A] * [B]

        potential_reactions = int(self.reaction_rate * self.n_molecules_a * self.n_molecules_b)

        # Limit by available substrate
        actual_reactions = min(potential_reactions, self.n_molecules_a, self.n_molecules_b)

        # Consume
        self.n_molecules_a -= actual_reactions
        self.n_molecules_b -= actual_reactions

        return actual_reactions
