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
class TimeCrystalLayer:
    """
    Simulates a Discrete Time Crystal (DTC).
    Exhibits stable sub-harmonic oscillations (Period Doubling).
    """

    n_spins: int
    interaction_strength: float = 1.0
    disorder_strength: float = 0.5

    def __post_init__(self):
        # Initial states {-1, 1}
        self.spins = np.random.choice([-1, 1], self.n_spins).astype(np.float32)
        # Random disorder (static)
        self.disorder = np.random.uniform(
            -self.disorder_strength, self.disorder_strength, self.n_spins
        )

    def drive(self, flip_pulse: bool = True):
        """
        One cycle of the DTC drive:
        1. Discrete Flip (Pi-pulse)
        2. Interaction & Disorder (Unitary evolution)
        """
        # Step 1: Flip (approximate Pi-pulse)
        if flip_pulse:
            self.spins *= -1.0

        # Step 2: Local Interactions
        # H = Sum J*S_i*S_{i+1} + Sum h_i*S_i
        # For simplicity, we update phase/orientation
        # S_i = S_i * exp(j * H)
        # Here we use a simplified map:

        # Neighbor interaction (Circular)
        neighbors = np.roll(self.spins, 1)
        interaction = self.interaction_strength * neighbors

        # Total local field
        h = interaction + self.disorder

        # Evolution: Rotation by local field
        # In a real DTC, this is a unitary exp(-iHt)
        # Here we simulate rotation in 1D space:
        self.spins = np.cos(np.arccos(self.spins) + h)

        return self.spins

    def get_bitstream(self, cycles: int) -> np.ndarray:
        """
        Generates bitstream over multiple drive cycles.
        Should show 1, -1, 1, -1 (2T period) even with disorder.
        """
        history = []
        for i in range(cycles):
            state = self.drive(flip_pulse=True)
            # Map back to {0, 1}
            history.append((state[0] > 0).astype(np.uint8))
        return np.array(history)
