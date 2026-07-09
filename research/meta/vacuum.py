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
class VacuumNoiseSource:
    """
    Simulates harvesting computation from Vacuum Fluctuations (Zero Point Energy).
    Uses the Casimir effect logic to correlate noise streams.
    """

    dimension: int
    plate_distance: float = 1.0  # Smaller distance -> higher energy/correlation

    def generate_virtual_bits(self, length: int) -> np.ndarray:
        """
        Produces bitstreams derived from simulated quantum fluctuations.
        """
        # 1. Generate Raw Vacuum Noise (Gaussian White Noise)
        noise = np.random.normal(0, 1.0, (self.dimension, length))

        # 2. Apply Casimir-like correlation
        # The energy density is inversely proportional to distance^4
        energy_density = 1.0 / (self.plate_distance**4)

        # Correlation logic: nearby channels influence each other's fluctuations
        correlated_noise = noise + np.roll(noise, 1, axis=0) * (0.1 * energy_density)

        # 3. Rectification (Harvesting)
        # Convert balanced noise to biased bitstream (Probability p > 0.5)
        # This simulates the extraction of useful work/entropy
        harvested_probs = 0.5 + 0.1 * np.tanh(correlated_noise)

        rands = np.random.random(harvested_probs.shape)
        bits = (rands < harvested_probs).astype(np.uint8)

        return bits
