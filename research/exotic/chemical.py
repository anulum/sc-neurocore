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
class ReactionDiffusionSolver:
    """
    Chemical Computing using Gray-Scott Reaction-Diffusion.
    """

    width: int
    height: int
    Da: float = 0.16
    Db: float = 0.08
    f: float = 0.060
    k: float = 0.062
    dt: float = 1.0

    def __post_init__(self):
        self.A = np.ones((self.height, self.width))
        self.B = np.zeros((self.height, self.width))

        # Seed - adaptive radius based on grid size
        r = min(5, self.width // 4, self.height // 4)
        if r > 0:
            cx, cy = self.width // 2, self.height // 2
            # Ensure bounds are within grid
            y_start = max(0, cy - r)
            y_end = min(self.height, cy + r)
            x_start = max(0, cx - r)
            x_end = min(self.width, cx + r)
            seed_height = y_end - y_start
            seed_width = x_end - x_start
            self.B[y_start:y_end, x_start:x_end] = 0.25 + 0.25 * np.random.random(
                (seed_height, seed_width)
            )

    def laplacian(self, M):
        # Finite difference Laplacian with periodic boundary
        top = np.roll(M, 1, axis=0)
        bottom = np.roll(M, -1, axis=0)
        left = np.roll(M, 1, axis=1)
        right = np.roll(M, -1, axis=1)
        return top + bottom + left + right - 4 * M

    def step(self):
        La = self.laplacian(self.A)
        Lb = self.laplacian(self.B)

        # Reaction: A + 2B -> 3B
        reaction = self.A * (self.B**2)

        self.A += (self.Da * La - reaction + self.f * (1 - self.A)) * self.dt
        self.B += (self.Db * Lb + reaction - (self.k + self.f) * self.B) * self.dt

        self.A = np.clip(self.A, 0, 1)
        self.B = np.clip(self.B, 0, 1)

    def get_state(self):
        return self.B  # Usually visualize B
