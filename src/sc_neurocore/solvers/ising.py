from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class StochasticIsingGraph:
    """
    Quantum-Inspired Ising Machine Solver.

    Spins S_i in {-1, 1} (mapped to 0, 1 for SC).
    Energy E = -Sum(J_ij * S_i * S_j) - Sum(h_i * S_i).
    Goal: Find configuration that minimizes E.
    """

    num_spins: int
    J: np.ndarray  # Coupling matrix (symmetric, zero diagonal)
    h: np.ndarray  # Bias vector
    temperature: float = 1.0
    anneal_rate: float = 0.99

    def __post_init__(self):
        # Initialize spins randomly
        self.spins = np.random.randint(0, 2, self.num_spins).astype(np.int8)
        # Convert 0/1 to -1/1 for physics calc
        self.bipolar_spins = 2 * self.spins - 1

    def step(self):
        """
        Perform one Metropolis-Hastings update step (parallel / cellular automaton style).
        """
        # Calculate local field H_i = Sum(J_ij * S_j) + h_i
        # Using matrix multiplication
        local_field = np.dot(self.J, self.bipolar_spins) + self.h

        # Calculate Energy Difference Delta_E if we flip S_i
        # Delta_E = 2 * S_i * H_i
        # (Physics convention)

        delta_E = 2 * self.bipolar_spins * local_field

        # Probability of flipping: P = min(1, exp(-Delta_E / T))
        # If Delta_E < 0 (flip reduces energy), P=1 (always flip, greedy)
        # If Delta_E > 0 (flip increases energy), P = exp(...)

        # Vectorized probability calculation
        flip_prob = np.exp(-delta_E / self.temperature)
        flip_prob = np.minimum(1.0, flip_prob)

        # Determine flips
        random_draws = np.random.random(self.num_spins)
        should_flip = random_draws < flip_prob

        # Apply flips
        # Flip -1 to 1 and 1 to -1: S_new = -S_old
        self.bipolar_spins[should_flip] *= -1

        # Update 0/1 representation
        self.spins = (self.bipolar_spins + 1) // 2

        # Anneal
        self.temperature *= self.anneal_rate

        return self.get_energy()

    def get_energy(self) -> float:
        """Calculate global energy."""
        # E = -0.5 * S^T * J * S - h^T * S
        # Factor 0.5 because J_ij is counted twice in full matrix sum
        interaction = -0.5 * np.dot(self.bipolar_spins, np.dot(self.J, self.bipolar_spins))
        bias = -np.dot(self.h, self.bipolar_spins)
        return interaction + bias

    def get_config(self) -> np.ndarray:
        return self.spins
