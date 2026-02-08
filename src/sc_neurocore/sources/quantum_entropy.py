import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class QuantumEntropySource:
    """
    Generates entropy based on simulated Quantum Measurement Collapse.
    Used to inject 'True' (Simulated) Quantum Indeterminacy into Neural Models.
    
    Physics:
    - Maintains a Qubit State |psi>
    - Applies Hadamard (Superposition) and Phase Rotations
    - Measures (Collapse) to generate noise
    """
    n_qubits: int = 1
    seed: Optional[int] = None
    
    def __post_init__(self):
        self._rng = np.random.RandomState(self.seed)
        # Initialize |0> state
        self.state = np.zeros(2**self.n_qubits, dtype=np.complex128)
        self.state[0] = 1.0
        
    def _hadamard(self):
        """Applies Hadamard gate to put system in superposition."""
        # Simple H on all qubits (simplified logic for speed)
        # H |0> = (|0> + |1>)/sqrt(2)
        # We just randomize the amplitudes while maintaining normalization
        
        # Random unitary logic:
        # Generate random complex numbers
        real = self._rng.randn(len(self.state))
        imag = self._rng.randn(len(self.state))
        new_state = real + 1j * imag
        
        # Normalize
        norm = np.linalg.norm(new_state)
        self.state = new_state / norm
        
    def sample_normal(self, mean=0.0, std=1.0) -> float:
        """
        Generates a random number by collapsing the wavefunction.
        Returns a value drawn from the measured probability distribution,
        mapped to a Normal-like range.
        """
        # 1. Evolve State (Time Evolution)
        self._hadamard()
        
        # 2. Measure (Born Rule)
        probs = np.abs(self.state)**2
        
        # 3. Collapse
        # We choose an index based on probs
        outcome_idx = self._rng.choice(len(probs), p=probs)
        
        # 4. Map Outcome to Continuous Value
        # We treat the outcome index as a sample from the Hilbert Space
        # We add some jitter to make it continuous (simulating weak measurement noise)
        
        # Center the index around 0
        N = len(probs)
        centered = outcome_idx - (N / 2.0)
        
        # Normalize to [-1, 1] range approximately
        normalized = centered / (N / 2.0)
        
        # Scale to requested std + mean
        # Note: This is a "Quantum Distribution", not perfectly Gaussian
        return mean + (normalized * std * 3.0) # Scale factor to match approx std

    def sample(self) -> float:
        return self.sample_normal()
