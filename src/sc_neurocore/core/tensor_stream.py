import numpy as np
from dataclasses import dataclass


@dataclass
class TensorStream:
    """
    Unified Data Structure for sc-neurocore.
    Handles automatic conversion between domains.
    """

    data: np.ndarray
    domain: str  # 'prob', 'bitstream', 'quantum', 'spike'

    @classmethod
    def from_prob(cls, probs: np.ndarray):
        return cls(data=probs, domain="prob")

    def to_bitstream(self, length: int = 1024) -> np.ndarray:
        if self.domain == "bitstream":
            return self.data
        if self.domain == "prob":
            # Vectorized Bernoulli
            rands = np.random.random((*self.data.shape, length))
            return (rands < self.data[..., None]).astype(np.uint8)
        raise ValueError(f"Cannot convert {self.domain} to bitstream directly.")

    def to_prob(self) -> np.ndarray:
        if self.domain == "prob":
            return self.data
        if self.domain == "bitstream":
            # Mean along the last axis (time)
            return np.mean(self.data, axis=-1)
        if self.domain == "quantum":
            # Born Rule: p = |beta|^2
            return np.abs(self.data[..., 1]) ** 2
        return self.data  # Fallback

    def to_quantum(self) -> np.ndarray:
        if self.domain == "quantum":
            return self.data
        p = self.to_prob()
        # Map p to state vector: cos(theta/2)|0> + sin(theta/2)|1>
        # theta = p * pi
        theta = p * np.pi
        alpha = np.cos(theta / 2.0)
        beta = np.sin(theta / 2.0)
        # Result: (..., 2) complex array
        return np.stack([alpha, beta], axis=-1).astype(complex)
