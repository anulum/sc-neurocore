# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Unified Data Structure for sc-neurocore

"""Tensor container with conversions between probability, bitstream, and quantum domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TensorStream:
    """Unified tensor container for sc-neurocore.

    Handles automatic conversion between the probability, bitstream, and
    quantum domains.
    """

    data: np.ndarray[Any, Any]
    domain: str  # 'prob', 'bitstream', 'quantum', 'spike'

    @classmethod
    def from_prob(cls, probs: np.ndarray[Any, Any]) -> TensorStream:
        """Create a tensor stream whose data is already in probability form."""
        return cls(data=probs, domain="prob")

    def to_bitstream(self, length: int = 1024) -> np.ndarray[Any, Any]:
        """Convert probability-domain data into Bernoulli bitstreams."""
        if self.domain == "bitstream":
            return self.data
        if self.domain == "prob":
            # Vectorized Bernoulli
            rands = np.random.random((*self.data.shape, length))
            return (rands < self.data[..., None]).astype(np.uint8)
        raise ValueError(f"Cannot convert {self.domain} to bitstream directly.")

    def to_prob(self) -> np.ndarray[Any, Any]:
        """Convert supported domains into probability-domain tensors."""
        if self.domain == "prob":
            return self.data
        if self.domain == "bitstream":
            # Mean along the last axis (time)
            mean_prob: np.ndarray[Any, Any] = np.mean(self.data, axis=-1)
            return mean_prob
        if self.domain == "quantum":
            # Born Rule: p = |beta|^2
            born_prob: np.ndarray[Any, Any] = np.abs(self.data[..., 1]) ** 2
            return born_prob
        return self.data  # Fallback

    def to_quantum(self) -> np.ndarray[Any, Any]:
        """Convert probability-domain data into two-amplitude quantum encoding."""
        if self.domain == "quantum":
            return self.data
        p = np.clip(self.to_prob(), 0.0, 1.0)
        # Amplitude encoding: |psi> = sqrt(1-p)|0> + sqrt(p)|1>
        # Measurement P(|1>) = |beta|^2 = p — preserves SC probability exactly.
        # Matches CategoryTheoryBridge.stochastic_to_quantum().
        alpha = np.sqrt(1.0 - p)
        beta = np.sqrt(p)
        return np.stack([alpha, beta], axis=-1).astype(complex)
