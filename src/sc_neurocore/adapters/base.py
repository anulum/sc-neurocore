"""
Base interface for sc-neurocore adapters.

Adapters map domain-specific dynamics (Biology, Physics, etc.) into 
stochastic bitstreams and JAX-accelerated kernels.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np
from ..accel.jax_backend import jnp


class BaseStochasticAdapter(ABC):
    """
    Abstract base class for all domain-specific adapters.
    """

    @abstractmethod
    def encode(self, state: Any) -> jnp.ndarray:
        """Map domain state to stochastic bitstreams."""
        pass

    @abstractmethod
    def step_jax(self, dt: float, inputs: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """The JAX-accelerated mathematical kernel for the domain dynamics."""
        pass

    @abstractmethod
    def decode(self, bitstreams: jnp.ndarray) -> Any:
        """Map stochastic bitstreams back to domain-specific observables."""
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Return domain-specific metrics (e.g. Coherence, Concentration)."""
        pass
