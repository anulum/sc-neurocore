# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Base interface for sc-neurocore adapters

"""
Base interface for sc-neurocore adapters.

Adapters map domain-specific dynamics (Biology, Physics, etc.) into
stochastic bitstreams and JAX-accelerated kernels.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..accel.jax_backend import jnp


class BaseStochasticAdapter(ABC):
    """
    Abstract base class for all domain-specific adapters.
    """

    @abstractmethod
    def encode(self, state: Any) -> jnp.ndarray:
        """Map domain state to stochastic bitstreams."""
        ...

    @abstractmethod
    def step_jax(self, dt: float, inputs: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """The JAX-accelerated mathematical kernel for the domain dynamics."""
        ...

    @abstractmethod
    def decode(self, bitstreams: jnp.ndarray) -> Any:
        """Map stochastic bitstreams back to domain-specific observables."""
        ...

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Return domain-specific metrics (e.g. Coherence, Concentration)."""
        ...
