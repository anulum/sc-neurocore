# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L11 Noospheric-Cultural-Informational Adapter (JAX

"""
SCPN L11: Noospheric-Cultural-Informational Adapter (JAX Implementation)
========================================================================

This module implements the JAX-accelerated uplift of Layer 11, focusing on
the Noosphere-Technosphere Hybrid System (NTHS) Hamiltonian, social
polarization dynamics, and memetic percolation thresholds described in Paper 11.

Key Equations:
- NTHS Hamiltonian: H = -sum(J_ij * sigma_i * sigma_j) - sum(h_i * sigma_i)
- Memetic SIR: dS/dt = -beta * S * I / N; dI/dt = beta * S * I / N - gamma * I
- Cultural Percolation: Global_consensus emerges at p > 0.59
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..base import BaseStochasticAdapter
from ._jax_compat import jnp, make_rng, maybe_jit, split_rng, uniform


@dataclass
class L11_HolonomicParameters:
    """Parameters derived from Paper 11 and NTHS specifications."""

    n_nodes: int = 100
    bitstream_length: int = 1024

    # Spin-Glass Constants
    j_coupling: float = 0.5  # Symbolic interaction strength
    h_bias: float = 0.1  # Algorithmic bias / Forcing field

    # Memetic Constants
    beta_infection: float = 0.2  # Rate of memetic spread
    gamma_recovery: float = 0.05  # Rate of memetic decay (forgetting)


class L11_NoosphericAdapter(BaseStochasticAdapter):
    """
    JAX-traceable adapter for the SCPN Noospheric layer.
    """

    def __init__(self, params: Optional[L11_HolonomicParameters] = None, seed: int = 411) -> None:
        self.params = params or L11_HolonomicParameters()
        self.rng_key = make_rng(seed)

        # State: Cultural Spins (-1 to +1, represented as 0 to 1 probabilities)
        self.spins = jnp.full((self.params.n_nodes,), 0.5)
        # State: Information Density
        self.info_density = jnp.zeros((self.params.n_nodes,))

    def encode(self, domain_state: Any) -> jnp.ndarray:
        """
        Maps cultural spins to stochastic bitstreams.
        """
        self.rng_key, subkey = split_rng(self.rng_key)
        rands = uniform(subkey, (self.params.n_nodes, self.params.bitstream_length))
        bitstreams: jnp.ndarray = (rands < self.spins[:, None]).astype(jnp.uint8)
        return bitstreams

    @staticmethod
    @maybe_jit
    def _nths_kernel(
        spins: jnp.ndarray, field_input: jnp.ndarray, j_avg: float, h_bias: float, dt: float
    ) -> jnp.ndarray:
        """
        Solves the NTHS Spin-Glass dynamics:
        dSpin/dt = J * MeanField + h_bias + field_input - decay
        """
        mean_field = jnp.mean(spins)
        # H = -J * s_i * sum(s_j) -> mapped to probability drift
        d_spin = j_avg * mean_field + h_bias + field_input - 0.1 * spins
        return jnp.clip(spins + d_spin * dt, 0.0, 1.0)

    def step_jax(self, dt: float, inputs: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Advances the L11 holonomic dynamics using JAX.

        inputs: (n_nodes, bitstream_length) representing L7 Symbolic or L10 Firewall signals.
        Returns: (n_nodes, bitstream_length) output bitstreams.
        """
        # 1. Extract Informational Forcing (L7/L10 -> L11)
        if inputs is not None:
            info_drive = jnp.mean(inputs.astype(jnp.float32), axis=1)
            # Map input dimensions
            if info_drive.shape[0] != self.params.n_nodes:
                info_drive = jnp.full((self.params.n_nodes,), jnp.mean(info_drive))
        else:
            info_drive = jnp.zeros((self.params.n_nodes,))

        # 2. Execute NTHS Kernel
        self.spins = self._nths_kernel(
            self.spins, info_drive, self.params.j_coupling, self.params.h_bias, dt
        )

        # 3. Update Information Density (Proxy for memetic SIR)
        self.info_density = 0.9 * self.info_density + 0.1 * jnp.abs(self.spins - 0.5)

        # 4. Return encoded bitstreams
        return self.encode(None)

    def decode(self, bitstreams: jnp.ndarray) -> Dict[str, float]:
        """
        Maps bitstreams back to Noospheric Polarization index.
        """
        spins = jnp.mean(bitstreams.astype(jnp.float32), axis=1)
        polarization = jnp.std(spins)
        return {
            "noospheric_polarization": float(polarization),
            "collective_coherence_r11": float(jnp.mean(spins)),
        }

    def get_metrics(self) -> Dict[str, float]:
        """
        Returns L11-specific metrics like Polarization and Info Density.
        """
        return {
            "avg_polarization": float(jnp.std(self.spins)),
            "noospheric_entropy": float(-jnp.sum(self.spins * jnp.log(self.spins + 1e-6))),
            "info_saturation": float(jnp.mean(self.info_density)),
        }
