# SPDX-License-Identifier: AGPL-3.0-or-later
"""
SCPN L13: Source-Field / Meta-Universal Adapter (JAX Implementation)
====================================================================

This module implements the JAX-accelerated uplift of Layer 13, focusing on
the Constructor-Theoretic Causal Closure, Vacuum Lattice Dynamics, and the
primordial 'Scission' described in Paper 13.

Key Equations:
- Vacuum Lattice Hamiltonian: H = sum(J * sigma_i * sigma_j) + h * sum(sigma_i)
- Universal Metric: ds^2 = g_FIM (interaction distance)
- Causal Closure: Possible vs. Impossible transformations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

try:
    import jax
    import jax.numpy as jnp

    HAS_JAX = True
except ImportError:
    jax = None  # type: ignore[assignment]
    import numpy as jnp  # type: ignore[no-redef]

    HAS_JAX = False

from ..base import BaseStochasticAdapter


@dataclass
class L13_HolonomicParameters:
    """Parameters derived from Paper 13 and Vacuum Lattice specs."""

    n_vacuum_nodes: int = 256
    bitstream_length: int = 1024

    # Ontological Constants
    j_primordial_coupling: float = 1.0
    h_potential_bias: float = 0.01
    lambda_scission: float = 0.1  # Rate of symmetry breaking


class L13_SourceAdapter(BaseStochasticAdapter):
    """
    JAX-traceable adapter for the SCPN Source-Field layer.
    """

    def __init__(self, params: Optional[L13_HolonomicParameters] = None, seed: int = 413) -> None:
        self.params = params or L13_HolonomicParameters()
        self.rng_key = jax.random.PRNGKey(seed)

        # State: Vacuum Potential (0.0 to 1.0)
        self.vacuum_state = jnp.full((self.params.n_vacuum_nodes,), 0.5)
        # State: Fisher Information Metric Density
        self.fim_density = jnp.zeros((self.params.n_vacuum_nodes,))

    def encode(self, domain_state: Any) -> jnp.ndarray:
        """
        Maps vacuum potential to stochastic bitstreams.
        """
        self.rng_key, subkey = jax.random.split(self.rng_key)
        rands = jax.random.uniform(
            subkey, (self.params.n_vacuum_nodes, self.params.bitstream_length)
        )
        bitstreams = (rands < self.vacuum_state[:, None]).astype(jnp.uint8)
        return bitstreams

    @staticmethod
    @jax.jit
    def _vacuum_kernel(state: jnp.ndarray, coupling: float, bias: float, dt: float) -> jnp.ndarray:
        """
        Solves the Vacuum Lattice dynamics (Mean-field approximation):
        dPsi/dt = J * mean(Psi) + h - decay
        """
        mean_pot = jnp.mean(state)
        # Primordial drive toward potentialization
        d_state = coupling * mean_pot + bias - 0.05 * state
        return jnp.clip(state + d_state * dt, 0.0, 1.0)

    def step_jax(self, dt: float, inputs: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Advances the L13 holonomic dynamics using JAX.

        inputs: Optional feedback from L16 (Cybernetic Closure).
        Returns: (n_vacuum_nodes, bitstream_length) output bitstreams.
        """
        # 1. Update Vacuum State
        self.vacuum_state = self._vacuum_kernel(
            self.vacuum_state, self.params.j_primordial_coupling, self.params.h_potential_bias, dt
        )

        # 2. Update FIM Density (Measures rate of change / information work)
        # delta_Psi ~ rate of information creation
        self.fim_density = 0.9 * self.fim_density + 0.1 * jnp.abs(self.vacuum_state - 0.5)

        # 3. Return encoded bitstreams (The primordial carrier)
        return self.encode(None)

    def decode(self, bitstreams: jnp.ndarray) -> Dict[str, float]:
        """
        Maps bitstreams back to Primordial Coherence.
        """
        return {"source_coherence_r13": float(jnp.mean(bitstreams.astype(jnp.float32)))}

    def get_metrics(self) -> Dict[str, float]:
        """
        Returns L13-specific metrics.
        """
        return {
            "vacuum_potential": float(jnp.mean(self.vacuum_state)),
            "fisher_information_metric": float(jnp.mean(self.fim_density)),
        }
