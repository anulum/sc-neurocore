"""
SCPN L10: Projective Field Boundary Control Adapter (JAX Implementation)
========================================================================

This module implements the JAX-accelerated uplift of Layer 10, focusing on
the Topological Firewall, Phase-Knotting insulation, and Dissonance-Triggered
Rejection described in Paper 10.

Key Equations:
- Boundary Insulation: Shielding ~ exp(-|nabla V|^2 / sigma)
- Dissonance Metric: D_topo = 1 - overlap(Psi_local, Psi_template)
- Terminal Steering: Intention mapped to terminal boundary conditions.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import jax
import jax.numpy as jnp
import numpy as np

from ..base import BaseStochasticAdapter
from ...accel.jax_backend import HAS_JAX, to_jax, to_host


@dataclass
class L10_HolonomicParameters:
    """Parameters derived from Paper 10 and Topological Insulation specs."""
    n_boundary_nodes: int = 100
    bitstream_length: int = 1024

    # Firewall Constants
    rejection_threshold: float = 0.4
    shielding_strength: float = 1.5
    steering_gain: float = 0.2


class L10_FirewallAdapter(BaseStochasticAdapter):
    """
    JAX-traceable adapter for the SCPN Topological Firewall layer.
    """

    def __init__(self, params: Optional[L10_HolonomicParameters] = None, seed: int = 410) -> None:
        self.params = params or L10_HolonomicParameters()

        self.rng_key = jax.random.PRNGKey(seed)
        
        # State: Firewall integrity (0 to 1)
        self.firewall_strength = jnp.full((self.params.n_boundary_nodes,), 0.9)
        # State: Local Intention potential
        self.intention_potential = jnp.zeros((self.params.n_boundary_nodes,))

    def encode(self, domain_state: Any) -> jnp.ndarray:
        """
        Maps firewall strength to stochastic bitstreams.
        """
        self.rng_key, subkey = jax.random.split(self.rng_key)
        rands = jax.random.uniform(subkey, (self.params.n_boundary_nodes, self.params.bitstream_length))
        bitstreams = (rands < self.firewall_strength[:, None]).astype(jnp.uint8)
        return bitstreams

    @staticmethod
    @jax.jit
    def _firewall_kernel(strength: jnp.ndarray, intention: jnp.ndarray, 
                        noise_inputs: jnp.ndarray, gain: float, dt: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Solves the Firewall / Topological dynamics:
        dStrength/dt = -D_topo * Strength + Intention_Steering
        """
        # Dissonance is high when noise inputs don't match intention
        dissonance = jnp.abs(noise_inputs - intention)
        
        # Strength decays with dissonance, grows with steering
        d_strength = -dissonance * strength + gain * intention - 0.01 * strength
        strength_next = jnp.clip(strength + d_strength * dt, 0.0, 1.0)
        
        return strength_next, dissonance

    def step_jax(self, dt: float, inputs: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Advances the L10 holonomic dynamics using JAX.
        
        inputs: (n_boundary_nodes, bitstream_length) representing external noise or L14 signals.
        Returns: (n_boundary_nodes, bitstream_length) output bitstreams (Shielding signals).
        """
        # 1. Extract External Pressure (Inputs -> L10)
        if inputs is not None:
            external_noise = jnp.mean(inputs.astype(jnp.float32), axis=1)
            if external_noise.shape[0] != self.params.n_boundary_nodes:
                external_noise = jnp.full((self.params.n_boundary_nodes,), jnp.mean(external_noise))
        else:
            external_noise = jnp.zeros((self.params.n_boundary_nodes,))

        # 2. Execute Firewall Kernel
        self.firewall_strength, dissonance = self._firewall_kernel(
            self.firewall_strength, self.intention_potential, 
            external_noise, self.params.steering_gain, dt
        )

        # 3. Return encoded bitstreams (Shielding status)
        return self.encode(None)

    def decode(self, bitstreams: jnp.ndarray) -> Dict[str, float]:
        """
        Maps bitstreams back to Firewall Integrity index.
        """
        return {
            "firewall_integrity_r10": float(jnp.mean(bitstreams.astype(jnp.float32)))
        }

    def get_metrics(self) -> Dict[str, float]:
        """
        Returns L10-specific metrics.
        """
        return {
            "avg_shielding_potential": float(jnp.mean(self.firewall_strength)),
            "topological_dissonance": float(jnp.std(self.firewall_strength))
        }
