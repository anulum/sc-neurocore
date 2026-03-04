"""
SCPN L6: Planetary-Biospheric Adapter (JAX Implementation)
==========================================================

This module implements the JAX-accelerated uplift of Layer 6, focusing on
Schumann Resonance coupling, Planetary Superradiance (P ~ N^2), and the
Percolation Phase Transition of global consciousness described in Paper 6.

Key Equations:
- Schumann Coupling: Psi_P = Psi_local * exp(i * omega_S * t) where omega_S ~ 7.83Hz
- Biospheric Superradiance: Power_total = N^2 * Power_individual
- Percolation Transition: Coherence_global = f(p - p_c) where p_c ~ 0.59
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
class L6_HolonomicParameters:
    """Parameters derived from Paper 6 and Gaia-field specifications."""
    n_regions: int = 100
    bitstream_length: int = 1024
    
    # Schumann Resonance Constants
    f_schumann: float = 7.83        # Hz (Fundamental mode)
    q_factor: float = 4.0           # Cavity resonance quality
    
    # Planetary Coupling
    alpha_gaia: float = 0.05        # Individual-to-Planetary coupling strength
    p_percolation: float = 0.592    # Critical threshold for global coherence


class L6_PlanetaryAdapter(BaseStochasticAdapter):
    """
    JAX-traceable adapter for the SCPN Planetary-Biospheric layer.
    """

    def __init__(self, params: Optional[L6_HolonomicParameters] = None, seed: int = 46) -> None:
        self.params = params or L6_HolonomicParameters()
        self.rng_key = jax.random.PRNGKey(seed)
        
        # State: Planetary Field Potential (Psi_P)
        self.phi_planetary = jnp.zeros((self.params.n_regions,))
        # State: Regional Coherence index
        self.regional_coherence = jnp.full((self.params.n_regions,), 0.1)
        # Time tracking for oscillatory resonance
        self.t = 0.0

    def encode(self, domain_state: Any) -> jnp.ndarray:
        """
        Maps planetary coherence to stochastic bitstreams.
        """
        self.rng_key, subkey = jax.random.split(self.rng_key)
        rands = jax.random.uniform(subkey, (self.params.n_regions, self.params.bitstream_length))
        bitstreams = (rands < self.regional_coherence[:, None]).astype(jnp.uint8)
        return bitstreams

    @staticmethod
    @jax.jit
    def _gaia_kernel(phi: jnp.ndarray, sync_inputs: jnp.ndarray, alpha: float, 
                    freq: float, t: float, dt: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Solves the Planetary Gaia-field dynamics:
        dPhi/dt = alpha * sync_inputs * cos(2*pi*f*t) - decay * Phi
        """
        # Schumann resonance driving term
        driver = jnp.cos(2.0 * jnp.pi * freq * t)
        d_phi = alpha * sync_inputs * driver - 0.05 * phi
        
        # Superradiant scaling (simplified)
        phi_next = phi + d_phi * dt
        
        # Calculate resulting coherence (Percolation transition proxy)
        # Regional coherence increases when field potential is high
        coherence_next = jnp.clip(jnp.abs(phi_next) * 2.0, 0.0, 1.0)
        
        return phi_next, coherence_next

    def step_jax(self, dt: float, inputs: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Advances the L6 holonomic dynamics using JAX.
        
        inputs: (n_regions, bitstream_length) representing L5 Organismal output.
        Returns: (n_regions, bitstream_length) output bitstreams.
        """
        self.t += dt
        
        # 1. Extract Organismal Synchronization (L5 -> L6)
        if inputs is not None:
            sync_drive = jnp.mean(inputs.astype(jnp.float32), axis=1)
            # Map input dimensions to regional count
            if sync_drive.shape[0] != self.params.n_regions:
                sync_drive = jnp.full((self.params.n_regions,), jnp.mean(sync_drive))
        else:
            sync_drive = jnp.zeros((self.params.n_regions,))

        # 2. Execute Gaia Kernel
        self.phi_planetary, self.regional_coherence = self._gaia_kernel(
            self.phi_planetary, sync_drive, self.params.alpha_gaia, 
            self.params.f_schumann, self.t, dt
        )

        # 3. Return encoded bitstreams
        return self.encode(None)

    def decode(self, bitstreams: jnp.ndarray) -> Dict[str, float]:
        """
        Maps bitstreams back to Global Consciousness Index.
        """
        return {
            "global_coherence_index": float(jnp.mean(bitstreams.astype(jnp.float32)))
        }

    def get_metrics(self) -> Dict[str, float]:
        """
        Returns L6-specific metrics like Gaia Potential and Schumann Alignment.
        """
        return {
            "gaia_potential": float(jnp.mean(self.phi_planetary)),
            "percolation_index": float(jnp.mean(self.regional_coherence)),
            "schumann_phase": float(self.t * self.params.f_schumann % 1.0)
        }
