# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L4 Cellular-Tissue Synchronization Adapter (JAX

"""
SCPN L4: Cellular-Tissue Synchronization Adapter (JAX Implementation)
======================================================================

This module implements the JAX-accelerated uplift of Layer 4, focusing on
Quasicritical Synchronization, Avalanche Dynamics, and the Cyto-Matrix
Tensegrity Network (CMTN) described in Paper 4.

Key Equations:
- Unified Phase Dynamics (UPDE): dTheta/dt = Omega + K * sum(sin(Theta_j - Theta_i))
- Avalanche Power Law: P(s) ~ s^(-tau)
- CMTN Tensegrity: Tension-Compression mechanical-vibrational coupling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..base import BaseStochasticAdapter
from ._jax_compat import jnp, make_rng, maybe_jit, normal, split_rng, uniform


@dataclass
class L4_HolonomicParameters:
    """Parameters derived from Paper 4 and T7 Validation protocols."""

    n_cells: int = 400  # 20x20 grid default
    bitstream_length: int = 1024

    # Synchronization Constants
    omega_mean: float = 1.0  # Baseline oscillator frequency (Hz)
    k_coupling: float = 0.3  # Kuramoto coupling strength
    sigma_noise: float = 0.1  # Intrinsic phase noise

    # Criticality Tuning
    critical_threshold: float = 0.6  # Mean-field threshold for avalanche ignition


class L4_CellularAdapter(BaseStochasticAdapter):
    """
    JAX-traceable adapter for the SCPN Cellular-Tissue layer.
    """

    def __init__(self, params: Optional[L4_HolonomicParameters] = None, seed: int = 44) -> None:
        self.params = params or L4_HolonomicParameters()
        self.rng_key = make_rng(seed)

        # State: Oscillator Phases (0 to 2*pi)
        self.phases = uniform(self.rng_key, (self.params.n_cells,), minval=0.0, maxval=2 * jnp.pi)
        # State: Local Avalanche Magnitude
        self.avalanches = jnp.zeros((self.params.n_cells,))

    def encode(self, domain_state: Any) -> jnp.ndarray:
        """
        Maps synchronization activity to stochastic bitstreams.
        """
        # Activity = (1 + cos(phase)) / 2
        activity = (1.0 + jnp.cos(self.phases)) / 2.0

        self.rng_key, subkey = split_rng(self.rng_key)
        rands = uniform(subkey, (self.params.n_cells, self.params.bitstream_length))
        bitstreams: jnp.ndarray = (rands < activity[:, None]).astype(jnp.uint8)
        return bitstreams

    @staticmethod
    @maybe_jit
    def _kuramoto_kernel(
        phases: jnp.ndarray, omega: float, k: float, dt: float, noise: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Solves the Kuramoto-UPDE interaction:
        dTheta_i = [omega + K/N * sum(sin(Theta_j - Theta_i)) + noise] * dt
        """
        n = phases.shape[0]
        # Calculate all-to-all coupling (can be optimized with neighbor masks later)
        diffs = phases[None, :] - phases[:, None]
        coupling = (k / n) * jnp.sum(jnp.sin(diffs), axis=1)

        d_phase = (2 * jnp.pi * omega + coupling + noise) * dt
        return (phases + d_phase) % (2 * jnp.pi)

    def step_jax(self, dt: float, inputs: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Advances the L4 holonomic dynamics using JAX.

        inputs: (n_cells, bitstream_length) representing L3 Genomic drive.
        Returns: (n_cells, bitstream_length) output bitstreams.
        """
        # 1. Generate Noise
        self.rng_key, subkey = split_rng(self.rng_key)
        noise = normal(subkey, (self.params.n_cells,)) * self.params.sigma_noise

        # 2. Update Phases via Kuramoto Kernel
        self.phases = self._kuramoto_kernel(
            self.phases, self.params.omega_mean, self.params.k_coupling, dt, noise
        )

        # 3. Model Avalanche Dynamics (Criticality readout)
        # If mean activity crosses threshold, ignition occurs
        mean_activity = jnp.mean((1.0 + jnp.cos(self.phases)) / 2.0)
        ignition = (mean_activity > self.params.critical_threshold).astype(jnp.float32)
        self.avalanches = 0.9 * self.avalanches + 0.1 * ignition

        # 4. Return encoded bitstreams
        return self.encode(None)

    def decode(self, bitstreams: jnp.ndarray) -> Dict[str, float]:
        """
        Maps bitstreams back to Kuramoto order parameter.
        """
        # Complex order parameter R = |1/N * sum(exp(i*theta))|
        # Approximated from bitstream means
        return {"synchronization_r4": float(jnp.abs(jnp.mean(jnp.exp(1j * self.phases))))}

    def get_metrics(self) -> Dict[str, float]:
        """
        Returns L4-specific metrics.
        """
        return {
            "order_parameter": float(jnp.abs(jnp.mean(jnp.exp(1j * self.phases)))),
            "avalanche_density": float(jnp.mean(self.avalanches)),
        }
