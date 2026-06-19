# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L8 Cosmic Phase-Locking Adapter (JAX Implementation)

"""
SCPN L8: Cosmic Phase-Locking Adapter (JAX Implementation)
==========================================================

This module implements the JAX-accelerated uplift of Layer 8, focusing on
Pulsar Timing Array (PTA) synchronization and Cosmic Phase-Locking
dynamics described in Paper 8.

Key Equations:
- Cosmic Sync: dTheta/dt = Omega_local + K_cosmic * sum(I_p * sin(Theta_p - Theta_local))
- PTA Clock: Omega_p = constant signatures from millisecond pulsars.
- Orthogenesis Drive: Evolution as a biased random walk on cosmic fitness landscapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from ..base import BaseStochasticAdapter
from ._jax_compat import jnp, make_rng, maybe_jit, split_rng, uniform


@dataclass
class L8_HolonomicParameters:
    """Parameters derived from Paper 8 and PTA specifications."""

    n_pulsars: int = 12  # Number of simulated pulsar clock references
    bitstream_length: int = 1024

    # Cosmic Constants
    k_cosmic: float = 0.05  # Global cosmic coupling strength
    pta_stability: float = 1e-15  # Target clock stability proxy

    # Fundamental Frequencies (Simulated PTA signatures)
    pulsar_omegas: jnp.ndarray = field(
        default_factory=lambda: jnp.array(
            [1.6, 2.3, 0.8, 4.1, 1.1, 0.5, 3.2, 2.7, 1.9, 0.4, 5.5, 0.2]
        )
    )


class L8_CosmicAdapter(BaseStochasticAdapter):
    """
    JAX-traceable adapter for the SCPN Cosmic Phase-Locking layer.
    """

    def __init__(self, params: Optional[L8_HolonomicParameters] = None, seed: int = 48) -> None:
        self.params = params or L8_HolonomicParameters()
        self.rng_key = make_rng(seed)

        # State: Local System Phases (locked to pulsars)
        self.system_phases = jnp.zeros((self.params.n_pulsars,))
        # State: Cosmic Clock time
        self.t_cosmic = 0.0

    def encode(self, domain_state: Any) -> jnp.ndarray:
        """
        Maps cosmic phases to stochastic bitstreams.
        """
        activation = (1.0 + jnp.cos(self.system_phases)) / 2.0

        self.rng_key, subkey = split_rng(self.rng_key)
        rands = uniform(subkey, (self.params.n_pulsars, self.params.bitstream_length))
        bitstreams: jnp.ndarray = (rands < activation[:, None]).astype(jnp.uint8)
        return bitstreams

    @staticmethod
    @maybe_jit
    def _cosmic_kernel(
        phases: jnp.ndarray, pulsar_omegas: jnp.ndarray, k_cosmic: float, dt: float
    ) -> jnp.ndarray:
        """
        Solves the Cosmic Phase-Locking dynamics:
        dTheta = [Omega_p + K * sin(Theta_p - Theta)] * dt
        """
        # Theta_pulsar is simulated as Omega_p * t
        # For simplicity in the JIT kernel, we assume pulsar phases are pre-calculated
        # or we just drive the local oscillators by their omegas with a coupling term.
        d_phase = pulsar_omegas + k_cosmic * jnp.sin(-phases)
        return (phases + d_phase * dt) % (2 * jnp.pi)

    def step_jax(self, dt: float, inputs: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Advances the L8 holonomic dynamics using JAX.

        inputs: (n_pulsars, bitstream_length) representing L7 Symbolic feedback.
        Returns: (n_pulsars, bitstream_length) output bitstreams.
        """
        self.t_cosmic += dt

        # 1. Update system phases via Cosmic Kernel
        self.system_phases = self._cosmic_kernel(
            self.system_phases, self.params.pulsar_omegas, self.params.k_cosmic, dt
        )

        # 2. Apply feedback from L7 (Symbolic) if present
        if inputs is not None:
            symbolic_drive = jnp.mean(inputs.astype(jnp.float32), axis=1)
            # Map input dimensions
            if symbolic_drive.shape[0] != self.params.n_pulsars:
                symbolic_drive = jnp.full((self.params.n_pulsars,), jnp.mean(symbolic_drive))
            self.system_phases = (self.system_phases + 0.1 * symbolic_drive * dt) % (2 * jnp.pi)

        # 3. Return encoded bitstreams
        return self.encode(None)

    def decode(self, bitstreams: jnp.ndarray) -> Dict[str, float]:
        """
        Maps bitstreams back to Cosmic Alignment.
        """
        return {"cosmic_alignment_r8": float(jnp.abs(jnp.mean(jnp.exp(1j * self.system_phases))))}

    def get_metrics(self) -> Dict[str, float]:
        """
        Returns L8-specific metrics.
        """
        return {
            "clock_stability": float(jnp.std(self.system_phases)),
            "pta_locking_index": float(jnp.abs(jnp.mean(jnp.exp(1j * self.system_phases)))),
        }
