# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L12 Ecological-Gaian Synchrony Adapter (JAX

"""
SCPN L12: Ecological-Gaian Synchrony Adapter (JAX Implementation)
=================================================================

This module implements the JAX-accelerated uplift of Layer 12, focusing on
the Mycorrhizal Quantum Network (MQN), Gaian Synchrony Operators, and
Environment-Assisted Quantum Transport (ENAQT) described in Paper 12.

Key Equations:
- MQN Hamiltonian: H = sum(E_i |i><i|) + sum(J_ij (|i><j| + h.c.))
- ENAQT Gain: dCoherence/dt = J_ij * noise_factor (Optimized transport)
- Gaian Sync: Theta_G = mean(Theta_local) phase-locked to solar/lunar cycles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..base import BaseStochasticAdapter
from ._jax_compat import jnp, make_rng, maybe_jit, split_rng, uniform


@dataclass
class L12_HolonomicParameters:
    """Parameters derived from Paper 12 and MQN specifications."""

    n_nodes: int = 100
    bitstream_length: int = 1024

    # ENAQT Constants
    j_coherent_coupling: float = 0.4
    noise_assistance_factor: float = 0.1

    # Ecological Sync
    gaian_decay: float = 0.05
    solar_lunar_omega: float = 0.01  # Frequency of environmental driving


class L12_GaianAdapter(BaseStochasticAdapter):
    """
    JAX-traceable adapter for the SCPN Ecological-Gaian layer.
    """

    def __init__(self, params: Optional[L12_HolonomicParameters] = None, seed: int = 412) -> None:
        self.params = params or L12_HolonomicParameters()
        self.rng_key = make_rng(seed)

        # State: Ecological Coherence (0 to 1)
        self.eco_coherence = jnp.full((self.params.n_nodes,), 0.2)
        # State: Nutrient/Information Flow density
        self.flow_density = jnp.zeros((self.params.n_nodes,))
        # State: Environmental Phase
        self.env_phase = 0.0

    def encode(self, domain_state: Any) -> jnp.ndarray:
        """
        Maps ecological coherence to stochastic bitstreams.
        """
        self.rng_key, subkey = split_rng(self.rng_key)
        rands = uniform(subkey, (self.params.n_nodes, self.params.bitstream_length))
        bitstreams: jnp.ndarray = (rands < self.eco_coherence[:, None]).astype(jnp.uint8)
        return bitstreams

    @staticmethod
    @maybe_jit
    def _enaqt_kernel(
        coherence: jnp.ndarray, flow: jnp.ndarray, j_coupling: float, noise_gain: float, dt: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Solves the ENAQT transport dynamics:
        dC/dt = J * noise * (1 - C) - decay
        """
        # Noise-assisted transport increases coherence
        d_coherence = j_coupling * noise_gain * (1.0 - coherence) - 0.05 * coherence
        coherence_next = jnp.clip(coherence + d_coherence * dt, 0.0, 1.0)

        # Flow density is proportional to coherence gradients
        new_flow = coherence_next * 0.5

        return coherence_next, new_flow

    def step_jax(self, dt: float, inputs: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Advances the L12 holonomic dynamics using JAX.

        inputs: (n_nodes, bitstream_length) representing L6 Planetary or L11 Noospheric drive.
        Returns: (n_nodes, bitstream_length) output bitstreams.
        """
        self.env_phase += self.params.solar_lunar_omega * dt

        # 1. Extract Environmental Forcing (L6/L11 -> L12)
        if inputs is not None:
            raw_input = jnp.mean(inputs.astype(jnp.float32), axis=1)
            # Map input dimensions
            if raw_input.shape[0] != self.params.n_nodes:
                env_drive = jnp.full((self.params.n_nodes,), jnp.mean(raw_input))
            else:
                env_drive = raw_input
        else:
            env_drive = jnp.zeros((self.params.n_nodes,))

        # 2. Execute ENAQT Kernel
        # Incorporate environmental drive into noise-assistance
        effective_noise = self.params.noise_assistance_factor * (1.0 + env_drive)
        self.eco_coherence, self.flow_density = self._enaqt_kernel(
            self.eco_coherence,
            self.flow_density,
            self.params.j_coherent_coupling,
            jnp.mean(effective_noise),
            dt,
        )

        # 3. Return encoded bitstreams
        return self.encode(None)

    def decode(self, bitstreams: jnp.ndarray) -> Dict[str, float]:
        """
        Maps bitstreams back to Gaian Synchrony Index.
        """
        return {
            "gaian_synchrony_index": float(jnp.mean(bitstreams.astype(jnp.float32))),
            "mycorrhizal_flow_rate": float(jnp.mean(self.flow_density)),
        }

    def get_metrics(self) -> Dict[str, float]:
        """
        Returns L12-specific metrics like Coherence and Flow.
        """
        return {
            "eco_system_coherence": float(jnp.mean(self.eco_coherence)),
            "global_nutrient_flow": float(jnp.mean(self.flow_density)),
            "environmental_alignment": float(jnp.sin(self.env_phase)),
        }
