# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L5 Organismal-Psychoemotional Adapter (JAX

"""
SCPN L5: Organismal-Psychoemotional Adapter (JAX Implementation)
================================================================

This module implements the JAX-accelerated uplift of Layer 5, focusing on
the Integrated Self (Strange Loop), Autonomic Regulation (HRV), and
Emotional Attractor dynamics described in Paper 5.

Key Equations:
- Strange Loop Feedback: I = Model(I) -> mapped to recursive bitstream state.
- Affective Field: A = -nabla F (Emotion as free energy gradient).
- HRV Coherence: Resonance between sympathetic and parasympathetic nodes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..base import BaseStochasticAdapter
from ._jax_compat import jnp, make_rng, maybe_jit, split_rng, uniform


@dataclass
class L5_HolonomicParameters:
    """Parameters derived from Paper 5 and FEP (Free Energy Principle)."""

    n_nodes: int = 100
    n_emotional_dims: int = 8
    bitstream_length: int = 1024

    # Autonomic Constants
    tau_autonomic: float = 5.0  # Seconds
    hrv_resonance: float = 0.25  # Hz (Respiratory Sinus Arrhythmia)

    # Emotional Constants
    emotional_decay: float = 0.1
    attractor_strength: float = 0.3


class L5_OrganismalAdapter(BaseStochasticAdapter):
    """
    JAX-traceable adapter for the SCPN Organismal-Psychoemotional layer.
    """

    def __init__(self, params: Optional[L5_HolonomicParameters] = None, seed: int = 45) -> None:
        self.params = params or L5_HolonomicParameters()
        self.rng_key = make_rng(seed)

        # State: Emotional Valence Vector (n_dims)
        self.emotions = jnp.full((self.params.n_emotional_dims,), 0.5)
        # State: Autonomic Tone (Sympathetic, Parasympathetic)
        self.autonomic = jnp.array([0.4, 0.6])  # [Symp, Para]
        # State: Strange Loop Recursive Model (Self-Soliton)
        self.self_soliton = jnp.zeros((self.params.n_nodes,))

    def encode(self, domain_state: Any) -> jnp.ndarray:
        """
        Maps organismal state to stochastic bitstreams.
        """
        # Composite probability from emotions and autonomic tone
        avg_tone = jnp.mean(self.autonomic)
        probs = jnp.concatenate([self.emotions, self.autonomic])
        # Project to node count
        node_probs = jnp.tile(probs, (self.params.n_nodes // probs.shape[0]) + 1)[
            : self.params.n_nodes
        ]

        self.rng_key, subkey = split_rng(self.rng_key)
        rands = uniform(subkey, (self.params.n_nodes, self.params.bitstream_length))
        bitstreams: jnp.ndarray = (rands < node_probs[:, None]).astype(jnp.uint8)
        return bitstreams

    @staticmethod
    @maybe_jit
    def _autonomic_kernel(
        current: jnp.ndarray, target: jnp.ndarray, tau: float, dt: float
    ) -> jnp.ndarray:
        """
        Euler-integration of autonomic homeostasis.
        """
        return current + (target - current) * (dt / tau)

    def step_jax(self, dt: float, inputs: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Advances the L5 holonomic dynamics using JAX.

        inputs: (n_nodes, bitstream_length) representing L4 synchronization.
        Returns: (n_nodes, bitstream_length) output bitstreams.
        """
        # 1. Update Autonomic Tone based on L4 Synchronization
        if inputs is not None:
            sync = jnp.abs(jnp.mean(jnp.exp(1j * jnp.mean(inputs.astype(jnp.float32), axis=1))))
            # Higher sync drives Parasympathetic tone
            target_para = 0.5 + 0.4 * sync
            target_symp = 1.0 - target_para
            target = jnp.array([target_symp, target_para])
            self.autonomic = self._autonomic_kernel(
                self.autonomic, target, self.params.tau_autonomic, dt
            )

        # 2. Emotional Attractor Dynamics (Simplified)
        # Decay toward neutral [0.5]
        self.emotions = self.emotions + (0.5 - self.emotions) * self.params.emotional_decay * dt

        # 3. Recursive Strange Loop Update (The Self-Soliton)
        # self_soliton = f(self_soliton, emotions)
        self.self_soliton = 0.95 * self.self_soliton + 0.05 * jnp.mean(self.emotions)

        # 4. Return encoded bitstreams
        return self.encode(None)

    def decode(self, bitstreams: jnp.ndarray) -> Dict[str, float]:
        """
        Maps bitstreams back to average valence and HRV coherence.
        """
        return {
            "organismal_valence": float(jnp.mean(self.emotions)),
            "autonomic_balance": float(self.autonomic[1] / (self.autonomic[0] + 1e-6)),
        }

    def get_metrics(self) -> Dict[str, float]:
        """
        Returns L5-specific metrics.
        """
        return {
            "hrv_coherence_r5": float(self.autonomic[1]),
            "self_soliton_magnitude": float(jnp.mean(self.self_soliton)),
            "emotional_valence": float(self.emotions[0]),
        }
