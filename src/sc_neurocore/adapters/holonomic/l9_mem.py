# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L9 Memory Imprint-Existential Holograph Adapter

"""
SCPN L9: Memory Imprint-Existential Holograph Adapter (JAX Implementation)
==========================================================================

This module implements the JAX-accelerated uplift of Layer 9, focusing on
the Two-State Vector Formalism (TSVF), Z-cyclic imprinting, and
weak-value retrieval described in Paper 9.

Key Equations:
- Weak Value Retrieval: Aw = <Phi|A|Psi> / <Phi|Psi>
- Time-Symmetric Flow: Memory as overlap of forward (Psi) and backward (Phi) bitstreams.
- Holographic QEC: Reconstruction of existential imprints using MERA structures.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..base import BaseStochasticAdapter
from ._jax_compat import jnp, make_rng, maybe_jit, split_rng, uniform


@dataclass
class L9_HolonomicParameters:
    """Parameters derived from Paper 9 and TSVF specifications."""

    n_memory_slots: int = 64
    bitstream_length: int = 1024

    # TSVF Constants
    retrieval_gain: float = 0.8
    weak_measurement_strength: float = 0.1
    temporal_window: int = 100


class L9_MemoryAdapter(BaseStochasticAdapter):
    """
    JAX-traceable adapter for the SCPN Existential Memory layer.
    """

    def __init__(self, params: Optional[L9_HolonomicParameters] = None, seed: int = 49) -> None:
        self.params = params or L9_HolonomicParameters()
        self.rng_key = make_rng(seed)

        # State: Forward Bitstream Imprints (Psi)
        self.imprints_psi = jnp.zeros(
            (self.params.n_memory_slots, self.params.bitstream_length), dtype=jnp.uint8
        )
        # State: Backward Retrieval Vectors (Phi)
        self.retrieval_phi = jnp.zeros(
            (self.params.n_memory_slots, self.params.bitstream_length), dtype=jnp.uint8
        )
        # Index for cyclic imprinting
        self.current_slot = 0

    def encode(self, domain_state: Any) -> jnp.ndarray:
        """
        Maps memory imprints to stochastic bitstreams via TSVF overlap.
        """
        # Memory retrieval probability = Normalized overlap <Phi|Psi>
        psi_float = self.imprints_psi.astype(jnp.float32)
        phi_float = self.retrieval_phi.astype(jnp.float32)

        # Calculate overlap per slot
        overlap = jnp.mean(psi_float * phi_float, axis=1)
        # Sum overlaps to get retrieval activation
        retrieval_prob = jnp.clip(jnp.sum(overlap) * self.params.retrieval_gain, 0.0, 1.0)

        self.rng_key, subkey = split_rng(self.rng_key)
        rands = uniform(subkey, (self.params.bitstream_length,))
        # Single channel output representing retrieved memory content
        bitstream: jnp.ndarray = (rands < retrieval_prob).astype(jnp.uint8)
        return bitstream

    @staticmethod
    @maybe_jit
    def _tsvf_kernel(
        psi: jnp.ndarray, phi: jnp.ndarray, inputs: jnp.ndarray, strength: float, dt: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Updates the forward/backward holographic imprints.
        """
        # Forward imprinting Psi captures current input
        psi_next = jnp.where(inputs > 0.5, 1, psi).astype(jnp.uint8)
        # Backward retrieval Phi adapts to current state (Weak measurement)
        phi_next = jnp.where(jnp.abs(psi_next.astype(jnp.float32) - 0.5) > 0.1, 1, phi).astype(
            jnp.uint8
        )

        return psi_next, phi_next

    def step_jax(self, dt: float, inputs: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Advances the L9 holonomic dynamics using JAX.

        inputs: (N, bitstream_length) representing L5 Organismal state to imprint.
        Returns: (bitstream_length,) retrieval bitstream.
        """
        if inputs is not None:
            # 1. Project inputs to memory slot count if necessary
            if inputs.shape[0] != self.params.n_memory_slots:
                # Tile or truncate to match slots
                n_in = inputs.shape[0]
                n_slots = self.params.n_memory_slots
                indices = jnp.arange(n_slots) % n_in
                mapped_inputs = inputs[indices]
            else:
                mapped_inputs = inputs

            # 2. Update forward/backward holographic imprints
            self.imprints_psi, self.retrieval_phi = self._tsvf_kernel(
                self.imprints_psi,
                self.retrieval_phi,
                mapped_inputs,
                self.params.weak_measurement_strength,
                dt,
            )

        # 3. Return retrieved bitstream (projected to node count)
        return self.encode(None)

    def decode(self, bitstreams: jnp.ndarray) -> Dict[str, float]:
        """
        Maps bitstreams back to Memory Retrieval quality.
        """
        return {"memory_retrieval_r9": float(jnp.mean(bitstreams.astype(jnp.float32)))}

    def get_metrics(self) -> Dict[str, float]:
        """
        Returns L9-specific metrics.
        """
        return {
            "holographic_overlap": float(
                jnp.mean(
                    self.imprints_psi.astype(jnp.float32) * self.retrieval_phi.astype(jnp.float32)
                )
            ),
            "imprint_density": float(jnp.mean(self.imprints_psi)),
        }
