# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L9 Memory Imprint-Existential Holograph Adapter

"""SCPN L9 memory-imprint holograph adapter.

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

import numpy as np

from ..base import BaseStochasticAdapter
from ._jax_compat import jnp, make_rng, maybe_jit, split_rng, uniform


@dataclass
class L9_HolonomicParameters:
    """Configuration for the Layer 9 TSVF memory adapter.

    Parameters
    ----------
    n_memory_slots:
        Number of holographic memory rows retained by the adapter.
    bitstream_length:
        Number of stochastic bits carried by each memory row.
    retrieval_gain:
        Non-negative multiplier applied to the total TSVF overlap.
    weak_measurement_strength:
        Bounded measurement coupling reserved for the TSVF update kernel.
    temporal_window:
        Positive temporal-memory window used by downstream Layer 9 consumers.
    """

    n_memory_slots: int = 64
    bitstream_length: int = 1024

    # TSVF Constants
    retrieval_gain: float = 0.8
    weak_measurement_strength: float = 0.1
    temporal_window: int = 100


class L9_MemoryAdapter(BaseStochasticAdapter):
    """JAX-traceable adapter for the SCPN existential-memory layer."""

    def __init__(self, params: Optional[L9_HolonomicParameters] = None, seed: int = 49) -> None:
        """Initialise the Layer 9 memory adapter.

        Parameters
        ----------
        params:
            Optional Layer 9 configuration. Defaults preserve the historical
            64-slot, 1024-bitstream contract.
        seed:
            Random seed forwarded to the JAX or NumPy compatibility RNG.

        Raises
        ------
        ValueError
            If any parameter would create an invalid memory tensor or unsafe
            retrieval contract.
        """
        self.params = params or L9_HolonomicParameters()
        self._validate_params(self.params)
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

    @staticmethod
    def _validate_positive_int(name: str, value: int) -> None:
        """Validate a strict positive integer configuration field."""
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    @classmethod
    def _validate_params(cls, params: L9_HolonomicParameters) -> None:
        """Validate Layer 9 parameters before allocating backend arrays."""
        cls._validate_positive_int("n_memory_slots", params.n_memory_slots)
        cls._validate_positive_int("bitstream_length", params.bitstream_length)
        cls._validate_positive_int("temporal_window", params.temporal_window)
        if not np.isfinite(params.retrieval_gain) or params.retrieval_gain < 0.0:
            raise ValueError("retrieval_gain must be finite and non-negative.")
        if not np.isfinite(params.weak_measurement_strength) or not (
            0.0 <= params.weak_measurement_strength <= 1.0
        ):
            raise ValueError("weak_measurement_strength must be finite and in [0, 1].")

    @staticmethod
    def _validate_dt(dt: float) -> None:
        """Validate a positive finite simulation timestep."""
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive.")

    def _validate_input_batch(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Validate and normalise an upstream L9 bitstream batch."""
        input_batch: jnp.ndarray = jnp.asarray(inputs)
        if input_batch.ndim != 2:
            raise ValueError("inputs must be a rank-2 bitstream batch.")
        if input_batch.shape[0] <= 0:
            raise ValueError("inputs must contain at least one row.")
        if input_batch.shape[1] != self.params.bitstream_length:
            raise ValueError("inputs bitstream_length must match adapter parameters.")
        if not bool(np.all(np.isfinite(np.asarray(input_batch)))):
            raise ValueError("inputs must contain only finite values.")
        return input_batch

    def encode(self, domain_state: Any) -> jnp.ndarray:
        """Map memory imprints to stochastic bitstreams via TSVF overlap.

        Parameters
        ----------
        domain_state:
            Reserved adapter payload for interface compatibility. Layer 9 uses
            internal TSVF state for encoding.

        Returns
        -------
        jnp.ndarray
            One-dimensional retrieved-memory bitstream with
            ``bitstream_length`` entries.
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
        """Update the forward and backward holographic imprints.

        Parameters
        ----------
        psi:
            Forward memory-imprint tensor.
        phi:
            Backward retrieval-vector tensor.
        inputs:
            Validated rank-2 upstream bitstream tensor.
        strength:
            Weak-measurement coupling reserved for the traced TSVF update.
        dt:
            Positive simulation timestep.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Updated ``psi`` and ``phi`` tensors.
        """
        # Forward imprinting Psi captures current input
        psi_next = jnp.where(inputs > 0.5, 1, psi).astype(jnp.uint8)
        # Backward retrieval Phi adapts to current state (Weak measurement)
        phi_next = jnp.where(jnp.abs(psi_next.astype(jnp.float32) - 0.5) > strength, 1, phi).astype(
            jnp.uint8
        )

        return psi_next, phi_next

    def step_jax(self, dt: float, inputs: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Advance the L9 holonomic dynamics using JAX-compatible arrays.

        Parameters
        ----------
        dt:
            Positive finite simulation timestep.
        inputs:
            Optional ``(N, bitstream_length)`` upstream state to imprint. If
            ``N`` differs from ``n_memory_slots``, rows are tiled
            deterministically until all memory slots receive a row.

        Returns
        -------
        jnp.ndarray
            Retrieved-memory bitstream with shape ``(bitstream_length,)``.

        Raises
        ------
        ValueError
            If ``dt`` or ``inputs`` violates the bounded adapter contract.
        """
        self._validate_dt(dt)
        if inputs is not None:
            input_batch = self._validate_input_batch(inputs)
            # 1. Project inputs to memory slot count if necessary
            if input_batch.shape[0] != self.params.n_memory_slots:
                # Tile or truncate to match slots
                n_in = input_batch.shape[0]
                n_slots = self.params.n_memory_slots
                indices = jnp.arange(n_slots) % n_in
                mapped_inputs = input_batch[indices]
            else:
                mapped_inputs = input_batch

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
        """Map bitstreams back to memory-retrieval quality.

        Parameters
        ----------
        bitstreams:
            Retrieved stochastic memory bitstream.

        Returns
        -------
        dict[str, float]
            Telemetry dictionary containing ``memory_retrieval_r9``.
        """
        return {"memory_retrieval_r9": float(jnp.mean(bitstreams.astype(jnp.float32)))}

    def get_metrics(self) -> Dict[str, float]:
        """Return L9-specific overlap and imprint-density metrics.

        Returns
        -------
        dict[str, float]
            Current holographic-overlap and imprint-density telemetry.
        """
        return {
            "holographic_overlap": float(
                jnp.mean(
                    self.imprints_psi.astype(jnp.float32) * self.retrieval_phi.astype(jnp.float32)
                )
            ),
            "imprint_density": float(jnp.mean(self.imprints_psi)),
        }
