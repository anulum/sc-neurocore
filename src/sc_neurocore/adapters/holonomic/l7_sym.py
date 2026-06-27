# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L7 Geometrical-Symbolic Adapter (JAX Implementation)

"""SCPN L7 geometrical-symbolic adapter.

This module implements the JAX-accelerated uplift of Layer 7, focusing on
Metatron's Cube as a routing matrix and Symbolic Operators that phase-shift
stochastic bitstreams as described in Paper 7.

Key Equations:
- Symbolic Actuation: P = sum(G_k * exp(i * theta_k))
- Functorial Mapping: S: Geom -> Psi Field
- Routing: Out_m = sum(M_mn * In_n) where M is Metatron's Matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..base import BaseStochasticAdapter
from ._jax_compat import jnp, make_rng, maybe_jit, split_rng, uniform


@dataclass
class L7_HolonomicParameters:
    """Parameters derived from Paper 7 and Metatron's Cube geometry."""

    n_nodes: int = 13  # Standard Metatron's Cube node count
    bitstream_length: int = 1024

    # Symbolic Constants
    g_geometric_gain: float = 1.2
    phi_golden_ratio: float = 1.61803398875

    # Routing Constants
    coupling_leak: float = 0.05


class L7_SymbolicAdapter(BaseStochasticAdapter):
    """JAX-traceable adapter for the SCPN geometrical-symbolic layer."""

    def __init__(self, params: Optional[L7_HolonomicParameters] = None, seed: int = 47) -> None:
        self.params = params or L7_HolonomicParameters()
        self._validate_params(self.params)
        self.rng_key = make_rng(seed)

        # State: Node Phases (representing symbolic glyphs)
        self.node_phases = jnp.zeros((self.params.n_nodes,))
        # State: Metatron's Cube Adjacency Matrix (13x13)
        self.metatron_matrix = self._init_metatron_matrix()

    def _init_metatron_matrix(self) -> jnp.ndarray:
        """Initialise a symmetric bounded Metatron routing topology."""
        n = self.params.n_nodes
        if n == 1:
            return jnp.array([[1.0]], dtype=jnp.float32)

        coords = self._metatron_coordinates(n)
        deltas = coords[:, None, :] - coords[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)

        with np.errstate(over="ignore", under="ignore"):
            off_diag = np.exp(-distances / self.params.phi_golden_ratio)
        off_diag *= self.params.g_geometric_gain
        np.fill_diagonal(off_diag, 0.0)

        row_sums = off_diag.sum(axis=1)
        max_row_sum = float(np.max(row_sums))
        if max_row_sum <= 0.0:
            raise ValueError("Metatron topology requires at least one off-diagonal edge.")

        off_diag *= (1.0 - self.params.coupling_leak) / max_row_sum
        matrix = off_diag
        np.fill_diagonal(matrix, 1.0 - matrix.sum(axis=1))
        return jnp.array(matrix, dtype=jnp.float32)

    @staticmethod
    def _metatron_coordinates(n_nodes: int) -> np.ndarray[Any, Any]:
        if n_nodes == 13:
            angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
            inner = np.stack([np.cos(angles), np.sin(angles)], axis=1)
            outer = 2.0 * inner
            return np.vstack([np.zeros((1, 2)), inner, outer]).astype(np.float64)

        angles = np.linspace(0.0, 2.0 * np.pi, n_nodes - 1, endpoint=False)
        ring = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        return np.vstack([np.zeros((1, 2)), ring]).astype(np.float64)

    @staticmethod
    def _validate_params(params: L7_HolonomicParameters) -> None:
        if not isinstance(params.n_nodes, int) or isinstance(params.n_nodes, bool):
            raise ValueError("n_nodes must be a positive integer.")
        if params.n_nodes <= 0:
            raise ValueError("n_nodes must be positive.")
        if not isinstance(params.bitstream_length, int) or isinstance(
            params.bitstream_length, bool
        ):
            raise ValueError("bitstream_length must be a positive integer.")
        if params.bitstream_length <= 0:
            raise ValueError("bitstream_length must be positive.")
        if not np.isfinite(params.g_geometric_gain) or params.g_geometric_gain <= 0.0:
            raise ValueError("g_geometric_gain must be finite and positive.")
        if not np.isfinite(params.phi_golden_ratio) or params.phi_golden_ratio <= 0.0:
            raise ValueError("phi_golden_ratio must be finite and positive.")
        if not np.isfinite(params.coupling_leak) or not 0.0 <= params.coupling_leak < 1.0:
            raise ValueError("coupling_leak must be finite and in [0, 1).")

    def encode(self, domain_state: Any) -> jnp.ndarray:
        """Map symbolic phases to stochastic bitstreams."""
        # Activation = (1 + cos(phase)) / 2
        activation = (1.0 + jnp.cos(self.node_phases)) / 2.0

        self.rng_key, subkey = split_rng(self.rng_key)
        rands = uniform(subkey, (self.params.n_nodes, self.params.bitstream_length))
        bitstreams: jnp.ndarray = (rands < activation[:, None]).astype(jnp.uint8)
        return bitstreams

    @staticmethod
    @maybe_jit
    def _symbolic_kernel(
        phases: jnp.ndarray, metatron: jnp.ndarray, inputs: jnp.ndarray, dt: float
    ) -> jnp.ndarray:
        """Solve the symbolic routing dynamics.

        The update follows ``dTheta/dt = Metatron * inputs - decay``.
        """
        # Phases rotate based on weighted inputs from the Metatron routing
        drive = jnp.dot(metatron, inputs)
        d_phase = drive - 0.1 * phases
        return phases + d_phase * dt

    def step_jax(self, dt: float, inputs: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Advances the L7 holonomic dynamics using JAX.

        inputs: (n_nodes, bitstream_length) representing L6 or L8 signals.
        Returns: (n_nodes, bitstream_length) output bitstreams.
        """
        # 1. Extract Input Influence
        if inputs is not None:
            input_drive = jnp.mean(inputs.astype(jnp.float32), axis=1)
            if input_drive.shape[0] != self.params.n_nodes:
                input_drive = jnp.full((self.params.n_nodes,), jnp.mean(input_drive))
        else:
            input_drive = jnp.zeros((self.params.n_nodes,))

        # 2. Execute Symbolic Kernel
        self.node_phases = self._symbolic_kernel(
            self.node_phases, self.metatron_matrix, input_drive, dt
        )

        # 3. Return encoded bitstreams
        return self.encode(None)

    def decode(self, bitstreams: jnp.ndarray) -> Dict[str, float]:
        """Map bitstreams back to symbolic coherence telemetry."""
        return {"symbolic_unity_r7": float(jnp.abs(jnp.mean(jnp.exp(1j * self.node_phases))))}

    def get_metrics(self) -> Dict[str, float]:
        """Return L7-specific routing and phase-stability metrics."""
        return {
            "routing_coherence": float(jnp.abs(jnp.mean(jnp.exp(1j * self.node_phases)))),
            "metatron_stability": float(jnp.mean(jnp.cos(self.node_phases))),
        }
