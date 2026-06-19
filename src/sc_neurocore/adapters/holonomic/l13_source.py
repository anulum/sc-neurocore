# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L13 Source-Field / Meta-Universal Adapter (JAX

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

import numpy as np

from ..base import BaseStochasticAdapter
from ._jax_compat import jnp, make_rng, maybe_jit, split_rng, uniform


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
        self._validate_params(self.params)
        self.rng_key = make_rng(seed)

        # State: Vacuum Potential (0.0 to 1.0)
        self.vacuum_state = jnp.full((self.params.n_vacuum_nodes,), 0.5)
        if self.params.lambda_scission > 0.0:
            self.rng_key, subkey = split_rng(self.rng_key)
            amplitude = min(float(self.params.lambda_scission), 1.0) * 0.02
            perturbation = (uniform(subkey, (self.params.n_vacuum_nodes,)) - 0.5) * amplitude
            self.vacuum_state = jnp.clip(self.vacuum_state + perturbation, 0.0, 1.0)
        # State: Fisher Information Metric Density
        self.fim_density = jnp.zeros((self.params.n_vacuum_nodes,))

    @staticmethod
    def _validate_positive_int(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    @classmethod
    def _validate_params(cls, params: L13_HolonomicParameters) -> None:
        cls._validate_positive_int("n_vacuum_nodes", params.n_vacuum_nodes)
        cls._validate_positive_int("bitstream_length", params.bitstream_length)

        if not np.isfinite(params.j_primordial_coupling):
            raise ValueError("j_primordial_coupling must be finite.")
        if not np.isfinite(params.h_potential_bias):
            raise ValueError("h_potential_bias must be finite.")
        if not np.isfinite(params.lambda_scission) or params.lambda_scission < 0.0:
            raise ValueError("lambda_scission must be finite and non-negative.")

    def encode(self, domain_state: Any) -> jnp.ndarray:
        """
        Maps vacuum potential to stochastic bitstreams.
        """
        self.rng_key, subkey = split_rng(self.rng_key)
        rands = uniform(subkey, (self.params.n_vacuum_nodes, self.params.bitstream_length))
        bitstreams: jnp.ndarray = (rands < self.vacuum_state[:, None]).astype(jnp.uint8)
        return bitstreams

    @staticmethod
    @maybe_jit
    def _vacuum_kernel(state: jnp.ndarray, coupling: float, bias: float, dt: float) -> jnp.ndarray:
        """
        Advances local spin-like vacuum lattice dynamics.
        """
        result: jnp.ndarray = L13_SourceAdapter._vacuum_lattice_kernel(
            state, coupling, bias, 0.0, jnp.zeros_like(state), dt
        )
        return result

    @staticmethod
    @maybe_jit
    def _vacuum_lattice_kernel(
        state: jnp.ndarray,
        coupling: float,
        bias: float,
        scission_rate: float,
        feedback_drive: jnp.ndarray,
        dt: float,
    ) -> jnp.ndarray:
        """
        Advances local spin-like vacuum lattice dynamics.
        """
        spin = 2.0 * jnp.clip(state, 0.0, 1.0) - 1.0
        neighbour_field = 0.5 * (jnp.roll(spin, -1) + jnp.roll(spin, 1))
        hamiltonian_drive = coupling * neighbour_field + bias + 0.25 * feedback_drive
        scission_drive = scission_rate * (spin - spin * spin * spin)
        relaxation = -0.05 * spin
        spin_next = spin + (hamiltonian_drive + scission_drive + relaxation) * dt
        return jnp.clip(0.5 * (spin_next + 1.0), 0.0, 1.0)

    def _project_feedback(self, inputs: Optional[jnp.ndarray]) -> jnp.ndarray:
        if inputs is None:
            return jnp.zeros((self.params.n_vacuum_nodes,))

        raw_inputs = np.asarray(inputs, dtype=float)
        if not np.all(np.isfinite(raw_inputs)):
            raise ValueError("inputs must contain only finite values.")

        feedback = jnp.asarray(inputs).astype(jnp.float32)
        if feedback.ndim == 0:
            raw = jnp.full((self.params.n_vacuum_nodes,), feedback)
        elif feedback.ndim == 1:
            raw = feedback
        else:
            raw = jnp.mean(feedback, axis=1)

        if raw.shape[0] != self.params.n_vacuum_nodes:
            raw = jnp.full((self.params.n_vacuum_nodes,), jnp.mean(raw))
        return jnp.clip(2.0 * raw - 1.0, -1.0, 1.0)

    def step_jax(self, dt: float, inputs: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Advances the L13 holonomic dynamics using JAX.

        inputs: Optional feedback from L16 (Cybernetic Closure).
        Returns: (n_vacuum_nodes, bitstream_length) output bitstreams.
        """
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive.")

        previous_state = self.vacuum_state
        feedback_drive = self._project_feedback(inputs)

        # 1. Update Vacuum State
        self.vacuum_state = self._vacuum_lattice_kernel(
            self.vacuum_state,
            self.params.j_primordial_coupling,
            self.params.h_potential_bias,
            self.params.lambda_scission,
            feedback_drive,
            dt,
        )

        # 2. Update FIM Density (Measures rate of change / information work)
        # Bernoulli-local Fisher density from temporal and lattice gradients.
        variance = jnp.clip(self.vacuum_state * (1.0 - self.vacuum_state), 1e-6, None)
        temporal_delta = self.vacuum_state - previous_state
        lattice_delta = jnp.roll(self.vacuum_state, -1) - self.vacuum_state
        instant_fim = (temporal_delta * temporal_delta + lattice_delta * lattice_delta) / variance
        self.fim_density = 0.9 * self.fim_density + 0.1 * instant_fim

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
