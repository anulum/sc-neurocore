# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L13 Source-Field / Meta-Universal Adapter (JAX)

"""SCPN L13 source-field holonomic adapter.

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
    """Configuration for the Layer 13 source-field adapter.

    Parameters
    ----------
    n_vacuum_nodes:
        Number of nodes in the one-dimensional vacuum lattice.
    bitstream_length:
        Number of stochastic bits emitted per vacuum node.
    j_primordial_coupling:
        Finite nearest-neighbour primordial lattice coupling.
    h_potential_bias:
        Finite scalar source-field bias applied to each node.
    lambda_scission:
        Finite non-negative symmetry-breaking drive.
    """

    n_vacuum_nodes: int = 256
    bitstream_length: int = 1024

    # Ontological Constants
    j_primordial_coupling: float = 1.0
    h_potential_bias: float = 0.01
    lambda_scission: float = 0.1  # Rate of symmetry breaking


class L13_SourceAdapter(BaseStochasticAdapter):
    """JAX-traceable adapter for the SCPN source-field layer."""

    def __init__(self, params: Optional[L13_HolonomicParameters] = None, seed: int = 413) -> None:
        """Initialise the Layer 13 source-field adapter.

        Parameters
        ----------
        params:
            Optional vacuum-lattice configuration. Defaults preserve the
            historical 256-node, 1024-bitstream contract.
        seed:
            Random seed forwarded to the JAX or NumPy compatibility RNG.

        Raises
        ------
        ValueError
            If configuration values cannot produce finite bounded dynamics.
        """
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
        """Validate a strict positive integer configuration field."""
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    @classmethod
    def _validate_params(cls, params: L13_HolonomicParameters) -> None:
        """Validate Layer 13 parameters before allocating backend arrays."""
        cls._validate_positive_int("n_vacuum_nodes", params.n_vacuum_nodes)
        cls._validate_positive_int("bitstream_length", params.bitstream_length)

        if not np.isfinite(params.j_primordial_coupling):
            raise ValueError("j_primordial_coupling must be finite.")
        if not np.isfinite(params.h_potential_bias):
            raise ValueError("h_potential_bias must be finite.")
        if not np.isfinite(params.lambda_scission) or params.lambda_scission < 0.0:
            raise ValueError("lambda_scission must be finite and non-negative.")

    def encode(self, domain_state: Any) -> jnp.ndarray:
        """Map vacuum potential to stochastic source-field bitstreams.

        Parameters
        ----------
        domain_state:
            Reserved adapter payload for interface compatibility. Layer 13 uses
            its internal vacuum lattice state for encoding.

        Returns
        -------
        jnp.ndarray
            Rank-2 bitstream matrix with shape
            ``(n_vacuum_nodes, bitstream_length)``.
        """
        self.rng_key, subkey = split_rng(self.rng_key)
        rands = uniform(subkey, (self.params.n_vacuum_nodes, self.params.bitstream_length))
        bitstreams: jnp.ndarray = (rands < self.vacuum_state[:, None]).astype(jnp.uint8)
        return bitstreams

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
        """Advance local spin-like vacuum lattice dynamics.

        Parameters
        ----------
        state:
            Current bounded vacuum potential vector.
        coupling:
            Finite nearest-neighbour primordial lattice coupling.
        bias:
            Finite scalar source-field bias.
        scission_rate:
            Non-negative symmetry-breaking drive.
        feedback_drive:
            Bounded L16 cybernetic-closure feedback vector in ``[-1, 1]``.
        dt:
            Positive finite simulation timestep.

        Returns
        -------
        jnp.ndarray
            Updated bounded vacuum potential vector.
        """
        spin = 2.0 * jnp.clip(state, 0.0, 1.0) - 1.0
        neighbour_field = 0.5 * (jnp.roll(spin, -1) + jnp.roll(spin, 1))
        hamiltonian_drive = coupling * neighbour_field + bias + 0.25 * feedback_drive
        scission_drive = scission_rate * (spin - spin * spin * spin)
        relaxation = -0.05 * spin
        spin_next = spin + (hamiltonian_drive + scission_drive + relaxation) * dt
        return jnp.clip(0.5 * (spin_next + 1.0), 0.0, 1.0)

    @staticmethod
    def _validate_dt(dt: float) -> None:
        """Validate a positive finite simulation timestep."""
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive.")

    def _project_feedback(self, inputs: Optional[jnp.ndarray]) -> jnp.ndarray:
        """Project optional L16 feedback onto the configured vacuum lattice."""
        if inputs is None:
            return jnp.zeros((self.params.n_vacuum_nodes,))

        feedback = jnp.asarray(inputs)
        raw_inputs = np.asarray(feedback, dtype=float)
        if raw_inputs.ndim > 2:
            raise ValueError("inputs must have rank 0, 1, or 2.")
        if raw_inputs.ndim == 1 and raw_inputs.shape[0] == 0:
            raise ValueError("inputs must contain at least one value.")
        if raw_inputs.ndim == 2:
            if raw_inputs.shape[0] == 0:
                raise ValueError("inputs must contain at least one row.")
            if raw_inputs.shape[1] == 0:
                raise ValueError("inputs must contain at least one column.")
        if not np.all(np.isfinite(raw_inputs)):
            raise ValueError("inputs must contain only finite values.")

        if feedback.ndim == 0:
            raw = jnp.full((self.params.n_vacuum_nodes,), feedback)
        elif feedback.ndim == 1:
            raw = feedback.astype(jnp.float32)
        else:
            raw = jnp.mean(feedback.astype(jnp.float32), axis=1)

        if raw.shape[0] != self.params.n_vacuum_nodes:
            raw = jnp.full((self.params.n_vacuum_nodes,), jnp.mean(raw))
        return jnp.clip(2.0 * raw - 1.0, -1.0, 1.0)

    def step_jax(self, dt: float, inputs: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Advance the L13 holonomic dynamics using JAX-compatible arrays.

        Parameters
        ----------
        dt:
            Positive finite simulation timestep.
        inputs:
            Optional L16 cybernetic-closure feedback. Scalars broadcast across
            all nodes, rank-1 arrays map directly or broadcast by their mean
            when length differs, and rank-2 batches collapse by row mean before
            the same node-count projection.

        Returns
        -------
        jnp.ndarray
            Output bitstreams with shape ``(n_vacuum_nodes, bitstream_length)``.

        Raises
        ------
        ValueError
            If ``dt`` or ``inputs`` violates the bounded adapter contract.
        """
        self._validate_dt(dt)

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
        """Map bitstreams back to primordial source coherence.

        Parameters
        ----------
        bitstreams:
            Source-field stochastic bitstream matrix.

        Returns
        -------
        dict[str, float]
            Telemetry dictionary containing ``source_coherence_r13``.

        Raises
        ------
        ValueError
            If ``bitstreams`` is not a finite non-empty rank-2 matrix.
        """
        bitstream_batch: jnp.ndarray = jnp.asarray(bitstreams)
        raw_bitstreams = np.asarray(bitstream_batch, dtype=float)
        if raw_bitstreams.ndim != 2:
            raise ValueError("bitstreams must be a rank-2 matrix.")
        if raw_bitstreams.shape[0] == 0 or raw_bitstreams.shape[1] == 0:
            raise ValueError("bitstreams must be a non-empty matrix.")
        if not np.all(np.isfinite(raw_bitstreams)):
            raise ValueError("bitstreams must contain only finite values.")
        return {"source_coherence_r13": float(jnp.mean(bitstream_batch.astype(jnp.float32)))}

    def get_metrics(self) -> Dict[str, float]:
        """Return L13-specific vacuum and Fisher-metric telemetry.

        Returns
        -------
        dict[str, float]
            Current vacuum potential and Fisher information metric density.
        """
        return {
            "vacuum_potential": float(jnp.mean(self.vacuum_state)),
            "fisher_information_metric": float(jnp.mean(self.fim_density)),
        }
