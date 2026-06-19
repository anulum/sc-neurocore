# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L2 Neurochemical-Neurological Adapter (JAX

"""
SCPN L2: Neurochemical-Neurological Adapter (JAX Implementation)
================================================================

This module implements the JAX-accelerated uplift of Layer 2, specifically
focused on the IIIEF (Integrated Information-Induced EM Field) mechanism
and the H_QC Quantum-Classical bridge described in Papers 0-21.

Key Equations:
- IIIEF Field: nabla^2 Phi - (1/c^2) d^2 Phi/dt^2 = 4pi * alpha * Integrated_Info
- H_QC Bridge: H_vesicle + H_SNARE + H_Ca_sensor + H_trigger
- Neurochemical Transfer: H(omega) band-selection gating
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from ..base import BaseStochasticAdapter
from ._jax_compat import jnp, make_rng, maybe_jit, split_rng, uniform


@dataclass
class L2_HolonomicParameters:
    """Parameters derived from Paper 2 and Monograph 28."""

    n_transmitters: int = 4
    n_receptors: int = 500
    bitstream_length: int = 1024

    # IIIEF Constants
    alpha_iiief: float = 0.01  # Information-to-Field coupling constant
    c_info: float = 300.0  # Effective information propagation velocity

    # H_QC Bridge Parameters
    g_snare: float = 0.8  # SNARE complex formation gain
    v_critical: float = 1.2  # Critical voltage for quantum trigger

    # Neurochemical Tonus (L2 -> Core modulation)
    dopamine_gain: float = 1.5
    serotonin_leak: float = 0.9


class L2_NeurochemicalAdapter(BaseStochasticAdapter):
    """
    JAX-traceable adapter for the SCPN Neurochemical layer.
    """

    def __init__(self, params: Optional[L2_HolonomicParameters] = None, seed: int = 42) -> None:
        self.params = params or L2_HolonomicParameters()
        self._validate_params(self.params)
        self.rng_key = make_rng(seed)

        # State: Receptors (n_types, n_receptors)
        self.receptor_states = jnp.zeros((self.params.n_transmitters, self.params.n_receptors))
        # State: Information-Geometric Field potential
        self.phi_field = jnp.zeros((self.params.n_transmitters,))
        # State: Field velocity for the second-order IIIEF wave dynamics
        self.phi_velocity = jnp.zeros((self.params.n_transmitters,))
        # State: Concentrations
        self.concentrations = jnp.full((self.params.n_transmitters,), 0.5)

    @staticmethod
    def _validate_positive_int(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    @classmethod
    def _validate_params(cls, params: L2_HolonomicParameters) -> None:
        cls._validate_positive_int("n_transmitters", params.n_transmitters)
        cls._validate_positive_int("n_receptors", params.n_receptors)
        cls._validate_positive_int("bitstream_length", params.bitstream_length)

        if not np.isfinite(params.alpha_iiief) or params.alpha_iiief < 0.0:
            raise ValueError("alpha_iiief must be finite and non-negative.")
        if not np.isfinite(params.c_info) or params.c_info <= 0.0:
            raise ValueError("c_info must be finite and positive.")
        if not np.isfinite(params.g_snare) or params.g_snare <= 0.0:
            raise ValueError("g_snare must be finite and positive.")
        if not np.isfinite(params.v_critical) or params.v_critical <= 0.0:
            raise ValueError("v_critical must be finite and positive.")
        if not np.isfinite(params.dopamine_gain) or params.dopamine_gain <= 0.0:
            raise ValueError("dopamine_gain must be finite and positive.")
        if not np.isfinite(params.serotonin_leak) or not 0.0 <= params.serotonin_leak <= 1.0:
            raise ValueError("serotonin_leak must be finite and in the interval [0, 1].")

    def encode(self, domain_state: Any) -> jnp.ndarray:
        """
        Maps neurochemical concentrations to stochastic bitstreams.
        """
        # (n_transmitters, bitstream_length)
        self.rng_key, subkey = split_rng(self.rng_key)
        rands = uniform(subkey, (self.params.n_transmitters, self.params.bitstream_length))
        bitstreams: jnp.ndarray = (rands < self.concentrations[:, None]).astype(jnp.uint8)
        return bitstreams

    @staticmethod
    @maybe_jit
    def _iiief_kernel(
        phi: jnp.ndarray,
        velocity: jnp.ndarray,
        integrated_info: jnp.ndarray,
        alpha: float,
        c_info: float,
        dt: float,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Advances the damped finite-difference IIIEF wave equation.
        """
        source = jnp.clip(integrated_info, 0.0, 1.0)
        laplacian = jnp.roll(phi, -1) - 2.0 * phi + jnp.roll(phi, 1)
        courant = c_info / (1.0 + c_info)
        acceleration = (
            4.0 * jnp.pi * alpha * source
            + courant * courant * laplacian
            - 0.15 * velocity
            - 0.05 * phi
        )
        velocity_next = velocity + acceleration * dt
        phi_next = phi + velocity_next * dt
        return phi_next, velocity_next

    def step_jax(self, dt: float, inputs: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Advances the L2 holonomic dynamics using JAX.

        inputs: (n_transmitters, bitstream_length) representing L1 or L5 feedback.
        Returns: (n_transmitters, bitstream_length) output bitstreams.
        """
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive.")

        # 1. Calculate Integrated Information Proxy (Phi_integrated) from inputs
        if inputs is not None:
            raw_phi = jnp.mean(inputs.astype(jnp.float32), axis=1)
            # Map input dimensions to transmitter count if necessary
            if raw_phi.shape[0] != self.params.n_transmitters:
                # Simple average-pooling projection
                phi_int = jnp.full((self.params.n_transmitters,), jnp.mean(raw_phi))
            else:
                phi_int = raw_phi
        else:
            phi_int = jnp.zeros((self.params.n_transmitters,))

        # 2. Update IIIEF Field
        self.phi_field, self.phi_velocity = self._iiief_kernel(
            self.phi_field,
            self.phi_velocity,
            phi_int,
            self.params.alpha_iiief,
            self.params.c_info,
            dt,
        )

        # 3. H_QC Bridge: Field modulates concentrations (Vesicle release)
        # H_int = -lambda * Psi * sigma -> mapped to P_release modulation
        trigger = 1.0 / (
            1.0 + jnp.exp(-self.params.dopamine_gain * (self.phi_field - self.params.v_critical))
        )
        release_mod = (
            self.params.serotonin_leak
            + (1.0 - self.params.serotonin_leak) * self.params.g_snare * trigger
        )
        self.concentrations = jnp.clip(self.concentrations * release_mod, 0.0, 1.0)

        # 4. Return encoded bitstreams for hardware consumption
        return self.encode(None)

    def decode(self, bitstreams: jnp.ndarray) -> Dict[str, float]:
        """
        Maps bitstreams back to neurochemical concentrations.
        """
        means = jnp.mean(bitstreams.astype(jnp.float32), axis=1)
        return {
            "dopamine": float(means[0]),
            "serotonin": float(means[1]),
            "norepinephrine": float(means[2]),
            "acetylcholine": float(means[3]),
        }

    def get_metrics(self) -> Dict[str, float]:
        """
        Returns L2-specific metrics like Field Potential and Tonus.
        """
        return {
            "avg_field_potential": float(jnp.mean(self.phi_field)),
            "avg_field_velocity": float(jnp.mean(jnp.abs(self.phi_velocity))),
            "system_coherence_r2": float(jnp.mean(self.concentrations)),
        }
