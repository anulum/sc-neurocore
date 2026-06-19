# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L6 Planetary-Biospheric Adapter (JAX Implementation)

"""
SCPN L6: Planetary-Biospheric Adapter (JAX Implementation)
==========================================================

This module implements the JAX-accelerated uplift of Layer 6, focusing on
Schumann Resonance coupling, Planetary Superradiance (P ~ N^2), and the
Percolation Phase Transition of global consciousness described in Paper 6.

Key Equations:
- Schumann Coupling: Psi_P = Psi_local * exp(i * omega_S * t) where omega_S ~ 7.83Hz
- Biospheric Superradiance: Power_total = N^2 * Power_individual
- Percolation Transition: Coherence_global = f(p - p_c) where p_c ~ 0.59
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..base import BaseStochasticAdapter
from ._jax_compat import jnp, make_rng, maybe_jit, split_rng, uniform


@dataclass
class L6_HolonomicParameters:
    """Parameters derived from Paper 6 and Gaia-field specifications."""

    n_regions: int = 100
    bitstream_length: int = 1024

    # Schumann Resonance Constants
    f_schumann: float = 7.83  # Hz (Fundamental mode)
    q_factor: float = 4.0  # Cavity resonance quality

    # Planetary Coupling
    alpha_gaia: float = 0.05  # Individual-to-Planetary coupling strength
    p_percolation: float = 0.592  # Critical threshold for global coherence


class L6_PlanetaryAdapter(BaseStochasticAdapter):
    """
    JAX-traceable adapter for the SCPN Planetary-Biospheric layer.
    """

    def __init__(self, params: Optional[L6_HolonomicParameters] = None, seed: int = 46) -> None:
        self.params = params or L6_HolonomicParameters()
        self._validate_params(self.params)
        self.rng_key = make_rng(seed)

        # State: Planetary Field Potential (Psi_P)
        self.phi_planetary = jnp.zeros((self.params.n_regions,))
        # State: Regional Coherence index
        self.regional_coherence = jnp.full((self.params.n_regions,), 0.1)
        # Time tracking for oscillatory resonance
        self.t = 0.0

    def encode(self, domain_state: Any) -> jnp.ndarray:
        """
        Maps planetary coherence to stochastic bitstreams.
        """
        self.rng_key, subkey = split_rng(self.rng_key)
        rands = uniform(subkey, (self.params.n_regions, self.params.bitstream_length))
        bitstreams: jnp.ndarray = (rands < self.regional_coherence[:, None]).astype(jnp.uint8)
        return bitstreams

    @staticmethod
    @maybe_jit
    def _gaia_kernel(
        phi: jnp.ndarray,
        sync_inputs: jnp.ndarray,
        alpha: float,
        freq: float,
        q_factor: float,
        p_percolation: float,
        t: float,
        dt: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Solves the Planetary Gaia-field dynamics:
        dPhi/dt = alpha * sync_inputs * G(R, Q) * cos(2*pi*f*t) - decay * Phi
        """
        bounded_sync = jnp.clip(sync_inputs, 0.0, 1.0)
        order_parameter = jnp.clip(jnp.mean(bounded_sync), 0.0, 1.0)

        # Schumann resonance driving term
        driver = jnp.cos(2.0 * jnp.pi * freq * t)
        superradiant_gain = 1.0 + q_factor * order_parameter**2
        d_phi = alpha * bounded_sync * superradiant_gain * driver - 0.05 * phi

        phi_next = phi + d_phi * dt

        percolation_gate = 1.0 / (1.0 + jnp.exp(-q_factor * (order_parameter - p_percolation)))
        local_field_activation = 1.0 - jnp.exp(-q_factor * jnp.abs(phi_next))
        coherence_next = jnp.clip(percolation_gate * local_field_activation, 0.0, 1.0)

        return phi_next, coherence_next

    @staticmethod
    def _validate_params(params: L6_HolonomicParameters) -> None:
        if not isinstance(params.n_regions, int) or isinstance(params.n_regions, bool):
            raise ValueError("n_regions must be a positive integer.")
        if params.n_regions <= 0:
            raise ValueError("n_regions must be positive.")
        if not isinstance(params.bitstream_length, int) or isinstance(
            params.bitstream_length, bool
        ):
            raise ValueError("bitstream_length must be a positive integer.")
        if params.bitstream_length <= 0:
            raise ValueError("bitstream_length must be positive.")
        for field_name in ("f_schumann", "q_factor", "alpha_gaia"):
            value = float(getattr(params, field_name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive.")
        if not np.isfinite(params.p_percolation) or not 0.0 < params.p_percolation < 1.0:
            raise ValueError("p_percolation must be finite and in (0, 1).")

    def step_jax(self, dt: float, inputs: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Advances the L6 holonomic dynamics using JAX.

        inputs: (n_regions, bitstream_length) representing L5 Organismal output.
        Returns: (n_regions, bitstream_length) output bitstreams.
        """
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive.")
        self.t += dt

        # 1. Extract Organismal Synchronization (L5 -> L6)
        if inputs is not None:
            sync_drive = jnp.mean(inputs.astype(jnp.float32), axis=1)
            # Map input dimensions to regional count
            if sync_drive.shape[0] != self.params.n_regions:
                sync_drive = jnp.full((self.params.n_regions,), jnp.mean(sync_drive))
        else:
            sync_drive = jnp.zeros((self.params.n_regions,))

        # 2. Execute Gaia Kernel
        self.phi_planetary, self.regional_coherence = self._gaia_kernel(
            self.phi_planetary,
            sync_drive,
            self.params.alpha_gaia,
            self.params.f_schumann,
            self.params.q_factor,
            self.params.p_percolation,
            self.t,
            dt,
        )

        # 3. Return encoded bitstreams
        return self.encode(None)

    def decode(self, bitstreams: jnp.ndarray) -> Dict[str, float]:
        """
        Maps bitstreams back to Global Consciousness Index.
        """
        return {"global_coherence_index": float(jnp.mean(bitstreams.astype(jnp.float32)))}

    def get_metrics(self) -> Dict[str, float]:
        """
        Returns L6-specific metrics like Gaia Potential and Schumann Alignment.
        """
        return {
            "gaia_potential": float(jnp.mean(self.phi_planetary)),
            "percolation_index": float(jnp.mean(self.regional_coherence)),
            "schumann_phase": float(self.t * self.params.f_schumann % 1.0),
        }
