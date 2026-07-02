# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L6 Planetary-Biospheric Adapter (JAX Implementation)

"""SCPN L6 planetary-biospheric holonomic adapter.

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
    """Configuration for the Layer 6 Gaia-field adapter.

    Parameters
    ----------
    n_regions:
        Number of regional planetary field nodes.
    bitstream_length:
        Number of stochastic bits emitted per region.
    f_schumann:
        Positive finite Schumann resonance frequency in hertz.
    q_factor:
        Positive finite cavity quality factor.
    alpha_gaia:
        Positive finite regional-to-planetary coupling strength.
    p_percolation:
        Critical percolation threshold in the open interval ``(0, 1)``.
    """

    n_regions: int = 100
    bitstream_length: int = 1024

    # Schumann Resonance Constants
    f_schumann: float = 7.83  # Hz (Fundamental mode)
    q_factor: float = 4.0  # Cavity resonance quality

    # Planetary Coupling
    alpha_gaia: float = 0.05  # Individual-to-Planetary coupling strength
    p_percolation: float = 0.592  # Critical threshold for global coherence


class L6_PlanetaryAdapter(BaseStochasticAdapter):
    """JAX-traceable adapter for the SCPN planetary-biospheric layer."""

    def __init__(self, params: Optional[L6_HolonomicParameters] = None, seed: int = 46) -> None:
        """Initialise the Layer 6 planetary adapter.

        Parameters
        ----------
        params:
            Optional Gaia-field configuration. Defaults keep the historical
            100-region, 1024-bitstream contract.
        seed:
            Random seed forwarded to the JAX or NumPy compatibility RNG.

        Raises
        ------
        ValueError
            If configuration values cannot produce bounded finite dynamics.
        """
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
        """Map planetary coherence to stochastic regional bitstreams.

        Parameters
        ----------
        domain_state:
            Reserved adapter payload for interface compatibility. Layer 6 uses
            its internal regional coherence state for encoding.

        Returns
        -------
        jnp.ndarray
            Rank-2 bitstream matrix with shape ``(n_regions, bitstream_length)``.
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
        """Solve the planetary Gaia-field dynamics.

        Parameters
        ----------
        phi:
            Current planetary field potential.
        sync_inputs:
            Bounded regional synchronisation drive.
        alpha:
            Gaia coupling strength.
        freq:
            Schumann resonance frequency.
        q_factor:
            Resonance quality factor controlling coherent gain.
        p_percolation:
            Critical percolation threshold.
        t:
            Simulation time after the current step increment.
        dt:
            Positive finite simulation timestep.

        Returns
        -------
        tuple[jnp.ndarray, jnp.ndarray]
            Updated field potential and regional coherence vectors.
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
    def _validate_positive_int(name: str, value: int) -> None:
        """Validate a strict positive integer configuration field."""
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")

    @classmethod
    def _validate_params(cls, params: L6_HolonomicParameters) -> None:
        """Validate Layer 6 parameters before allocating backend arrays."""
        cls._validate_positive_int("n_regions", params.n_regions)
        cls._validate_positive_int("bitstream_length", params.bitstream_length)
        for field_name in ("f_schumann", "q_factor", "alpha_gaia"):
            value = float(getattr(params, field_name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field_name} must be finite and positive.")
        if not np.isfinite(params.p_percolation) or not 0.0 < params.p_percolation < 1.0:
            raise ValueError("p_percolation must be finite and in (0, 1).")

    @staticmethod
    def _validate_dt(dt: float) -> None:
        """Validate a positive finite simulation timestep."""
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive.")

    def _validate_input_batch(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """Validate and normalise an upstream L6 bitstream batch."""
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

    def _project_sync_drive(self, inputs: Optional[jnp.ndarray]) -> jnp.ndarray:
        """Project optional upstream bitstreams onto the configured L6 regions."""
        if inputs is None:
            return jnp.zeros((self.params.n_regions,))

        input_batch = self._validate_input_batch(inputs)
        sync_drive = jnp.mean(input_batch.astype(jnp.float32), axis=1)
        if sync_drive.shape[0] != self.params.n_regions:
            sync_drive = jnp.full((self.params.n_regions,), jnp.mean(sync_drive))
        return sync_drive

    def step_jax(self, dt: float, inputs: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Advance the L6 holonomic dynamics using JAX-compatible arrays.

        Parameters
        ----------
        dt:
            Positive finite simulation timestep.
        inputs:
            Optional ``(N, bitstream_length)`` upstream organismal output. If
            ``N`` differs from ``n_regions``, the mean regional drive is
            broadcast across all configured regions.

        Returns
        -------
        jnp.ndarray
            Output bitstreams with shape ``(n_regions, bitstream_length)``.

        Raises
        ------
        ValueError
            If ``dt`` or ``inputs`` violates the bounded adapter contract.
        """
        self._validate_dt(dt)
        sync_drive = self._project_sync_drive(inputs)
        self.t += dt

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
        """Map bitstreams back to the global coherence index.

        Parameters
        ----------
        bitstreams:
            Regional stochastic bitstream matrix.

        Returns
        -------
        dict[str, float]
            Telemetry dictionary containing ``global_coherence_index``.
        """
        return {"global_coherence_index": float(jnp.mean(bitstreams.astype(jnp.float32)))}

    def get_metrics(self) -> Dict[str, float]:
        """Return L6-specific Gaia and Schumann telemetry.

        Returns
        -------
        dict[str, float]
            Current Gaia potential, percolation index, and Schumann phase.
        """
        return {
            "gaia_potential": float(jnp.mean(self.phi_planetary)),
            "percolation_index": float(jnp.mean(self.regional_coherence)),
            "schumann_phase": float(self.t * self.params.f_schumann % 1.0),
        }
