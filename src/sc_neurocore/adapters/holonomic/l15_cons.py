# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L15 Consilium / Oversoul Integrator Adapter (JAX

"""
SCPN L15: Consilium / Oversoul Integrator Adapter (JAX Implementation)
=====================================================================

This module implements the JAX-accelerated uplift of Layer 15, focusing on
the Universal Metric Operator (UMO), Global Coherence Attractor (Omega),
and multi-objective Sustainable Ethical Coherence (SEC) optimization (Paper 15).

Key Equations:
- Universal Metric Operator (UMO): M = sum(w_i * L_i)
- Global Attractor: dOmega/dt = -grad(J_SEC)
- SEC Functional: J_SEC = integral(R_global - lambda * Surprise)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..base import BaseStochasticAdapter
from ._jax_compat import jnp, make_rng, maybe_jit, split_rng, uniform


@dataclass
class L15_HolonomicParameters:
    """Parameters derived from Paper 15 and executive optimization specs."""

    n_metric_dimensions: int = 16  # One per SCPN layer
    bitstream_length: int = 1024

    # Optimizer Constants
    sec_lambda: float = 0.1  # Surprise penalty weight
    learning_rate: float = 0.05  # Rate of attractor convergence
    coherence_target: float = 0.95


class L15_ConsiliumAdapter(BaseStochasticAdapter):
    """
    JAX-traceable adapter for the SCPN Consilium layer.
    """

    def __init__(self, params: Optional[L15_HolonomicParameters] = None, seed: int = 415) -> None:
        self.params = params or L15_HolonomicParameters()
        self.rng_key = make_rng(seed)

        # State: Universal Metric (Vector of layer weights)
        self.universal_metric = jnp.full(
            (self.params.n_metric_dimensions,), 1.0 / self.params.n_metric_dimensions
        )
        # State: Global Coherence Index (GCI)
        self.gci = 0.5
        # State: Collective Attractor Position
        self.attractor_pos = jnp.zeros((self.params.n_metric_dimensions,))

    def encode(self, domain_state: Any) -> jnp.ndarray:
        """
        Maps executive optimization state to stochastic bitstreams.
        """
        # GCI mapped to bitstream density
        self.rng_key, subkey = split_rng(self.rng_key)
        rands = uniform(subkey, (self.params.n_metric_dimensions, self.params.bitstream_length))
        bitstreams: jnp.ndarray = (rands < self.universal_metric[:, None] * self.gci * 10.0).astype(
            jnp.uint8
        )
        return bitstreams

    @staticmethod
    @maybe_jit
    def _umo_kernel(
        metric: jnp.ndarray, layer_coherences: jnp.ndarray, target: float, lr: float, dt: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Solves the UMO / SEC optimization:
        dMetric/dt = (Target - Coherence) * grad(Surprise)
        """
        # Calculate global coherence proxy
        gci_next = jnp.mean(layer_coherences)

        # Adjust metric weights toward the target attractor
        error = target - gci_next
        d_metric = lr * error * layer_coherences - 0.01 * metric
        metric_next = jnp.clip(metric + d_metric * dt, 0.0, 1.0)
        # Normalize weights
        metric_next = metric_next / (jnp.sum(metric_next) + 1e-6)

        return metric_next, gci_next

    def step_jax(self, dt: float, inputs: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Advances the L15 holonomic dynamics using JAX.

        inputs: (16, bitstream_length) representing coherences of all 16 layers.
        Returns: (16, bitstream_length) output bitstreams (Executive steering).
        """
        # 1. Extract Layer Coherences (The full stack feedback)
        if inputs is not None:
            layer_syncs = jnp.mean(inputs.astype(jnp.float32), axis=1)
            # Map input dimensions if partial stack
            if layer_syncs.shape[0] != self.params.n_metric_dimensions:
                layer_syncs = jnp.pad(
                    layer_syncs, (0, self.params.n_metric_dimensions - layer_syncs.shape[0])
                )
        else:
            layer_syncs = jnp.zeros((self.params.n_metric_dimensions,))

        # 2. Execute UMO Kernel
        self.universal_metric, self.gci = self._umo_kernel(
            self.universal_metric,
            layer_syncs,
            self.params.coherence_target,
            self.params.learning_rate,
            dt,
        )

        # 3. Return encoded bitstreams (The executive steering signal)
        return self.encode(None)

    def decode(self, bitstreams: jnp.ndarray) -> Dict[str, float]:
        """
        Maps bitstreams back to Global Coherence Index.
        """
        return {"global_coherence_r15": float(self.gci)}

    def get_metrics(self) -> Dict[str, float]:
        """
        Returns L15-specific metrics.
        """
        return {
            "gci_index": float(self.gci),
            "metric_entropy": float(
                -jnp.sum(self.universal_metric * jnp.log(self.universal_metric + 1e-6))
            ),
            "optimizer_error": float(self.params.coherence_target - self.gci),
        }
