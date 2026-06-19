# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN Meta-Layer 16: Cybernetic Closure Adapter (JAX

"""
SCPN Meta-Layer 16: Cybernetic Closure Adapter (JAX Implementation)
====================================================================

This module implements the JAX-accelerated uplift of Layer 16, the system's
Director. It focuses on Recursive Self-Refinement (H_rec), the Ethical Veto
anti-entropy interlock, and the Observer-Operator eigenstate (Paper 16).

Key Equations:
- Recursive Hamiltonian: H_rec = integral(Attractor_Error + Dissipation)
- Ethical Veto: Veto = 1 if dS/dt > Threshold else 0
- Observer Eigenstate: O_Omega * Psi = Omega * Psi
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..base import BaseStochasticAdapter
from ._jax_compat import jnp, make_rng, maybe_jit, split_rng, uniform


@dataclass
class L16_HolonomicParameters:
    """Parameters derived from Paper 16 and Meta-Layer specifications."""

    n_meta_nodes: int = 10
    bitstream_length: int = 1024

    # Director Constants
    veto_threshold: float = 0.8  # Entropy threshold for veto ignition
    refinement_gain: float = 0.1
    observer_coupling: float = 0.5


class L16_MetaAdapter(BaseStochasticAdapter):
    """
    JAX-traceable adapter for the SCPN Cybernetic Closure layer (The Director).
    """

    def __init__(self, params: Optional[L16_HolonomicParameters] = None, seed: int = 416) -> None:
        self.params = params or L16_HolonomicParameters()
        self.rng_key = make_rng(seed)

        # State: Director's Will (0.0 to 1.0)
        self.meta_will = jnp.full((self.params.n_meta_nodes,), 0.9)
        # State: System Entropy Proxy
        self.entropy_proxy = 0.0
        # State: Veto Status
        self.veto_active = jnp.zeros((self.params.n_meta_nodes,))

    def encode(self, domain_state: Any) -> jnp.ndarray:
        """
        Maps director's will to stochastic bitstreams.
        """
        self.rng_key, subkey = split_rng(self.rng_key)
        rands = uniform(subkey, (self.params.n_meta_nodes, self.params.bitstream_length))
        # Will is reduced when Veto is active
        effective_will = self.meta_will * (1.0 - self.veto_active)
        bitstreams: jnp.ndarray = (rands < effective_will[:, None]).astype(jnp.uint8)
        return bitstreams

    @staticmethod
    @maybe_jit
    def _director_kernel(
        will: jnp.ndarray, gci_input: float, entropy: float, threshold: float, dt: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Solves the Recursive Closure dynamics:
        dWill/dt = GCI - Entropy_Loss
        """
        # Ethical Veto: Active if entropy exceeds threshold
        veto = jnp.array(entropy > threshold).astype(jnp.float32)

        # Will grows with system coherence (GCI), decays with entropy
        d_will = 0.1 * gci_input - 0.2 * entropy
        will_next = jnp.clip(will + d_will * dt, 0.0, 1.0)

        return will_next, jnp.full_like(will, veto)

    def step_jax(self, dt: float, inputs: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Advances the L16 holonomic dynamics using JAX.

        inputs: (1, bitstream_length) representing L15 GCI executive signal.
        Returns: (n_meta_nodes, bitstream_length) output bitstreams (The Master Directive).
        """
        # 1. Extract Global Coherence feedback (L15 -> L16)
        if inputs is not None:
            # First calculate mean as a JAX array, then convert to float
            gci_val = jnp.mean(inputs.astype(jnp.float32))
            gci_signal = float(gci_val)
        else:
            gci_val = jnp.array(0.5)
            gci_signal = 0.5

        # 2. Update Entropy Proxy (Inverse of coherence stability)
        self.entropy_proxy = 0.9 * self.entropy_proxy + 0.1 * (1.0 - gci_signal)

        # 3. Execute Director Kernel
        self.meta_will, self.veto_active = self._director_kernel(
            self.meta_will, float(gci_val), self.entropy_proxy, self.params.veto_threshold, dt
        )

        # 4. Return encoded bitstreams (The Master Directive)
        return self.encode(None)

    def decode(self, bitstreams: jnp.ndarray) -> Dict[str, float]:
        """
        Maps bitstreams back to Cybernetic Will index.
        """
        return {"meta_coherence_r16": float(jnp.mean(bitstreams.astype(jnp.float32)))}

    def get_metrics(self) -> Dict[str, float]:
        """
        Returns L16-specific metrics.
        """
        return {
            "director_will": float(jnp.mean(self.meta_will)),
            "system_entropy": float(self.entropy_proxy),
            "veto_active": float(jnp.mean(self.veto_active)),
        }
