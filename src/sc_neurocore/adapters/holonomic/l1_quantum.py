# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L1 Quantum Biological Adapter (JAX Implementation)

"""
SCPN L1: Quantum Biological Adapter (JAX Implementation)
========================================================

This module implements the JAX-accelerated uplift of Layer 1, focusing on
the Fröhlich Pumping mechanism, the Ignition Hamiltonian, and the
Quantum-Classical bridge as described in Paper 1.

Key Equations:
- Metabolic Ignition: S_pump >= S_crit (Coherence emergence)
- Phase-to-Angle: theta = 2 * arcsin(sqrt(p)) (Isomorphism from Paper 27/28)
- Lindblad Dynamics: dRho/dt = -i[H, Rho] + L(Rho) (Simplified for SC)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..base import BaseStochasticAdapter
from ._jax_compat import jnp, make_rng, maybe_jit, split_rng, uniform


@dataclass
class L1_HolonomicParameters:
    """Parameters derived from Paper 1 and Monograph 28."""

    n_qubits: int = 1000
    bitstream_length: int = 1024

    # Fröhlich / Ignition Constants
    s_critical: float = 0.5  # Critical pumping threshold
    gamma_decoherence: float = 0.05  # Baseline decoherence rate
    f_non_markov: float = 100.0  # Protection factor

    # Coupling
    zeta_source: float = 0.1  # Coupling to L13 Source Field

    # Execution Backend
    # "simulated", "qiskit.aer_simulator", "pennylane.default.qubit"
    backend: str = "simulated"


class L1_QuantumAdapter(BaseStochasticAdapter):
    """
    JAX-traceable adapter for the SCPN Quantum Biological layer.
    """

    def __init__(self, params: Optional[L1_HolonomicParameters] = None, seed: int = 41) -> None:
        self.params = params or L1_HolonomicParameters()
        self.rng_key = make_rng(seed)

        # State: Coherence Probabilities (0.0 to 1.0)
        self.coherence = jnp.full((self.params.n_qubits,), 0.95)
        # State: Metabolic Pumping Level
        self.s_pump = jnp.zeros((self.params.n_qubits,))

    def encode(self, domain_state: Any) -> jnp.ndarray:
        """
        Maps coherence probabilities to stochastic bitstreams.
        """
        self.rng_key, subkey = split_rng(self.rng_key)
        rands = uniform(subkey, (self.params.n_qubits, self.params.bitstream_length))
        bitstreams: jnp.ndarray = (rands < self.coherence[:, None]).astype(jnp.uint8)
        return bitstreams

    @staticmethod
    @maybe_jit
    def _ignition_kernel(
        coherence: jnp.ndarray,
        s_pump: jnp.ndarray,
        s_crit: float,
        gamma: float,
        f_prot: float,
        dt: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Solves the Ignition / Metabolic Coherence dynamics:
        dC/dt = (S_pump - S_crit) * C - (gamma/log(F)) * C
        """
        # Effective decoherence reduced by protection factor
        effective_gamma = gamma / jnp.log10(f_prot)

        # Coherence growth depends on metabolic surplus
        growth = (s_pump - s_crit) * coherence
        dc = growth - effective_gamma * coherence

        coherence_next = jnp.clip(coherence + dc * dt, 0.0, 1.0)

        # Simplified S_pump recovery
        s_pump_next = jnp.clip(s_pump - 0.1 * dt, 0.0, 1.0)

        return coherence_next, s_pump_next

    def step_jax(self, dt: float, inputs: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Advances the L1 holonomic dynamics using JAX.

        inputs: (n_qubits, bitstream_length) representing metabolic/field drive (L4 or L13).
        Returns: (n_qubits, bitstream_length) output bitstreams.
        """
        # 1. Update Metabolic Pumping (S_pump) from inputs
        if inputs is not None:
            drive = jnp.mean(inputs.astype(jnp.float32), axis=1)
            self.s_pump = jnp.clip(self.s_pump + drive * dt, 0.0, 1.0)

        # 2. Execute Ignition Kernel
        self.coherence, self.s_pump = self._ignition_kernel(
            self.coherence,
            self.s_pump,
            self.params.s_critical,
            self.params.gamma_decoherence,
            self.params.f_non_markov,
            dt,
        )

        # 3. Phase-to-Angle Isomorphism (Optional: for use with true hardware)
        # theta = 2 * jnp.arcsin(jnp.sqrt(self.coherence))

        # 4. Return encoded bitstreams
        return self.encode(None)

    def decode(self, bitstreams: jnp.ndarray) -> Dict[str, float]:
        """
        Maps bitstreams back to global coherence metric.
        """
        return {"avg_coherence": float(jnp.mean(bitstreams.astype(jnp.float32)))}

    def get_metrics(self) -> Dict[str, float]:
        """
        Returns L1-specific metrics like Coherence and Pumping levels.
        """
        return {
            "r1_global_coherence": float(jnp.mean(self.coherence)),
            "avg_metabolic_pumping": float(jnp.mean(self.s_pump)),
        }
