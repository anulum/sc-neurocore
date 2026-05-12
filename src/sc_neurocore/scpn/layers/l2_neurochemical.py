# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L2 Neurochemical Layer (Stochastic Implementation)

from typing import Any, Optional

"""
SCPN L2: Neurochemical Layer (Stochastic Implementation)
=========================================================

Implements Layer 2 of the SCPN framework: Neurochemical signaling including
receptors, neurotransmitters, and second messenger cascades.

Key Features:
- Stochastic receptor binding dynamics
- Neurotransmitter diffusion via bitstream spreading
- Integration with bio/neuromodulation.py

"""

import logging
import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class L2_StochasticParameters:
    """Parameters for the Stochastic L2 Neurochemical Layer."""

    n_receptors: int = 500
    n_neurotransmitter_types: int = 4  # DA, 5HT, NE, ACh
    bitstream_length: int = 1024

    # Receptor dynamics
    binding_affinity: float = 0.7  # Probability of binding when NT present
    unbinding_rate: float = 0.1  # Rate of receptor release

    # Diffusion parameters
    diffusion_rate: float = 0.05
    reuptake_rate: float = 0.03

    # Coupling to L1 (quantum) and L3 (genomic)
    quantum_coupling: float = 0.1
    genomic_coupling: float = 0.15
    rng_seed: Optional[int] = None


class L2_NeurochemicalLayer:
    """
    Stochastic implementation of the Neurochemical Signaling Layer.

    Models receptor-ligand binding, neurotransmitter dynamics, and
    second messenger cascades using bitstream representations.
    """

    # Neurotransmitter indices
    DA = 0  # Dopamine
    SEROTONIN = 1  # 5-HT
    NE = 2  # Norepinephrine
    ACH = 3  # Acetylcholine

    def __init__(self, params: Optional[L2_StochasticParameters] = None):
        self.params = params or L2_StochasticParameters()
        self._validate_params(self.params)
        self._rng = np.random.default_rng(self.params.rng_seed)

        # Receptor states: 0 = unbound, 1 = bound
        self.receptor_states = np.zeros(
            (self.params.n_neurotransmitter_types, self.params.n_receptors), dtype=np.float32
        )

        # Neurotransmitter concentrations (as probabilities for bitstream encoding)
        self.nt_concentrations = np.ones(self.params.n_neurotransmitter_types) * 0.5

        # Second messenger cascade state (cAMP, Ca2+, etc.)
        self.second_messenger_levels = np.zeros(self.params.n_neurotransmitter_types)

        # History for temporal dynamics
        self.history: list[Any] = []

    def step(
        self,
        dt: float,
        nt_release: Optional[np.ndarray[Any, Any]] = None,
        l1_input: Optional[np.ndarray[Any, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Advance the layer by one time step.

        Args:
            dt: Time step in seconds.
            nt_release: Neurotransmitter release rates [4] (0-1 normalized).
            l1_input: Quantum layer input (coherence modulation).

        Returns:
            Dict with receptor_activity, second_messengers, output_bitstreams
        """
        self._validate_step_inputs(dt, nt_release, l1_input, self.params.n_neurotransmitter_types)
        # 1. Update neurotransmitter concentrations from release
        if nt_release is not None:
            self.nt_concentrations = np.clip(
                self.nt_concentrations
                + np.asarray(nt_release, dtype=np.float64) * dt
                - self.params.reuptake_rate * dt,
                0.0,
                1.0,
            )

        # 2. Receptor binding dynamics (stochastic)
        for nt_idx in range(self.params.n_neurotransmitter_types):
            nt_conc = self.nt_concentrations[nt_idx]

            # Binding: P(bind) = affinity * concentration * (1 - current_state)
            binding_prob = self.params.binding_affinity * nt_conc
            bind_mask = self._rng.random(self.params.n_receptors) < binding_prob * dt

            # Unbinding: P(unbind) = unbinding_rate * current_state
            unbind_mask = (
                self._rng.random(self.params.n_receptors) < self.params.unbinding_rate * dt
            )

            # Update states
            self.receptor_states[nt_idx] = np.where(
                bind_mask & (self.receptor_states[nt_idx] < 0.5), 1.0, self.receptor_states[nt_idx]
            )
            self.receptor_states[nt_idx] = np.where(
                unbind_mask & (self.receptor_states[nt_idx] > 0.5),
                0.0,
                self.receptor_states[nt_idx],
            )

        # 3. Second messenger cascade
        receptor_activity = np.mean(self.receptor_states, axis=1)
        self.second_messenger_levels = 0.9 * self.second_messenger_levels + 0.1 * receptor_activity

        # 4. Quantum coupling (L1 modulates receptor sensitivity)
        if l1_input is not None:
            quantum_mod = self._finite_mean(l1_input, "l1_input") * self.params.quantum_coupling
            self.receptor_states *= 1.0 + quantum_mod
            self.receptor_states = np.clip(self.receptor_states, 0.0, 1.0).astype(
                np.float32, copy=False
            )

        # 5. Generate output bitstreams
        receptor_activity = np.mean(self.receptor_states, axis=1)
        output_probs = np.clip(receptor_activity, 0.0, 1.0)
        rands = self._rng.random(
            (self.params.n_neurotransmitter_types, self.params.bitstream_length)
        )
        output_bitstreams = (rands < output_probs[:, None]).astype(np.uint8)
        genomic_drive = self.params.genomic_coupling * self.second_messenger_levels

        # Store history
        self.history.append(
            {
                "nt_concentrations": self.nt_concentrations.copy(),
                "receptor_activity": receptor_activity.copy(),
                "second_messengers": self.second_messenger_levels.copy(),
            }
        )
        if len(self.history) > 100:
            self.history.pop(0)

        return {
            "receptor_activity": receptor_activity,
            "second_messengers": self.second_messenger_levels.copy(),
            "genomic_drive": genomic_drive.copy(),
            "output_bitstreams": output_bitstreams,
            "nt_concentrations": self.nt_concentrations.copy(),
        }

    def release_neurotransmitter(self, nt_type: int, amount: float) -> None:
        """Trigger neurotransmitter release."""
        if not isinstance(nt_type, int) or isinstance(nt_type, bool):
            raise ValueError("nt_type must be a valid neurotransmitter index")
        if nt_type < 0 or nt_type >= self.params.n_neurotransmitter_types:
            raise ValueError("nt_type must be a valid neurotransmitter index")
        if not math.isfinite(float(amount)) or amount < 0.0:
            raise ValueError("amount must be finite and non-negative")
        self.nt_concentrations[nt_type] = np.clip(
            self.nt_concentrations[nt_type] + amount, 0.0, 1.0
        )

    def get_global_metric(self) -> float:
        """Return the global neurochemical activity metric."""
        return float(np.mean(self.receptor_states))

    def get_neuromodulation_state(self) -> Dict[str, float]:
        """Return named neurotransmitter levels for external use."""
        return {
            "dopamine": float(self.nt_concentrations[self.DA]),
            "serotonin": float(self.nt_concentrations[self.SEROTONIN]),
            "norepinephrine": float(self.nt_concentrations[self.NE]),
            "acetylcholine": float(self.nt_concentrations[self.ACH]),
        }

    @staticmethod
    def _validate_params(params: L2_StochasticParameters) -> None:
        if (
            not isinstance(params.n_receptors, int)
            or isinstance(params.n_receptors, bool)
            or params.n_receptors <= 0
        ):
            raise ValueError("n_receptors must be a positive integer")
        if (
            not isinstance(params.n_neurotransmitter_types, int)
            or isinstance(params.n_neurotransmitter_types, bool)
            or params.n_neurotransmitter_types <= 0
        ):
            raise ValueError("n_neurotransmitter_types must be a positive integer")
        if (
            not isinstance(params.bitstream_length, int)
            or isinstance(params.bitstream_length, bool)
            or params.bitstream_length <= 0
        ):
            raise ValueError("bitstream_length must be a positive integer")
        if (
            not math.isfinite(float(params.binding_affinity))
            or params.binding_affinity < 0.0
            or params.binding_affinity > 1.0
        ):
            raise ValueError("binding_affinity must be finite and within [0, 1]")
        for field_name in (
            "unbinding_rate",
            "diffusion_rate",
            "reuptake_rate",
            "quantum_coupling",
            "genomic_coupling",
        ):
            value = float(getattr(params, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if params.rng_seed is not None:
            if isinstance(params.rng_seed, bool) or not isinstance(params.rng_seed, int):
                raise ValueError("rng_seed must be a non-negative integer or None")
            if params.rng_seed < 0:
                raise ValueError("rng_seed must be a non-negative integer or None")

    @classmethod
    def _validate_step_inputs(
        cls,
        dt: float,
        nt_release: Optional[np.ndarray[Any, Any]],
        l1_input: Optional[np.ndarray[Any, Any]],
        n_neurotransmitter_types: int,
    ) -> None:
        if not math.isfinite(float(dt)) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if nt_release is not None:
            release = np.asarray(nt_release, dtype=np.float64)
            if release.size != n_neurotransmitter_types or not np.all(np.isfinite(release)):
                raise ValueError("nt_release must contain one finite value per neurotransmitter")
            if np.any(release < 0.0) or np.any(release > 1.0):
                raise ValueError("nt_release must be within [0, 1]")
        if l1_input is not None:
            cls._finite_mean(l1_input, "l1_input")

    @staticmethod
    def _finite_mean(values: Any, name: str) -> float:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain finite values")
        return float(np.mean(arr))
