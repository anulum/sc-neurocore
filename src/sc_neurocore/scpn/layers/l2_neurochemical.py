# SPDX-License-Identifier: AGPL-3.0-or-later
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

from dataclasses import dataclass
import numpy as np
import logging
from typing import Dict

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
    ) -> Dict[str, np.ndarray[Any, Any]]:
        """
        Advance the layer by one time step.

        Args:
            dt: Time step in seconds.
            nt_release: Neurotransmitter release rates [4] (0-1 normalized).
            l1_input: Quantum layer input (coherence modulation).

        Returns:
            Dict with receptor_activity, second_messengers, output_bitstreams
        """
        # 1. Update neurotransmitter concentrations from release
        if nt_release is not None:
            self.nt_concentrations = np.clip(
                self.nt_concentrations + nt_release * dt - self.params.reuptake_rate * dt, 0.0, 1.0
            )

        # 2. Receptor binding dynamics (stochastic)
        for nt_idx in range(self.params.n_neurotransmitter_types):
            nt_conc = self.nt_concentrations[nt_idx]

            # Binding: P(bind) = affinity * concentration * (1 - current_state)
            binding_prob = self.params.binding_affinity * nt_conc
            bind_mask = np.random.random(self.params.n_receptors) < binding_prob * dt

            # Unbinding: P(unbind) = unbinding_rate * current_state
            unbind_mask = (
                np.random.random(self.params.n_receptors) < self.params.unbinding_rate * dt
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
            quantum_mod = np.mean(l1_input) * self.params.quantum_coupling
            self.receptor_states *= 1.0 + quantum_mod
            self.receptor_states = np.clip(self.receptor_states, 0.0, 1.0)

        # 5. Generate output bitstreams
        output_probs = receptor_activity
        rands = np.random.random(
            (self.params.n_neurotransmitter_types, self.params.bitstream_length)
        )
        output_bitstreams = (rands < output_probs[:, None]).astype(np.uint8)

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
            "output_bitstreams": output_bitstreams,
            "nt_concentrations": self.nt_concentrations.copy(),
        }

    def release_neurotransmitter(self, nt_type: int, amount: float) -> None:
        """Trigger neurotransmitter release."""
        if 0 <= nt_type < self.params.n_neurotransmitter_types:
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
