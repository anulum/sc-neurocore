# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L5 Organismal-Psychoemotional Layer (Stochastic

from typing import Any, Optional

"""
SCPN L5: Organismal-Psychoemotional Layer (Stochastic Implementation)
======================================================================

Implements Layer 5 of the SCPN framework: Organismal integration and
psychoemotional states including HRV, autonomic nervous system,
and emotional valence dynamics.

Key Features:
- Stochastic autonomic nervous system modeling
- Heart Rate Variability (HRV) dynamics
- Emotional state as attractor dynamics
- Interoceptive inference integration

"""

from dataclasses import dataclass
import numpy as np
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class L5_StochasticParameters:
    """Parameters for the Stochastic L5 Organismal Layer."""

    n_emotional_dims: int = 8  # Valence, arousal, dominance, etc.
    n_autonomic_nodes: int = 100
    bitstream_length: int = 1024

    # Autonomic dynamics
    sympathetic_baseline: float = 0.4
    parasympathetic_baseline: float = 0.6
    autonomic_time_constant: float = 5.0  # seconds

    # HRV parameters
    base_heart_rate: float = 70.0  # BPM
    hrv_amplitude: float = 0.1
    respiratory_frequency: float = 0.25  # Hz

    # Emotional dynamics
    emotional_decay: float = 0.1
    emotional_noise: float = 0.05
    attractor_strength: float = 0.3

    # Inter-layer coupling
    cellular_coupling: float = 0.15  # From L4
    ecological_coupling: float = 0.1  # To L6


class L5_OrganismalLayer:
    """
    Stochastic implementation of the Organismal-Psychoemotional Layer.

    Models whole-organism integration, autonomic regulation, and
    emotional dynamics using bitstream representations.
    """

    # Emotional dimension indices
    VALENCE = 0  # Pleasant-Unpleasant
    AROUSAL = 1  # Activated-Deactivated
    DOMINANCE = 2  # Dominant-Submissive
    APPROACH = 3  # Approach-Avoid
    CERTAINTY = 4  # Certain-Uncertain
    ATTENTION = 5  # Focused-Diffuse
    FAIRNESS = 6  # Fair-Unfair
    SAFETY = 7  # Safe-Threatened

    def __init__(self, params: Optional[L5_StochasticParameters] = None):
        self.params = params or L5_StochasticParameters()

        # Emotional state vector
        self.emotional_state = np.zeros(self.params.n_emotional_dims)
        self.emotional_state[self.VALENCE] = 0.5  # Neutral valence
        self.emotional_state[self.AROUSAL] = 0.3  # Low arousal baseline
        self.emotional_state[self.SAFETY] = 0.7  # Safe baseline

        # Autonomic nervous system state
        self.sympathetic = self.params.sympathetic_baseline
        self.parasympathetic = self.params.parasympathetic_baseline

        # Heart dynamics
        self.heart_rate = self.params.base_heart_rate
        self.hrv_phase = 0.0
        self.rr_intervals: List[float] = []

        # Interoceptive state (body sense)
        self.interoceptive_state = np.random.random(self.params.n_autonomic_nodes) * 0.3

        # Attractor basins for emotional states
        self.attractors = self._init_emotional_attractors()

        # Time tracking
        self.time = 0.0

    def _init_emotional_attractors(self) -> np.ndarray[Any, Any]:
        """Initialize emotional attractor states."""
        # Define stable emotional configurations
        attractors = np.array(
            [
                [0.8, 0.3, 0.6, 0.7, 0.7, 0.5, 0.6, 0.8],  # Joy/contentment
                [0.2, 0.8, 0.3, 0.2, 0.3, 0.8, 0.3, 0.2],  # Fear/anxiety
                [0.2, 0.7, 0.7, 0.8, 0.6, 0.7, 0.2, 0.4],  # Anger
                [0.3, 0.2, 0.2, 0.2, 0.4, 0.3, 0.5, 0.5],  # Sadness
                [0.5, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6],  # Neutral
            ]
        )
        return attractors

    def step(
        self,
        dt: float,
        l4_input: Optional[Dict[str, Any]] = None,
        external_event: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, np.ndarray[Any, Any]]:
        """
        Advance the layer by one time step.

        Args:
            dt: Time step in seconds.
            l4_input: Cellular layer output (synchronization).
            external_event: External emotional trigger {valence, arousal, ...}.

        Returns:
            Dict with emotional_state, autonomic, heart_rate, output_bitstreams
        """
        self.time += dt

        # 1. Process external emotional events
        if external_event is not None:
            for dim, value in external_event.items():
                if isinstance(dim, int) and 0 <= dim < self.params.n_emotional_dims:
                    self.emotional_state[dim] += value * 0.3

        # 2. Attractor dynamics (emotional states converge to stable patterns)
        # Find nearest attractor
        distances = np.linalg.norm(self.attractors - self.emotional_state, axis=1)
        nearest_attractor = self.attractors[np.argmin(distances)]

        # Pull toward attractor
        self.emotional_state += (
            self.params.attractor_strength * (nearest_attractor - self.emotional_state) * dt
        )

        # Add noise
        self.emotional_state += (
            self.params.emotional_noise * np.random.normal(0, 1, self.params.n_emotional_dims) * dt
        )

        # Decay toward baseline
        baseline = np.array([0.5, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5, 0.6])
        self.emotional_state += self.params.emotional_decay * (baseline - self.emotional_state) * dt

        self.emotional_state = np.clip(self.emotional_state, 0.0, 1.0)

        # 3. Autonomic nervous system dynamics
        # Sympathetic driven by arousal and threat
        target_symp = (
            self.emotional_state[self.AROUSAL] * 0.5 + (1 - self.emotional_state[self.SAFETY]) * 0.5
        )
        # Parasympathetic driven by valence and safety
        target_para = (
            self.emotional_state[self.VALENCE] * 0.3 + self.emotional_state[self.SAFETY] * 0.7
        )

        tau = self.params.autonomic_time_constant
        self.sympathetic += (target_symp - self.sympathetic) * dt / tau
        self.parasympathetic += (target_para - self.parasympathetic) * dt / tau

        self.sympathetic = np.clip(self.sympathetic, 0.0, 1.0)
        self.parasympathetic = np.clip(self.parasympathetic, 0.0, 1.0)

        # 4. Heart rate and HRV
        # RSA (Respiratory Sinus Arrhythmia)
        self.hrv_phase += 2 * np.pi * self.params.respiratory_frequency * dt
        rsa_component = self.params.hrv_amplitude * np.sin(self.hrv_phase) * self.parasympathetic

        # Sympathetic raises HR, parasympathetic lowers it
        target_hr = self.params.base_heart_rate + 20 * self.sympathetic - 15 * self.parasympathetic
        self.heart_rate += (target_hr - self.heart_rate) * dt * 0.5
        self.heart_rate += rsa_component * 10  # RSA effect

        # Track RR intervals
        rr = 60000.0 / self.heart_rate  # ms
        self.rr_intervals.append(rr)
        if len(self.rr_intervals) > 100:
            self.rr_intervals.pop(0)

        # 5. Cellular input coupling (L4 synchronization affects coherence)
        if l4_input is not None and "synchronization" in l4_input:
            sync = l4_input["synchronization"]
            # High cellular sync improves emotional stability
            self.emotional_state[self.CERTAINTY] += sync * self.params.cellular_coupling * dt
            self.emotional_state = np.clip(self.emotional_state, 0.0, 1.0)

        # 6. Update interoceptive state
        self.interoceptive_state = (
            0.8 * self.interoceptive_state
            + 0.2
            * np.tile(
                [self.sympathetic, self.parasympathetic, self.heart_rate / 100],
                self.params.n_autonomic_nodes // 3 + 1,
            )[: self.params.n_autonomic_nodes]
        )

        # 7. Generate output bitstreams
        output_probs = np.concatenate(
            [self.emotional_state, [self.sympathetic, self.parasympathetic, self.heart_rate / 100]]
        )
        output_probs = np.tile(output_probs, self.params.n_autonomic_nodes // len(output_probs) + 1)
        output_probs = output_probs[: self.params.n_autonomic_nodes]

        rands = np.random.random((self.params.n_autonomic_nodes, self.params.bitstream_length))
        output_bitstreams = (rands < output_probs[:, None]).astype(np.uint8)

        return {
            "emotional_state": self.emotional_state.copy(),
            "sympathetic": self.sympathetic,  # type: ignore
            "parasympathetic": self.parasympathetic,  # type: ignore
            "heart_rate": self.heart_rate,  # type: ignore
            "hrv_rmssd": self._compute_rmssd(),  # type: ignore
            "interoceptive_state": self.interoceptive_state.copy(),
            "output_bitstreams": output_bitstreams,
        }

    def _compute_rmssd(self) -> float:
        """Compute RMSSD (root mean square of successive differences)."""
        if len(self.rr_intervals) < 2:
            return 0.0
        rr = np.array(self.rr_intervals)
        diff = np.diff(rr)
        return float(np.sqrt(np.mean(diff**2)))

    def get_global_metric(self) -> float:
        """Return the global organismal coherence metric."""
        # Combine HRV coherence with emotional stability
        hrv_coherence = self._compute_rmssd() / 100  # Normalize
        emotional_stability = 1.0 - np.std(self.emotional_state)
        return float(0.5 * hrv_coherence + 0.5 * emotional_stability)

    def get_emotional_valence(self) -> float:
        """Return current emotional valence."""
        return float(self.emotional_state[self.VALENCE])
