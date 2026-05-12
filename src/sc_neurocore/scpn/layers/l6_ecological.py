# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L6 Ecological-Planetary Layer (Stochastic

from typing import Any, Optional

"""
SCPN L6: Ecological-Planetary Layer (Stochastic Implementation)
================================================================

Implements Layer 6 of the SCPN framework: Ecological and planetary-scale
dynamics including Schumann resonances, geomagnetic coupling, and
biospheric network effects.

Key Features:
- Stochastic Schumann resonance simulation
- Geomagnetic field coupling
- Circadian rhythm integration
- Planetary consciousness field modeling

"""

import logging
import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class L6_StochasticParameters:
    """Parameters for the Stochastic L6 Ecological Layer."""

    n_field_nodes: int = 256
    bitstream_length: int = 1024

    # Schumann resonance parameters (Hz)
    schumann_frequencies: tuple[Any, ...] = (7.83, 14.3, 20.8, 27.3, 33.8)
    schumann_amplitude: float = 0.5
    schumann_noise: float = 0.1

    # Geomagnetic parameters
    geomag_baseline: float = 50.0  # μT (Earth's field)
    geomag_variation: float = 0.1

    # Circadian parameters
    circadian_period: float = 24.0 * 3600  # seconds
    circadian_amplitude: float = 0.3

    # Biospheric network
    network_coupling: float = 0.2
    network_noise: float = 0.05

    # Inter-layer coupling
    organismal_coupling: float = 0.15  # From L5
    symbolic_coupling: float = 0.1  # To L7
    rng_seed: Optional[int] = None


class L6_EcologicalLayer:
    """
    Stochastic implementation of the Ecological-Planetary Layer.

    Models planetary-scale electromagnetic fields, Schumann resonances,
    and biospheric network dynamics using bitstream representations.
    """

    def __init__(self, params: Optional[L6_StochasticParameters] = None):
        self.params = params or L6_StochasticParameters()
        self._validate_params(self.params)
        self._rng = np.random.default_rng(self.params.rng_seed)

        # Schumann resonance field (superposition of modes)
        self.schumann_phases = np.zeros(len(self.params.schumann_frequencies))
        self.schumann_amplitudes = np.ones(len(self.params.schumann_frequencies))

        # Geomagnetic field state
        self.geomag_field = np.ones(self.params.n_field_nodes) * self.params.geomag_baseline

        # Circadian phase
        self.circadian_phase = 0.0

        # Biospheric network state (collective field)
        self.biospheric_field = self._rng.random(self.params.n_field_nodes) * 0.3

        # Planetary consciousness coherence
        self.planetary_coherence = 0.5

        # History for temporal patterns
        self.history: List[Dict[str, Any]] = []

        # Time tracking
        self.time = 0.0

    def step(
        self,
        dt: float,
        l5_input: Optional[Dict[str, Any]] = None,
        solar_activity: float = 0.5,
        lunar_phase: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Advance the layer by one time step.

        Args:
            dt: Time step in seconds.
            l5_input: Organismal layer output (emotional coherence).
            solar_activity: Solar activity index (0-1).
            lunar_phase: Lunar phase (0 to 2π).

        Returns:
            Dict with schumann_field, geomag, circadian, output_bitstreams
        """
        self._validate_step_inputs(dt, l5_input, solar_activity, lunar_phase)
        self.time += dt

        # 1. Schumann resonance dynamics
        for i, freq in enumerate(self.params.schumann_frequencies):
            self.schumann_phases[i] += 2 * np.pi * freq * dt
            self.schumann_phases[i] = self.schumann_phases[i] % (2 * np.pi)

        # Compute Schumann field as superposition
        schumann_signal = np.zeros(self.params.n_field_nodes)
        for i, freq in enumerate(self.params.schumann_frequencies):
            spatial_pattern = np.sin(np.linspace(0, 2 * np.pi * (i + 1), self.params.n_field_nodes))
            schumann_signal += (
                self.schumann_amplitudes[i]
                * self.params.schumann_amplitude
                * np.cos(self.schumann_phases[i])
                * spatial_pattern
            )

        # Add noise
        schumann_signal += self.params.schumann_noise * self._rng.normal(
            0, 1, self.params.n_field_nodes
        )

        # Normalize to [0, 1]
        schumann_field = (schumann_signal - schumann_signal.min()) / (
            schumann_signal.max() - schumann_signal.min() + 1e-8
        )

        # 2. Geomagnetic field dynamics
        # Solar activity modulates geomagnetic storms
        storm_factor = 1.0 + 0.5 * (solar_activity - 0.5)
        geomag_variation = (
            self.params.geomag_variation
            * storm_factor
            * self._rng.normal(0, 1, self.params.n_field_nodes)
        )
        self.geomag_field = np.clip(
            self.geomag_field + geomag_variation * dt,
            self.params.geomag_baseline * 0.5,
            self.params.geomag_baseline * 1.5,
        )

        # 3. Circadian rhythm
        self.circadian_phase += 2 * np.pi * dt / self.params.circadian_period
        self.circadian_phase = self.circadian_phase % (2 * np.pi)
        circadian_signal = 0.5 + self.params.circadian_amplitude * np.cos(self.circadian_phase)

        # 4. Biospheric network dynamics
        # Coupling between nodes
        network_coupling = np.zeros(self.params.n_field_nodes)
        for i in range(self.params.n_field_nodes):
            neighbors = [(i - 1) % self.params.n_field_nodes, (i + 1) % self.params.n_field_nodes]
            neighbor_mean = np.mean([self.biospheric_field[j] for j in neighbors])
            network_coupling[i] = neighbor_mean - self.biospheric_field[i]

        self.biospheric_field += (
            self.params.network_coupling * network_coupling
            + self.params.network_noise * self._rng.normal(0, 1, self.params.n_field_nodes)
        ) * dt

        # Modulate by Schumann and circadian
        self.biospheric_field *= (0.9 + 0.1 * schumann_field) * (0.8 + 0.2 * circadian_signal)
        self.biospheric_field = np.clip(self.biospheric_field, 0.0, 1.0)

        # 5. Organismal coupling (L5 collective emotional state affects field)
        if l5_input is not None:
            l5_effect = self._l5_organismal_effect(l5_input)
            if l5_effect != 0.0:
                self.biospheric_field += self.params.organismal_coupling * l5_effect * dt
                self.biospheric_field = np.clip(self.biospheric_field, 0.0, 1.0)

        # 6. Lunar phase modulation
        lunar_factor = 0.5 + 0.5 * np.cos(lunar_phase)
        self.schumann_amplitudes = np.ones(len(self.params.schumann_frequencies)) * (
            0.8 + 0.2 * lunar_factor
        )

        # 7. Compute planetary coherence
        self.planetary_coherence = float(
            np.abs(np.mean(np.exp(1j * 2 * np.pi * self.biospheric_field)))
        )

        # 8. Generate output bitstreams
        output_probs = self.biospheric_field * circadian_signal
        rands = self._rng.random((self.params.n_field_nodes, self.params.bitstream_length))
        output_bitstreams = (rands < output_probs[:, None]).astype(np.uint8)
        symbolic_drive = self.params.symbolic_coupling * schumann_field

        # Store history
        result = {
            "schumann_field": schumann_field,
            "schumann_phases": self.schumann_phases.copy(),
            "geomag_field": self.geomag_field.copy(),
            "circadian_phase": self.circadian_phase,
            "circadian_signal": circadian_signal,
            "biospheric_field": self.biospheric_field.copy(),
            "planetary_coherence": self.planetary_coherence,
            "symbolic_drive": symbolic_drive.copy(),
            "output_bitstreams": output_bitstreams,
        }

        self.history.append(
            {
                "time": self.time,
                "coherence": self.planetary_coherence,
                "schumann_power": float(np.mean(schumann_field**2)),
            }
        )
        if len(self.history) > 100:
            self.history.pop(0)

        return result

    def get_global_metric(self) -> float:
        """Return the global planetary coherence metric."""
        return self.planetary_coherence

    def get_schumann_spectrum(self) -> Dict[float, float]:
        """Return current Schumann resonance spectrum."""
        return {
            freq: float(amp * np.cos(phase))
            for freq, amp, phase in zip(
                self.params.schumann_frequencies, self.schumann_amplitudes, self.schumann_phases
            )
        }

    def get_circadian_time(self) -> float:
        """Return current circadian time (0-24 hours)."""
        return (self.circadian_phase / (2 * np.pi)) * 24.0

    @staticmethod
    def _validate_params(params: L6_StochasticParameters) -> None:
        if (
            not isinstance(params.n_field_nodes, int)
            or isinstance(params.n_field_nodes, bool)
            or params.n_field_nodes <= 0
        ):
            raise ValueError("n_field_nodes must be a positive integer")
        if (
            not isinstance(params.bitstream_length, int)
            or isinstance(params.bitstream_length, bool)
            or params.bitstream_length <= 0
        ):
            raise ValueError("bitstream_length must be a positive integer")
        if (
            not isinstance(params.schumann_frequencies, tuple)
            or len(params.schumann_frequencies) == 0
            or any(not math.isfinite(float(freq)) or float(freq) <= 0.0 for freq in params.schumann_frequencies)
        ):
            raise ValueError("schumann_frequencies must be a non-empty tuple of positive finite values")
        for field_name in (
            "schumann_amplitude",
            "schumann_noise",
            "geomag_variation",
            "circadian_amplitude",
            "network_coupling",
            "network_noise",
            "organismal_coupling",
            "symbolic_coupling",
        ):
            value = float(getattr(params, field_name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and non-negative")
        if not math.isfinite(float(params.geomag_baseline)) or params.geomag_baseline <= 0.0:
            raise ValueError("geomag_baseline must be finite and positive")
        if not math.isfinite(float(params.circadian_period)) or params.circadian_period <= 0.0:
            raise ValueError("circadian_period must be finite and positive")
        if params.circadian_amplitude > 0.5:
            raise ValueError("circadian_amplitude must keep circadian output within [0, 1]")
        if params.rng_seed is not None and (
            not isinstance(params.rng_seed, int)
            or isinstance(params.rng_seed, bool)
            or params.rng_seed < 0
        ):
            raise ValueError("rng_seed must be None or a non-negative integer")

    @classmethod
    def _validate_step_inputs(
        cls,
        dt: float,
        l5_input: Optional[Dict[str, Any]],
        solar_activity: float,
        lunar_phase: float,
    ) -> None:
        if not math.isfinite(float(dt)) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        if (
            not math.isfinite(float(solar_activity))
            or solar_activity < 0.0
            or solar_activity > 1.0
        ):
            raise ValueError("solar_activity must be finite and within [0, 1]")
        if not math.isfinite(float(lunar_phase)):
            raise ValueError("lunar_phase must be finite")
        if l5_input is not None:
            cls._l5_organismal_effect(l5_input)

    @staticmethod
    def _finite_mean(values: Any, name: str) -> float:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain finite values")
        return float(np.mean(arr))

    @classmethod
    def _l5_organismal_effect(cls, l5_input: Dict[str, Any]) -> float:
        if "ecological_drive" in l5_input:
            return cls._unit_mean(l5_input["ecological_drive"], "ecological_drive")
        if "emotional_state" in l5_input:
            emotional_coherence = cls._finite_mean(l5_input["emotional_state"], "emotional_state")
            return emotional_coherence - 0.5
        return 0.0

    @classmethod
    def _unit_mean(cls, values: Any, name: str) -> float:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain finite values")
        if np.any(arr < 0.0) or np.any(arr > 1.0):
            raise ValueError(f"{name} values must be within [0, 1]")
        return float(np.mean(arr))
