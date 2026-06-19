# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ↔ CCW/VIBRANA bridge

"""SC-NeuroCore ↔ CCW/VIBRANA bridge.

Converts stochastic bitstream outputs to audio parameters and
visualization states for the CCW application.
"""

from typing import Any, Dict, List, Optional, Tuple

import json
import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)

# Golden ratio
PHI = (1 + np.sqrt(5)) / 2


class CCWMode(str, Enum):
    """CCW modulation modes aligned with VIBRANA."""

    THEURGIC = "theurgic"
    HEALING = "healing"
    MEDITATION = "meditation"
    COSMIC = "cosmic"
    FOCUS = "focus"
    CREATIVITY = "creativity"


@dataclass
class CCWParameters:
    """Parameters for CCW audio generation."""

    base_frequency: float = 7.83  # Schumann resonance
    carrier_frequency: float = 432.0  # Verdi tuning (A4=432 Hz)
    binaural_offset: float = 10.0  # Hz
    modulation_depth: float = 0.5
    sample_rate: int = 44100


@dataclass
class VIBRANAState:
    """State for VIBRANA visualization sync."""

    mode: CCWMode = CCWMode.MEDITATION
    geometry_phase: float = 0.0
    color_intensity: float = 0.5
    rotation_speed: float = 1.0
    glyph_weights: np.ndarray[Any, Any] = field(default_factory=lambda: np.zeros(6))


class CCWBridge:
    """
    Bridge between SC-NeuroCore and CCW/VIBRANA systems.

    Converts bitstream outputs from SCPN layers into audio parameters
    and visualization states for the CCW application.
    """

    # SCPN metric to CCW parameter mappings
    METRIC_MAPPINGS = {
        "l1_quantum_coherence": ("modulation_depth", 0.3, 0.8),
        "l2_neurochemical_activity": ("carrier_blend", 0.0, 1.0),
        "l4_cellular_sync": ("binaural_offset", 4.0, 40.0),
        "l5_organismal_coherence": ("amplitude", 0.3, 1.0),
        "l6_planetary_coherence": ("schumann_blend", 0.0, 1.0),
        "l7_symbolic_health": ("sacred_geometry_intensity", 0.0, 1.0),
    }

    # Mode to frequency mapping (aligned with VIBRANA)
    MODE_FREQUENCIES = {
        CCWMode.THEURGIC: (7.83, 14.3),  # Schumann
        CCWMode.HEALING: (528.0, 432.0),  # Solfeggio
        CCWMode.MEDITATION: (4.0, 7.83),  # Theta-Schumann
        CCWMode.COSMIC: (136.1, 272.2),  # OM
        CCWMode.FOCUS: (14.0, 18.0),  # Beta
        CCWMode.CREATIVITY: (10.0, 12.0),  # Alpha
    }

    def __init__(self, params: Optional[CCWParameters] = None):
        self.params = params or CCWParameters()
        self.vibrana_state = VIBRANAState()

        # Audio generation state
        self.phase_left = 0.0
        self.phase_right = 0.0
        self.modulation_phase = 0.0

        # History for smoothing
        self.metric_history: Dict[str, List[float]] = {}
        self.smoothing_window = 10

    def bitstream_to_frequency(
        self, bitstream: np.ndarray[Any, Any], freq_min: float = 1.0, freq_max: float = 40.0
    ) -> float:
        """
        Convert a bitstream to a frequency value.

        Args:
            bitstream: Binary array from SC layer output
            freq_min: Minimum frequency (Hz)
            freq_max: Maximum frequency (Hz)

        Returns:
            Frequency in Hz mapped from bitstream probability
        """
        prob = np.mean(bitstream)
        return float(freq_min + prob * (freq_max - freq_min))

    def scpn_metrics_to_ccw(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Convert SCPN global metrics to CCW audio parameters.

        Args:
            metrics: Dict from get_global_metrics() of SCPN layers

        Returns:
            Dict of CCW-compatible audio parameters
        """
        ccw_params = {
            "base_frequency": self.params.base_frequency,
            "carrier_frequency": self.params.carrier_frequency,
            "binaural_offset": self.params.binaural_offset,
            "modulation_depth": self.params.modulation_depth,
            "amplitude": 0.5,
            "carrier_blend": 0.5,
            "schumann_blend": 0.5,
            "sacred_geometry_intensity": 0.5,
        }

        for metric_name, (param_name, min_val, max_val) in self.METRIC_MAPPINGS.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                # Smooth the value
                if metric_name not in self.metric_history:
                    self.metric_history[metric_name] = []
                self.metric_history[metric_name].append(value)
                if len(self.metric_history[metric_name]) > self.smoothing_window:
                    self.metric_history[metric_name].pop(0)
                smoothed = np.mean(self.metric_history[metric_name])

                # Map to parameter range
                ccw_params[param_name] = min_val + smoothed * (max_val - min_val)  # type: ignore[assignment]

        return ccw_params

    def glyph_vector_to_vibrana(self, glyph_vector: np.ndarray[Any, Any]) -> Dict[str, Any]:
        """
        Convert L7 glyph vector to VIBRANA visualization parameters.

        Args:
            glyph_vector: 6D vector [phi, fib, metatron, platonic, e8, health]

        Returns:
            Dict of VIBRANA visualization parameters
        """
        if len(glyph_vector) < 6:
            glyph_vector = np.pad(glyph_vector, (0, 6 - len(glyph_vector)))

        self.vibrana_state.glyph_weights = glyph_vector

        # Map glyph components to visualization
        phi_alignment = glyph_vector[0]
        fibonacci_alignment = glyph_vector[1]
        metatron_flow = glyph_vector[2]
        platonic_coherence = glyph_vector[3]
        e8_alignment = glyph_vector[4]
        symbolic_health = glyph_vector[5]

        # Determine best mode based on glyph pattern
        if metatron_flow > 0.7:
            self.vibrana_state.mode = CCWMode.THEURGIC
        elif phi_alignment > 0.8 and fibonacci_alignment > 0.8:
            self.vibrana_state.mode = CCWMode.COSMIC
        elif symbolic_health > 0.6:
            self.vibrana_state.mode = CCWMode.HEALING
        elif e8_alignment > 0.7:
            self.vibrana_state.mode = CCWMode.MEDITATION
        else:
            self.vibrana_state.mode = CCWMode.FOCUS

        # Set visualization parameters
        self.vibrana_state.color_intensity = symbolic_health
        self.vibrana_state.rotation_speed = 0.5 + metatron_flow * 2.0
        self.vibrana_state.geometry_phase += platonic_coherence * 0.1

        return {
            "mode": self.vibrana_state.mode.value,
            "geometry_phase": float(self.vibrana_state.geometry_phase % (2 * np.pi)),
            "color_intensity": float(self.vibrana_state.color_intensity),
            "rotation_speed": float(self.vibrana_state.rotation_speed),
            "glyph_weights": {
                "phi_alignment": float(phi_alignment),
                "fibonacci_alignment": float(fibonacci_alignment),
                "metatron_flow": float(metatron_flow),
                "platonic_coherence": float(platonic_coherence),
                "e8_alignment": float(e8_alignment),
                "symbolic_health": float(symbolic_health),
            },
            "frequencies": {
                "base": self.MODE_FREQUENCIES[self.vibrana_state.mode][0],
                "harmonic": self.MODE_FREQUENCIES[self.vibrana_state.mode][1],
            },
        }

    def generate_binaural_sample(
        self, ccw_params: Dict[str, float], duration_samples: int = 1024
    ) -> Tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """
        Generate binaural audio samples from CCW parameters.

        Args:
            ccw_params: Parameters from scpn_metrics_to_ccw()
            duration_samples: Number of samples to generate

        Returns:
            Tuple of (left_channel, right_channel) numpy arrays
        """
        sample_rate = self.params.sample_rate
        dt = 1.0 / sample_rate

        # Extract parameters
        carrier = ccw_params.get("carrier_frequency", 432.0)
        binaural = ccw_params.get("binaural_offset", 10.0)
        mod_depth = ccw_params.get("modulation_depth", 0.5)
        amplitude = ccw_params.get("amplitude", 0.5)
        base_freq = ccw_params.get("base_frequency", 7.83)

        # Time array
        t = np.arange(duration_samples) * dt

        # Generate binaural beat (carrier + offset for right channel)
        left_freq = carrier
        right_freq = carrier + binaural

        # Phase-continuous generation
        phase_increment_left = 2 * np.pi * left_freq * dt
        phase_increment_right = 2 * np.pi * right_freq * dt

        phases_left = self.phase_left + np.cumsum(np.ones(duration_samples) * phase_increment_left)
        phases_right = self.phase_right + np.cumsum(
            np.ones(duration_samples) * phase_increment_right
        )

        # Update phase state for continuity
        self.phase_left = phases_left[-1] % (2 * np.pi)
        self.phase_right = phases_right[-1] % (2 * np.pi)

        # Generate carriers
        left = np.sin(phases_left)
        right = np.sin(phases_right)

        # Add modulation envelope (low frequency oscillation)
        mod_phases = self.modulation_phase + np.cumsum(
            np.ones(duration_samples) * 2 * np.pi * base_freq * dt
        )
        self.modulation_phase = mod_phases[-1] % (2 * np.pi)

        modulation = 1.0 - mod_depth * (1 + np.sin(mod_phases)) / 2

        # Apply modulation and amplitude
        left = amplitude * left * modulation
        right = amplitude * right * modulation

        return left, right

    def generate_ccw_metadata(
        self, scpn_outputs: Dict[str, Any], glyph_vector: Optional[np.ndarray[Any, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate complete CCW metadata package for audio/visual sync.

        Args:
            scpn_outputs: Full output dict from run_integrated_step()
            glyph_vector: Optional L7 glyph vector

        Returns:
            Complete metadata dict for CCW system
        """
        # Extract metrics
        metrics = {}
        for layer_name, output in scpn_outputs.items():
            if isinstance(output, dict):
                if "coherence" in str(output.keys()).lower():
                    for k, v in output.items():
                        if isinstance(v, (int, float)):
                            metrics[f"{layer_name}_{k}"] = float(v)

        # Get glyph vector from L7 if not provided
        if glyph_vector is None and "l7" in scpn_outputs:
            l7_out = scpn_outputs["l7"]
            if isinstance(l7_out, dict) and "glyph_vector" in l7_out:
                glyph_vector = l7_out["glyph_vector"]

        # Convert to CCW parameters
        ccw_params = self.scpn_metrics_to_ccw(metrics)

        # Convert glyph to VIBRANA
        vibrana_params = {}
        if glyph_vector is not None:
            vibrana_params = self.glyph_vector_to_vibrana(glyph_vector)

        # Build complete metadata
        metadata = {
            "timestamp": float(np.datetime64("now").astype(np.float64)),
            "ccw_audio": ccw_params,
            "vibrana_visual": vibrana_params,
            "scpn_metrics": metrics,
            "mode": self.vibrana_state.mode.value,
            "bridge_version": "1.0.0",
        }

        return metadata

    def export_glyph_stream(
        self,
        glyph_vector: np.ndarray[Any, Any],
        cosmic_vector: Optional[Dict[str, float]] = None,
        filepath: Optional[str] = None,
    ) -> str:
        """
        Export glyph stream data for VIBRANA/CCW hardware playback.

        Args:
            glyph_vector: Normalized glyph vector from L7
            cosmic_vector: Optional L8 cosmic phase data
            filepath: Optional file path to save

        Returns:
            JSON string of glyph stream data
        """
        stream_data = {
            "glyph_vector": {
                "phi_alignment": float(glyph_vector[0]) if len(glyph_vector) > 0 else 0.0,
                "fibonacci_alignment": float(glyph_vector[1]) if len(glyph_vector) > 1 else 0.0,
                "metatron_flow": float(glyph_vector[2]) if len(glyph_vector) > 2 else 0.0,
                "platonic_coherence": float(glyph_vector[3]) if len(glyph_vector) > 3 else 0.0,
                "e8_alignment": float(glyph_vector[4]) if len(glyph_vector) > 4 else 0.0,
                "symbolic_health": float(glyph_vector[5]) if len(glyph_vector) > 5 else 0.0,
            },
            "cosmic_vector": cosmic_vector or {},
            "layer_weights": {
                "metatron_weight": 0.95,  # Default high weight for Metatron
                "phi_weight": 0.85,
                "e8_weight": 0.75,
            },
            "routing": {
                "target": "vibrana_hardware",
                "protocol": "bitstream",
                "encoding": "normalized_float",
            },
        }

        json_str = json.dumps(stream_data, indent=2)

        if filepath:
            with open(filepath, "w") as f:
                f.write(json_str)
            logger.info(f"Glyph stream exported to {filepath}")

        return json_str

    def create_session_config(
        self, mode: CCWMode = CCWMode.MEDITATION, duration_minutes: int = 20
    ) -> Dict[str, Any]:
        """
        Create a complete CCW session configuration.

        Args:
            mode: CCW/VIBRANA mode
            duration_minutes: Session duration

        Returns:
            Session configuration dict
        """
        base_freq, harmonic_freq = self.MODE_FREQUENCIES[mode]

        return {
            "session": {
                "mode": mode.value,
                "duration_minutes": duration_minutes,
                "created_at": str(np.datetime64("now")),
            },
            "audio": {
                "base_frequency": base_freq,
                "harmonic_frequency": harmonic_freq,
                "carrier_frequency": self.params.carrier_frequency,
                "binaural_offset": self.params.binaural_offset,
                "sample_rate": self.params.sample_rate,
            },
            "visual": {
                "geometry_pattern": "thirteen_fold",
                "rotation_enabled": True,
                "color_scheme": mode.value,
            },
            "scpn_integration": {
                "enabled": True,
                "update_rate_hz": 10,
                "layers": ["l1", "l4", "l5", "l6", "l7"],
            },
        }


def create_bridge(ccw_params: Optional[CCWParameters] = None) -> CCWBridge:
    """Factory function to create a CCW bridge instance."""
    return CCWBridge(ccw_params)
