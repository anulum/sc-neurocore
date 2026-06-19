# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC Audio Synthesis engine

from typing import Any
import numpy as np
from dataclasses import dataclass


@dataclass
class SCAudioSynthesizer:
    """
    SC Audio Synthesis engine.
    Converts bitstreams/probabilities to waveform buffers.
    """

    sample_rate: int = 44100

    def synthesize_tone(
        self, frequency: float, duration_ms: int, probability: float
    ) -> np.ndarray[Any, Any]:
        """
        Synthesize a simple sine tone modulated by probability (amplitude).
        """
        t = np.linspace(0, duration_ms / 1000, int(self.sample_rate * duration_ms / 1000))
        waveform: np.ndarray[Any, Any] = probability * np.sin(2 * np.pi * frequency * t)
        return waveform

    def bitstream_to_audio(self, bitstream: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """
        Roughly convert a bitstream to an audio signal (Filtering).
        """
        # Low-pass filter the bitstream to get 'analog' signal
        # Simplified: moving average
        window = 10
        audio = np.convolve(bitstream, np.ones(window) / window, mode="same")
        return audio
