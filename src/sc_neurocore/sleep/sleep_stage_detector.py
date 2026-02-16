"""
Sleep Stage Detector — EEG band power → sleep stage classification
===================================================================

Classifies 5 sleep stages from EEG spectral power ratios:
  WAKE, N1 (light drowsiness), N2 (spindles), N3 (deep/SWS), REM

Uses standard polysomnography rules adapted for single-channel EEG.

Author: Claude (Session 2026-02-16)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional

import numpy as np


class SleepStage(IntEnum):
    """Standard sleep stages."""
    WAKE = 0
    N1 = 1
    N2 = 2
    N3 = 3
    REM = 4


# Stage-specific EEG band power signatures (relative dominance)
STAGE_SIGNATURES = {
    SleepStage.WAKE: {"alpha": 0.3, "beta": 0.4, "gamma": 0.2, "theta": 0.05, "delta": 0.05},
    SleepStage.N1:   {"alpha": 0.15, "beta": 0.1, "theta": 0.5, "delta": 0.15, "gamma": 0.1},
    SleepStage.N2:   {"alpha": 0.1, "beta": 0.05, "theta": 0.3, "delta": 0.4, "gamma": 0.15},
    SleepStage.N3:   {"alpha": 0.05, "beta": 0.02, "theta": 0.1, "delta": 0.8, "gamma": 0.03},
    SleepStage.REM:  {"alpha": 0.1, "beta": 0.2, "theta": 0.4, "delta": 0.1, "gamma": 0.2},
}

EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}


@dataclass
class DetectorConfig:
    """Sleep stage detector configuration."""
    sample_rate: int = 256
    fft_window: int = 512
    smoothing_window: int = 5  # Epochs to smooth over
    min_samples: int = 128     # Minimum for classification


class SleepStageDetector:
    """
    Classifies sleep stage from EEG voltage samples.

    Uses band power ratios compared against known stage signatures.
    Includes temporal smoothing to prevent rapid stage oscillation.
    """

    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        self._buffer: deque = deque(maxlen=self.config.fft_window)
        self._stage_history: deque = deque(maxlen=self.config.smoothing_window)
        self.current_stage = SleepStage.WAKE
        self.band_powers: Dict[str, float] = {}

    def add_sample(self, voltage: float):
        """Add a single EEG voltage sample."""
        self._buffer.append(voltage)

    def add_samples(self, voltages: np.ndarray):
        """Add multiple samples."""
        for v in voltages:
            self._buffer.append(float(v))

    def detect(self) -> SleepStage:
        """Classify current sleep stage from buffer contents."""
        if len(self._buffer) < self.config.min_samples:
            return self.current_stage

        signal = np.array(self._buffer)
        self.band_powers = self._compute_band_powers(signal)

        # Normalize to relative powers
        total = sum(self.band_powers.values()) + 1e-12
        relative = {k: v / total for k, v in self.band_powers.items()}

        # Find best matching stage by cosine similarity
        best_stage = SleepStage.WAKE
        best_score = -1.0
        for stage, sig in STAGE_SIGNATURES.items():
            score = self._cosine_similarity(relative, sig)
            if score > best_score:
                best_score = score
                best_stage = stage

        self._stage_history.append(best_stage)

        # Temporal smoothing: most common stage in recent window
        if len(self._stage_history) >= 3:
            from collections import Counter
            counts = Counter(self._stage_history)
            self.current_stage = counts.most_common(1)[0][0]
        else:
            self.current_stage = best_stage

        return self.current_stage

    def _compute_band_powers(self, signal: np.ndarray) -> Dict[str, float]:
        """Compute power in standard EEG bands."""
        freqs = np.fft.rfftfreq(len(signal), 1.0 / self.config.sample_rate)
        psd = np.abs(np.fft.rfft(signal)) ** 2
        powers = {}
        for band, (lo, hi) in EEG_BANDS.items():
            mask = (freqs >= lo) & (freqs <= hi)
            powers[band] = float(psd[mask].mean()) if mask.any() else 0.0
        return powers

    @staticmethod
    def _cosine_similarity(a: Dict[str, float], b: Dict[str, float]) -> float:
        """Cosine similarity between two band-power dicts."""
        keys = sorted(set(a) | set(b))
        va = np.array([a.get(k, 0) for k in keys])
        vb = np.array([b.get(k, 0) for k in keys])
        dot = float(np.dot(va, vb))
        norm = float(np.linalg.norm(va) * np.linalg.norm(vb))
        return dot / max(norm, 1e-12)

    def get_band_powers(self) -> Dict[str, float]:
        """Return current band powers."""
        return dict(self.band_powers)

    def reset(self):
        """Reset detector state."""
        self._buffer.clear()
        self._stage_history.clear()
        self.current_stage = SleepStage.WAKE
        self.band_powers = {}
