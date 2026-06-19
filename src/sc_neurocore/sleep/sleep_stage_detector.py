# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — EEG band-power based sleep stage classifier

from __future__ import annotations
from typing import Any, Optional

"""EEG band-power based sleep stage classifier.

Uses cosine similarity between observed band-power vectors and canonical
stage signatures to classify 5 sleep stages (WAKE, N1, N2, N3, REM).
Temporal smoothing via a sliding majority-vote window reduces transient
misclassifications.
"""


from collections import Counter, deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Sleep stages
# ---------------------------------------------------------------------------


class SleepStage(IntEnum):
    """AASM sleep-stage labels."""

    WAKE = 0
    N1 = 1
    N2 = 2
    N3 = 3
    REM = 4


# ---------------------------------------------------------------------------
# EEG frequency bands (Hz)
# ---------------------------------------------------------------------------

EEG_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}

# ---------------------------------------------------------------------------
# Canonical band-power signatures per stage (normalised power fractions)
# Order: [delta, theta, alpha, beta, gamma]
# ---------------------------------------------------------------------------

STAGE_SIGNATURES: Dict[SleepStage, np.ndarray[Any, Any]] = {
    SleepStage.WAKE: np.array([0.05, 0.10, 0.35, 0.35, 0.15]),
    SleepStage.N1: np.array([0.10, 0.30, 0.25, 0.25, 0.10]),
    SleepStage.N2: np.array([0.25, 0.25, 0.20, 0.20, 0.10]),
    SleepStage.N3: np.array([0.60, 0.20, 0.10, 0.07, 0.03]),
    SleepStage.REM: np.array([0.10, 0.35, 0.15, 0.25, 0.15]),
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DetectorConfig:
    """Parameters for the sleep-stage detector."""

    sample_rate: int = 256
    fft_window: int = 512
    smoothing_window: int = 5
    min_samples: int = 128


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class SleepStageDetector:
    """Real-time sleep-stage detector from single-channel EEG.

    Usage::

        det = SleepStageDetector()
        for sample in eeg_stream:
            det.add_sample(sample)
            stage = det.detect()
            if stage is not None:
                print(stage.name)
    """

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()
        self._buffer: deque[float] = deque(maxlen=self.config.fft_window)
        self._stage_history: deque[SleepStage] = deque(maxlen=self.config.smoothing_window)
        self._band_powers: Optional[Dict[str, float]] = None

    # -- public API ---------------------------------------------------------

    def add_sample(self, sample: float) -> None:
        """Append a single EEG voltage sample to the internal buffer."""
        self._buffer.append(float(sample))

    def add_samples(self, samples: np.ndarray[Any, Any]) -> None:
        """Append an array of EEG voltage samples."""
        for s in np.asarray(samples).ravel():
            self._buffer.append(float(s))

    def detect(self) -> Optional[SleepStage]:
        """Return the smoothed sleep-stage classification, or ``None`` if
        insufficient data has been collected."""
        if len(self._buffer) < self.config.min_samples:
            return None

        powers = self._compute_band_powers()
        self._band_powers = powers

        power_vec = np.array([powers[b] for b in EEG_BANDS])
        raw_stage = self._classify(power_vec)
        self._stage_history.append(raw_stage)

        # temporal smoothing: majority vote over recent detections
        return self._smooth()

    def get_band_powers(self) -> Optional[Dict[str, float]]:
        """Return the most recently computed band-power dict, or ``None``."""
        return self._band_powers

    def reset(self) -> None:
        """Clear all internal state."""
        self._buffer.clear()
        self._stage_history.clear()
        self._band_powers = None

    # -- internals ----------------------------------------------------------

    def _compute_band_powers(self) -> Dict[str, float]:
        """Compute absolute band powers from the current buffer via FFT."""
        data = np.array(self._buffer, dtype=np.float64)
        # Apply Hann window
        window = np.hanning(len(data))
        data = data * window

        fft_vals = np.fft.rfft(data)
        psd = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(len(data), d=1.0 / self.config.sample_rate)

        powers: Dict[str, float] = {}
        for band_name, (lo, hi) in EEG_BANDS.items():
            mask = (freqs >= lo) & (freqs < hi)
            powers[band_name] = float(psd[mask].mean()) if mask.any() else 0.0

        return powers

    @staticmethod
    def _classify(power_vec: np.ndarray[Any, Any]) -> SleepStage:
        """Classify by cosine similarity to canonical signatures."""
        norm = np.linalg.norm(power_vec)
        if norm < 1e-12:
            return SleepStage.WAKE

        best_stage = SleepStage.WAKE
        best_sim = -1.0
        for stage, sig in STAGE_SIGNATURES.items():
            sim = float(np.dot(power_vec, sig) / (norm * np.linalg.norm(sig)))
            if sim > best_sim:
                best_sim = sim
                best_stage = stage
        return best_stage

    def _smooth(self) -> SleepStage:
        """Majority-vote smoothing over the recent stage history."""
        counter = Counter(self._stage_history)
        return counter.most_common(1)[0][0]
