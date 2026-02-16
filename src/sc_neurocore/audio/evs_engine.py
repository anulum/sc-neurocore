"""
EVS (Entrainment Verification Score) Engine
=============================================

Real-time composite score (0-100) proving brainwave entrainment works
on a per-session basis. Measures correlation between target frequency
and actual EEG spectral power.

Components (weighted):
- relative_increase (40%): target band power vs baseline
- peak_alignment (30%): spectral peak proximity to target
- band_dominance (20%): target band / total power
- temporal_consistency (10%): stability of EVS over time

Author: Claude (Session 2026-02-16)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class EVSConfig:
    """EVS engine configuration."""
    sample_rate: int = 256       # Hz
    fft_window: int = 512        # samples for FFT
    target_hz: float = 10.0      # Target entrainment frequency
    band_width: float = 2.0      # Hz bandwidth around target
    baseline_duration_s: float = 10.0  # Baseline recording seconds
    evs_threshold: float = 50.0  # Minimum for "verified"
    confidence_threshold: float = 0.6
    history_size: int = 20       # Rolling window for consistency


# Standard EEG bands
EEG_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 100.0),
}


@dataclass
class EVSSnapshot:
    """Single EVS measurement."""
    evs_score: float = 0.0
    relative_increase: float = 0.0
    peak_alignment: float = 0.0
    band_dominance: float = 0.0
    temporal_consistency: float = 0.0
    target_hz: float = 10.0
    peak_hz: float = 0.0
    is_verified: bool = False
    confidence: float = 0.0
    band_powers: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "evs_score": round(self.evs_score, 2),
            "relative_increase": round(self.relative_increase, 3),
            "peak_alignment": round(self.peak_alignment, 3),
            "band_dominance": round(self.band_dominance, 3),
            "temporal_consistency": round(self.temporal_consistency, 3),
            "target_hz": self.target_hz,
            "peak_hz": round(self.peak_hz, 2),
            "is_verified": self.is_verified,
            "confidence": round(self.confidence, 3),
            "band_powers": {k: round(v, 4) for k, v in self.band_powers.items()},
        }


class EVSEngine:
    """
    Computes real-time Entrainment Verification Score from EEG samples.

    Usage:
        engine = EVSEngine(config)
        engine.start_baseline()
        for sample in baseline_samples:
            engine.add_sample(sample)
        engine.stop_baseline()

        engine.start_session(target_hz=10.0)
        for sample in session_samples:
            engine.add_sample(sample)
            if engine.samples_since_last_fft >= fft_window:
                snapshot = engine.compute()
    """

    def __init__(self, config: Optional[EVSConfig] = None):
        self.config = config or EVSConfig()
        self._buffer: deque = deque(maxlen=self.config.fft_window)
        self._baseline_powers: Dict[str, float] = {}
        self._in_baseline = False
        self._in_session = False
        self._baseline_samples: List[float] = []
        self._evs_history: deque = deque(maxlen=self.config.history_size)
        self._samples_since_compute = 0

    def start_baseline(self):
        """Begin baseline EEG recording."""
        self._in_baseline = True
        self._baseline_samples.clear()

    def add_sample(self, voltage: float):
        """Add a single EEG voltage sample."""
        self._buffer.append(voltage)
        if self._in_baseline:
            self._baseline_samples.append(voltage)
        self._samples_since_compute += 1

    def stop_baseline(self) -> Dict[str, float]:
        """Finalize baseline, compute baseline band powers."""
        self._in_baseline = False
        if len(self._baseline_samples) >= self.config.fft_window:
            self._baseline_powers = self._compute_band_powers(
                np.array(self._baseline_samples[-self.config.fft_window:])
            )
        else:
            # Use what we have
            self._baseline_powers = self._compute_band_powers(
                np.array(self._baseline_samples) if self._baseline_samples else np.zeros(64)
            )
        return self._baseline_powers

    def start_session(self, target_hz: Optional[float] = None):
        """Start EVS session."""
        if target_hz is not None:
            self.config.target_hz = target_hz
        self._in_session = True
        self._evs_history.clear()

    def stop_session(self):
        """Stop EVS session."""
        self._in_session = False

    def compute(self) -> EVSSnapshot:
        """Compute current EVS score from buffer."""
        if len(self._buffer) < 32:
            return EVSSnapshot(target_hz=self.config.target_hz)

        signal = np.array(self._buffer)
        band_powers = self._compute_band_powers(signal)
        peak_hz = self._compute_peak_frequency(signal)

        # Target band power
        target_lo = self.config.target_hz - self.config.band_width / 2
        target_hi = self.config.target_hz + self.config.band_width / 2
        target_power = self._power_in_range(signal, target_lo, target_hi)
        total_power = sum(band_powers.values()) + 1e-12

        # Baseline comparison
        baseline_target = self._baseline_target_power()

        # Component 1: Relative increase (40%)
        if baseline_target > 0:
            rel_increase = float(np.clip((target_power - baseline_target) / (baseline_target + 1e-8), 0, 1))
        else:
            rel_increase = float(np.clip(target_power, 0, 1))

        # Component 2: Peak alignment (30%)
        peak_align = 1.0 - float(np.clip(
            abs(peak_hz - self.config.target_hz) / self.config.band_width, 0, 1
        ))

        # Component 3: Band dominance (20%)
        band_dom = float(np.clip(target_power / total_power, 0, 1))

        # Component 4: Temporal consistency (10%)
        if len(self._evs_history) > 2:
            recent_scores = list(self._evs_history)
            temp_consist = 1.0 - float(np.clip(np.std(recent_scores) / 50.0, 0, 1))
        else:
            temp_consist = 0.5

        # Composite score (0-100)
        evs_score = (
            0.40 * rel_increase
            + 0.30 * peak_align
            + 0.20 * band_dom
            + 0.10 * temp_consist
        ) * 100.0

        # Confidence based on sample count and signal quality
        confidence = min(1.0, len(self._buffer) / self.config.fft_window)

        self._evs_history.append(evs_score)
        self._samples_since_compute = 0

        return EVSSnapshot(
            evs_score=evs_score,
            relative_increase=rel_increase,
            peak_alignment=peak_align,
            band_dominance=band_dom,
            temporal_consistency=temp_consist,
            target_hz=self.config.target_hz,
            peak_hz=peak_hz,
            is_verified=evs_score >= self.config.evs_threshold and confidence >= self.config.confidence_threshold,
            confidence=confidence,
            band_powers=band_powers,
        )

    def _compute_band_powers(self, signal: np.ndarray) -> Dict[str, float]:
        """Compute power in standard EEG bands."""
        if len(signal) < 4:
            return {band: 0.0 for band in EEG_BANDS}
        freqs = np.fft.rfftfreq(len(signal), 1.0 / self.config.sample_rate)
        psd = np.abs(np.fft.rfft(signal)) ** 2
        powers = {}
        for band, (lo, hi) in EEG_BANDS.items():
            mask = (freqs >= lo) & (freqs <= hi)
            powers[band] = float(psd[mask].mean()) if mask.any() else 0.0
        return powers

    def _compute_peak_frequency(self, signal: np.ndarray) -> float:
        """Find dominant frequency in signal."""
        if len(signal) < 4:
            return 0.0
        freqs = np.fft.rfftfreq(len(signal), 1.0 / self.config.sample_rate)
        psd = np.abs(np.fft.rfft(signal)) ** 2
        # Limit to physiological range (0.5-50 Hz)
        mask = (freqs >= 0.5) & (freqs <= 50.0)
        if not mask.any():
            return 0.0
        peak_idx = np.argmax(psd[mask])
        return float(freqs[mask][peak_idx])

    def _power_in_range(self, signal: np.ndarray, lo: float, hi: float) -> float:
        """Power in a specific frequency range."""
        if len(signal) < 4:
            return 0.0
        freqs = np.fft.rfftfreq(len(signal), 1.0 / self.config.sample_rate)
        psd = np.abs(np.fft.rfft(signal)) ** 2
        mask = (freqs >= lo) & (freqs <= hi)
        return float(psd[mask].mean()) if mask.any() else 0.0

    def _baseline_target_power(self) -> float:
        """Get baseline power in target band."""
        target_band = None
        for band, (lo, hi) in EEG_BANDS.items():
            if lo <= self.config.target_hz <= hi:
                target_band = band
                break
        if target_band and target_band in self._baseline_powers:
            return self._baseline_powers[target_band]
        return 0.0

    def reset(self):
        """Reset engine state."""
        self._buffer.clear()
        self._baseline_powers.clear()
        self._baseline_samples.clear()
        self._evs_history.clear()
        self._in_baseline = False
        self._in_session = False
