# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — EVS Engine -- Entrainment Verification Score

"""
Entrainment Verification Score engine for adaptive audio.

Real-time composite score (0-100) proving that CCW audio entrainment is
working on a per-session basis. Measures the correlation
between the target brainwave frequency and actual EEG spectral power
via FFT-based band analysis.

Score formula (0-100):
    40% relative_increase   -- target band power vs baseline
    30% peak_alignment      -- spectral peak proximity to target Hz
    20% band_dominance      -- target band / total power
    10% temporal_consistency -- inverse of recent score variance

Verified = (score >= 50) AND (confidence >= 0.6)

"""

from __future__ import annotations

import math
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ── EEG Frequency Bands ─────────────────────────────────────────────

BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _hz_to_band(hz: float) -> str:
    """Map a frequency in Hz to its canonical EEG band name."""
    for name, (lo, hi) in BANDS.items():
        if lo <= hz < hi:
            return name
    return "gamma"


# ── Configuration ────────────────────────────────────────────────────


@dataclass
class EVSConfig:
    """Configuration for FFT-based entrainment scoring.

    Attributes
    ----------
    sample_rate:
        EEG sample rate in hertz.
    fft_window:
        Number of samples retained in the ring buffer for FFT scoring.
    baseline_duration_s:
        Baseline collection duration in seconds.
    update_interval_samples:
        Nominal sample interval between external EVS updates.
    """

    sample_rate: int = 256
    fft_window: int = 512
    baseline_duration_s: float = 30.0
    update_interval_samples: int = 128


# ── Snapshot ─────────────────────────────────────────────────────────


@dataclass
class EVSSnapshot:
    """Single-tick entrainment verification observation.

    Attributes
    ----------
    evs_score:
        Composite entrainment score in the inclusive range 0 to 100.
    relative_increase:
        Target-band power increase relative to baseline.
    peak_alignment:
        Alignment between the spectral peak and target frequency.
    band_dominance:
        Fraction of total spectral power in the target band.
    temporal_consistency:
        Stability score computed from recent EVS values.
    is_verified:
        Whether score and confidence clear the verification threshold.
    confidence:
        Confidence score derived from the number of scoring updates.
    target_hz:
        Target entrainment frequency in hertz.
    peak_hz:
        Dominant measured frequency in hertz.
    band_powers:
        Per-band FFT power estimates.
    timestamp:
        Snapshot creation time from ``time.time()``.
    """

    evs_score: float = 0.0
    relative_increase: float = 0.0
    peak_alignment: float = 0.0
    band_dominance: float = 0.0
    temporal_consistency: float = 0.0
    is_verified: bool = False
    confidence: float = 0.0
    target_hz: float = 10.0
    peak_hz: float = 0.0
    band_powers: dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise the snapshot into JSON-compatible telemetry.

        Returns
        -------
        dict[str, Any]
            Rounded score components, verification flags, target and peak
            frequencies, per-band powers, and timestamp.
        """
        return {
            "evs_score": round(self.evs_score, 2),
            "relative_increase": round(self.relative_increase, 4),
            "peak_alignment": round(self.peak_alignment, 4),
            "band_dominance": round(self.band_dominance, 4),
            "temporal_consistency": round(self.temporal_consistency, 4),
            "is_verified": self.is_verified,
            "confidence": round(self.confidence, 4),
            "target_hz": round(self.target_hz, 2),
            "peak_hz": round(self.peak_hz, 2),
            "band_powers": {k: round(v, 6) for k, v in self.band_powers.items()},
            "timestamp": self.timestamp,
        }


# ── Engine ───────────────────────────────────────────────────────────


class EVSEngine:
    """FFT-based Entrainment Verification Score engine.

    Workflow
    --------
    1. ``start_baseline()`` -- begin collecting baseline EEG
    2. ``add_sample(voltage)`` -- feed raw EEG samples one at a time
    3. After baseline_duration_s, baseline finalises automatically
    4. ``set_target(hz)`` -- set the entrainment target frequency
    5. ``compute()`` returns ``EVSSnapshot`` every *update_interval_samples*
    """

    def __init__(self, cfg: EVSConfig | None = None) -> None:
        """Initialise the EVS ring buffer and scoring state.

        Parameters
        ----------
        cfg:
            Optional FFT and baseline configuration. Defaults to ``EVSConfig``
            when omitted.
        """
        self.cfg = cfg or EVSConfig()
        c = self.cfg

        # Sample buffer (ring)
        self._buf = np.zeros(c.fft_window, dtype=np.float64)
        self._buf_idx: int = 0
        self._buf_full: bool = False
        self._total_samples: int = 0

        # Baseline
        self._baseline_active: bool = False
        self._baseline_done: bool = False
        self._baseline_samples: list[float] = []
        self._baseline_powers: dict[str, float] = {}

        # Target
        self._target_hz: float = 10.0

        # Score history for temporal consistency
        self._score_history: list[float] = []

    # ── Baseline ─────────────────────────────────────────────────────

    def start_baseline(self) -> None:
        """Begin baseline EEG collection."""
        self._baseline_active = True
        self._baseline_done = False
        self._baseline_samples.clear()
        self._baseline_powers.clear()
        logger.info("EVS baseline recording started")

    def _finalise_baseline(self) -> None:
        """Compute baseline band powers from collected samples."""
        arr = np.array(self._baseline_samples[-self.cfg.fft_window :])
        if len(arr) < 32:
            # Not enough samples; use flat baseline
            self._baseline_powers = {name: 1.0 for name in BANDS}
        else:
            self._baseline_powers = self._band_powers(arr)
        self._baseline_active = False
        self._baseline_done = True
        logger.info("EVS baseline finalised: %s", self._baseline_powers)

    # ── Sample Ingestion ─────────────────────────────────────────────

    def add_sample(self, voltage: float) -> None:
        """Feed one raw EEG voltage sample."""
        # Ring buffer
        self._buf[self._buf_idx] = voltage
        self._buf_idx = (self._buf_idx + 1) % self.cfg.fft_window
        if self._buf_idx == 0:
            self._buf_full = True
        self._total_samples += 1

        # Baseline collection
        if self._baseline_active:
            self._baseline_samples.append(voltage)
            needed = int(self.cfg.baseline_duration_s * self.cfg.sample_rate)
            if len(self._baseline_samples) >= needed:
                self._finalise_baseline()

    def set_target(self, hz: float) -> None:
        """Set the entrainment target frequency.

        Parameters
        ----------
        hz:
            Finite target frequency in hertz. Values outside the supported EEG
            range are clipped to 0.5-45.0 Hz.

        Raises
        ------
        ValueError
            If ``hz`` is not finite.
        """
        if not math.isfinite(hz):
            raise ValueError("target frequency must be finite")
        self._target_hz = float(np.clip(hz, 0.5, 45.0))

    # ── FFT Helpers ──────────────────────────────────────────────────

    def _ordered_buf(self) -> np.ndarray[Any, Any]:
        """Return the ring buffer in time-order."""
        if not self._buf_full:
            return self._buf[: self._buf_idx].copy()
        return np.concatenate([self._buf[self._buf_idx :], self._buf[: self._buf_idx]])

    def _band_powers(self, signal: np.ndarray[Any, Any]) -> dict[str, float]:
        """Compute power in each canonical EEG band via FFT."""
        n = len(signal)
        if n < 4:
            return {name: 0.0 for name in BANDS}

        # Hanning window
        windowed = signal * np.hanning(n)
        spectrum = np.abs(np.fft.rfft(windowed)) ** 2
        freqs = np.fft.rfftfreq(n, d=1.0 / self.cfg.sample_rate)

        powers: dict[str, float] = {}
        for name, (lo, hi) in BANDS.items():
            mask = (freqs >= lo) & (freqs < hi)
            powers[name] = float(np.mean(spectrum[mask])) if mask.any() else 0.0

        return powers

    def _peak_frequency(self, signal: np.ndarray[Any, Any]) -> float:
        """Dominant frequency in the signal."""
        n = len(signal)
        if n < 4:
            return 0.0
        windowed = signal * np.hanning(n)
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.cfg.sample_rate)
        # Ignore DC
        spectrum[0] = 0.0
        idx = int(np.argmax(spectrum))
        return float(freqs[idx])

    # ── Compute EVS ──────────────────────────────────────────────────

    def compute(self) -> EVSSnapshot | None:
        """Compute current EVS snapshot.

        Returns
        -------
        EVSSnapshot | None
            Current EVS telemetry, or ``None`` until baseline collection is
            complete and enough samples are available.
        """
        if not self._baseline_done:
            return None
        if not self._buf_full and self._buf_idx < 32:
            return None

        signal = self._ordered_buf()
        current_powers = self._band_powers(signal)
        peak_hz = self._peak_frequency(signal)

        target_band = _hz_to_band(self._target_hz)
        target_power = current_powers.get(target_band, 0.0)
        baseline_power = self._baseline_powers.get(target_band, 1.0)
        total_power = sum(current_powers.values()) or 1.0

        # -- Component scores (each 0-1) --

        # 1. Relative increase (40%)
        if baseline_power > 1e-12:
            ri = (target_power - baseline_power) / baseline_power
        else:
            ri = 0.0
        relative_increase = float(np.clip(ri, 0.0, 1.0))

        # 2. Peak alignment (30%)
        band_lo, band_hi = BANDS[target_band]
        band_width = band_hi - band_lo
        alignment = 1.0 - abs(peak_hz - self._target_hz) / band_width
        peak_alignment = float(np.clip(alignment, 0.0, 1.0))

        # 3. Band dominance (20%)
        band_dominance = float(np.clip(target_power / total_power, 0.0, 1.0))

        # 4. Temporal consistency (10%)
        if len(self._score_history) >= 3:
            recent_std = float(np.std(self._score_history[-10:]))
            temporal_consistency = float(np.clip(1.0 - recent_std / 50.0, 0.0, 1.0))
        else:
            temporal_consistency = 0.5

        # Composite score 0-100
        score = (
            40.0 * relative_increase
            + 30.0 * peak_alignment
            + 20.0 * band_dominance
            + 10.0 * temporal_consistency
        )
        score = float(np.clip(score, 0.0, 100.0))
        self._score_history.append(score)

        # Confidence (grows with samples, capped at 1.0)
        n_updates = len(self._score_history)
        confidence = float(np.clip(n_updates / 20.0, 0.0, 1.0))

        is_verified = (score >= 50.0) and (confidence >= 0.6)

        snap = EVSSnapshot(
            evs_score=score,
            relative_increase=relative_increase,
            peak_alignment=peak_alignment,
            band_dominance=band_dominance,
            temporal_consistency=temporal_consistency,
            is_verified=is_verified,
            confidence=confidence,
            target_hz=self._target_hz,
            peak_hz=peak_hz,
            band_powers=current_powers,
            timestamp=time.time(),
        )
        return snap

    # ── Utilities ────────────────────────────────────────────────────

    @property
    def baseline_done(self) -> bool:
        """Whether baseline EEG collection has been finalised."""
        return self._baseline_done

    @property
    def score_history(self) -> list[float]:
        """Return a copy of accumulated EVS scores."""
        return list(self._score_history)

    def reset(self) -> None:
        """Clear buffers, baseline state, and score history."""
        self._buf[:] = 0.0
        self._buf_idx = 0
        self._buf_full = False
        self._total_samples = 0
        self._baseline_active = False
        self._baseline_done = False
        self._baseline_samples.clear()
        self._baseline_powers.clear()
        self._score_history.clear()
