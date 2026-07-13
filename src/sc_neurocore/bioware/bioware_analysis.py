# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Culture, LFP, burst, and latency analysis

"""Culture health, LFP, network-burst, and latency analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .bioware_contracts import DetectedSpike
from .bioware_validation import (
    require_nonnegative,
    require_nonnegative_int,
    require_positive,
    require_positive_int,
    validate_voltage_matrix,
)


@dataclass
class CultureHealth:
    """Monitor organoid/culture viability from MEA activity."""

    min_active_channels: int = 5
    min_firing_rate_hz: float = 0.1
    max_firing_rate_hz: float = 100.0
    burst_threshold_hz: float = 50.0

    def __post_init__(self) -> None:
        """Validate rate thresholds used by the aggregate health heuristic."""
        require_positive_int(self.min_active_channels, "min_active_channels")
        require_nonnegative(self.min_firing_rate_hz, "min_firing_rate_hz")
        require_positive(self.max_firing_rate_hz, "max_firing_rate_hz")
        if self.max_firing_rate_hz <= self.min_firing_rate_hz:
            raise ValueError("max_firing_rate_hz must exceed min_firing_rate_hz")
        require_nonnegative(self.burst_threshold_hz, "burst_threshold_hz")

    def assess(self, spike_counts: np.ndarray[Any, Any], duration_s: float) -> Dict[str, float]:
        """Assess culture health from spike activity.

        spike_counts: per-channel spike counts over duration_s
        """
        if not isinstance(spike_counts, np.ndarray):
            raise TypeError("spike_counts must be a NumPy array")
        if spike_counts.ndim != 1 or spike_counts.size == 0:
            raise ValueError("spike_counts must be a non-empty one-dimensional array")
        if not np.issubdtype(spike_counts.dtype, np.number):
            raise TypeError("spike_counts must have a numeric dtype")
        if not np.all(np.isfinite(spike_counts)) or np.any(spike_counts < 0):
            raise ValueError("spike_counts must contain finite non-negative values")
        require_positive(duration_s, "duration_s")
        rates = spike_counts / duration_s
        active = np.sum(rates > self.min_firing_rate_hz)
        mean_rate = float(np.mean(rates[rates > 0])) if np.any(rates > 0) else 0.0
        bursting = np.sum(rates > self.burst_threshold_hz)

        health_score = 1.0
        if active < self.min_active_channels:
            health_score *= active / self.min_active_channels
        if mean_rate > self.max_firing_rate_hz:
            health_score *= self.max_firing_rate_hz / mean_rate

        return {
            "active_channels": int(active),
            "mean_firing_rate_hz": mean_rate,
            "bursting_channels": int(bursting),
            "health_score": float(np.clip(health_score, 0.0, 1.0)),
            "is_viable": bool(health_score > 0.5),
        }


@dataclass
class LFPBand:
    """Frequency band definition for LFP extraction."""

    name: str
    low_hz: float
    high_hz: float

    def __post_init__(self) -> None:
        """Validate a named half-open frequency interval."""
        if not self.name or not self.name.strip():
            raise ValueError("LFP band name must not be empty")
        require_nonnegative(self.low_hz, "low_hz")
        require_positive(self.high_hz, "high_hz")
        if self.high_hz <= self.low_hz:
            raise ValueError("high_hz must exceed low_hz")


DEFAULT_LFP_BANDS = [
    LFPBand("delta", 0.5, 4.0),
    LFPBand("theta", 4.0, 8.0),
    LFPBand("alpha", 8.0, 13.0),
    LFPBand("beta", 13.0, 30.0),
    LFPBand("gamma", 30.0, 100.0),
]


def extract_lfp_power(
    voltage_data: np.ndarray[Any, Any],
    sample_rate_hz: float,
    bands: Optional[List[LFPBand]] = None,
) -> Dict[str, np.ndarray[Any, Any]]:
    """Extract per-channel power in each LFP band.

    Uses FFT-based power spectral density estimation.
    Returns dict of band_name → per-channel power array.
    """
    if bands is None:
        bands = DEFAULT_LFP_BANDS

    validate_voltage_matrix(voltage_data)
    require_positive(sample_rate_hz, "sample_rate_hz")
    if not bands:
        raise ValueError("bands must not be empty")
    names: set[str] = set()
    for band in bands:
        if not isinstance(band, LFPBand):
            raise TypeError("bands must contain LFPBand instances")
        if band.name in names:
            raise ValueError(f"duplicate LFP band name: {band.name}")
        names.add(band.name)

    n_samples, n_channels = voltage_data.shape
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate_hz)
    fft_mag = np.abs(np.fft.rfft(voltage_data, axis=0)) ** 2

    result = {}
    for band in bands:
        mask = (freqs >= band.low_hz) & (freqs < band.high_hz)
        power = np.sum(fft_mag[mask, :], axis=0) if mask.any() else np.zeros(n_channels)
        result[band.name] = power
    return result


@dataclass
class LatencyBudget:
    """Tracks and enforces closed-loop latency requirements."""

    max_latency_us: float = 1000.0  # 1 ms default
    history: List[float] = field(default_factory=list)
    violations: int = 0

    def __post_init__(self) -> None:
        """Validate budget, history, and violation accounting."""
        require_positive(self.max_latency_us, "max_latency_us")
        for latency_us in self.history:
            require_nonnegative(latency_us, "history latency_us")
        require_nonnegative_int(self.violations, "violations")
        if self.violations > len(self.history):
            raise ValueError("violations cannot exceed the number of history samples")

    def record(self, latency_us: float) -> bool:
        """Record a latency measurement. Returns True if within budget."""
        require_nonnegative(latency_us, "latency_us")
        self.history.append(latency_us)
        if latency_us > self.max_latency_us:
            self.violations += 1
            return False
        return True

    @property
    def mean_latency_us(self) -> float:
        """Return the arithmetic mean of recorded loop latencies.

        Returns
        -------
        float
            Mean latency in microseconds, or ``0.0`` before any samples exist.
        """
        return float(np.mean(self.history)) if self.history else 0.0

    @property
    def p99_latency_us(self) -> float:
        """Return the 99th percentile closed-loop latency.

        Returns
        -------
        float
            99th percentile latency in microseconds, or ``0.0`` for an empty
            history.
        """
        return float(np.percentile(self.history, 99)) if self.history else 0.0

    @property
    def compliance_ratio(self) -> float:
        """Return the fraction of samples inside the latency budget.

        Returns
        -------
        float
            Ratio in ``[0.0, 1.0]``; an empty history is defined as fully
            compliant.
        """
        if not self.history:
            return 1.0
        return 1.0 - self.violations / len(self.history)


@dataclass
class NetworkBurst:
    """Detected network-wide synchronised burst event."""

    onset_s: float
    duration_s: float
    participating_channels: int
    total_spikes: int

    def __post_init__(self) -> None:
        """Validate a detected network-burst summary."""
        require_nonnegative(self.onset_s, "onset_s")
        require_positive(self.duration_s, "duration_s")
        require_positive_int(self.participating_channels, "participating_channels")
        require_positive_int(self.total_spikes, "total_spikes")


def detect_network_bursts(
    spikes: List[DetectedSpike],
    bin_width_s: float = 0.01,
    threshold_sigma: float = 3.0,
    min_channels: int = 3,
) -> List[NetworkBurst]:
    """Detect network-wide synchronised bursts.

    Bins spikes in time, detects bins with activity > threshold_sigma
    above the mean, and requires participation from ≥ min_channels.
    """
    require_positive(bin_width_s, "bin_width_s")
    require_nonnegative(threshold_sigma, "threshold_sigma")
    require_positive_int(min_channels, "min_channels")
    if not spikes:
        return []

    timestamps = np.array([s.timestamp_s for s in spikes])
    t_start, t_end = timestamps.min(), timestamps.max()
    if t_end <= t_start:
        return []

    n_bins = max(1, int((t_end - t_start) / bin_width_s) + 1)
    bin_counts = np.zeros(n_bins)
    bin_channels: List[set[int]] = [set() for _ in range(n_bins)]

    for s in spikes:
        idx = min(int((s.timestamp_s - t_start) / bin_width_s), n_bins - 1)
        bin_counts[idx] += 1
        bin_channels[idx].add(s.channel)

    mean_count = np.mean(bin_counts)
    std_count = np.std(bin_counts)
    if std_count == 0:
        return []
    threshold = mean_count + threshold_sigma * std_count

    bursts = []
    for i in range(n_bins):
        if bin_counts[i] >= threshold and len(bin_channels[i]) >= min_channels:
            bursts.append(
                NetworkBurst(
                    onset_s=t_start + i * bin_width_s,
                    duration_s=bin_width_s,
                    participating_channels=len(bin_channels[i]),
                    total_spikes=int(bin_counts[i]),
                )
            )
    return bursts
