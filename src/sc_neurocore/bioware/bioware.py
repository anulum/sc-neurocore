# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# mypy: ignore-errors
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bio-Hybrid Wetware Interface Primitives

"""Interface primitives for living neural cultures and organoids.

Bridges biological neural activity (from MEA recordings) to SC bitstreams
and vice-versa. Enables closed-loop bio-hybrid experiments where:

1. **MEA → AER**: Spike-sorts multi-electrode array data into AER events
   compatible with ``sc_aer_encoder.v`` / ``sc_aer_router.v``.
2. **AER → SC**: Converts AER events into SC bitstreams for deterministic
   stochastic processing.
3. **SC → Optogenetics**: Encodes SC output as optical pulse sequences
   for closed-loop stimulation.
4. **Biological Plasticity**: STDP/BCM adapters bridging biological
   time constants (ms) to SC clock rates (MHz).

Compatible with:
- ``hdl/sc_aer_encoder.v`` — AER spike encoding
- ``hdl/sc_aer_router.v`` — AER spike routing
- ``analysis/spike_stats`` — spike train analysis
- ``profiling/spike_profiler.py`` — spike rate profiling
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np


# ── MEA Configuration ────────────────────────────────────────────────


class MEALayout(Enum):
    """Standard MEA electrode layouts."""

    MEA_60 = "60ch"
    MEA_120 = "120ch"
    MEA_256 = "256ch"
    MEA_4096 = "4096ch"
    CUSTOM = "custom"


@dataclass
class MEAConfig:
    """Multi-electrode array configuration."""

    layout: MEALayout = MEALayout.MEA_60
    num_channels: int = 60
    sample_rate_hz: float = 20_000.0
    voltage_gain: float = 1000.0
    noise_floor_uv: float = 5.0
    spike_threshold_sigma: float = 5.0
    electrode_pitch_um: float = 200.0

    @classmethod
    def from_layout(cls, layout: MEALayout) -> MEAConfig:
        presets = {
            MEALayout.MEA_60: dict(num_channels=60, electrode_pitch_um=200.0),
            MEALayout.MEA_120: dict(num_channels=120, electrode_pitch_um=100.0),
            MEALayout.MEA_256: dict(num_channels=256, electrode_pitch_um=100.0),
            MEALayout.MEA_4096: dict(num_channels=4096, electrode_pitch_um=17.5),
            MEALayout.CUSTOM: dict(num_channels=60, electrode_pitch_um=200.0),
        }
        return cls(layout=layout, **presets[layout])


# ── Spike Detection + Sorting ────────────────────────────────────────


@dataclass
class DetectedSpike:
    """One detected spike event from MEA data."""

    channel: int
    timestamp_s: float
    amplitude_uv: float
    unit_id: int = 0  # cluster assignment
    waveform: Optional[np.ndarray] = None


@dataclass
class SpikeDetector:
    """Threshold-based spike detector for MEA voltage traces.

    Uses adaptive threshold: threshold = mean ± sigma * noise_estimate
    where noise_estimate = median(|x|) / 0.6745 (robust RMS).
    Supports configurable refractory period to prevent double-counting.
    """

    config: MEAConfig
    refractory_samples: int = 30
    _noise_estimates: Optional[np.ndarray] = field(default=None, repr=False)

    def estimate_noise(self, voltage_data: np.ndarray) -> np.ndarray:
        """Estimate per-channel noise from voltage data.

        Uses median absolute deviation (MAD) for robustness against spikes.
        voltage_data: shape (num_samples, num_channels)
        """
        mad = np.median(np.abs(voltage_data), axis=0) / 0.6745
        self._noise_estimates = mad
        return mad

    def detect(self, voltage_data: np.ndarray) -> List[DetectedSpike]:
        """Detect spikes in multi-channel voltage data.

        voltage_data: shape (num_samples, num_channels)
        Returns list of DetectedSpike events.
        """
        n_samples, n_channels = voltage_data.shape
        if self._noise_estimates is None:
            self.estimate_noise(voltage_data)
        assert self._noise_estimates is not None

        spikes = []
        dt = 1.0 / self.config.sample_rate_hz
        sigma = self.config.spike_threshold_sigma

        for ch in range(n_channels):
            threshold = sigma * self._noise_estimates[ch]
            above = np.abs(voltage_data[:, ch]) > threshold
            crossings = np.where(np.diff(above.astype(int)) == 1)[0]
            last_spike_idx = -self.refractory_samples - 1
            for idx in crossings:
                if idx - last_spike_idx < self.refractory_samples:
                    continue
                last_spike_idx = idx
                amp = float(voltage_data[idx, ch])
                ts = idx * dt
                spikes.append(
                    DetectedSpike(
                        channel=ch,
                        timestamp_s=ts,
                        amplitude_uv=amp,
                        unit_id=ch,
                    )
                )
        return spikes


# ── MEA → AER Transcoder ─────────────────────────────────────────────


@dataclass
class AEREvent:
    """Address-Event Representation packet.

    Compatible with sc_aer_encoder.v format:
    {valid, neuron_id, timestamp}
    """

    neuron_id: int
    timestamp: int  # clock ticks (not real time)
    valid: bool = True
    weight: int = 256  # Q8.8 = 1.0


@dataclass
class MEAToAERTranscoder:
    """Converts MEA spike events to AER events for hardware.

    Maps biological electrode channels to AER neuron IDs,
    converting real-time timestamps to hardware clock ticks.
    """

    hw_clock_hz: float = 1e6  # 1 MHz default AER clock
    channel_map: Optional[Dict[int, int]] = None  # electrode → neuron_id

    def transcode(
        self,
        spikes: List[DetectedSpike],
        t_start_s: float = 0.0,
    ) -> List[AEREvent]:
        """Convert detected spikes to AER events."""
        events = []
        for spike in spikes:
            neuron_id = self._map_channel(spike.channel)
            ts_hw = int((spike.timestamp_s - t_start_s) * self.hw_clock_hz) & 0xFFFF
            events.append(
                AEREvent(
                    neuron_id=neuron_id,
                    timestamp=ts_hw,
                    valid=True,
                )
            )
        # Sort by timestamp (AER is time-ordered)
        events.sort(key=lambda e: e.timestamp)
        return events

    def _map_channel(self, channel: int) -> int:
        if self.channel_map is not None:
            return self.channel_map.get(channel, channel)
        return channel


# ── AER → SC Bitstream Converter ─────────────────────────────────────


@dataclass
class AERToSCConverter:
    """Converts AER event streams to SC bitstreams.

    Uses a time-windowed rate code: count events per neuron per window,
    then LFSR-encode the resulting firing probabilities.
    """

    window_ticks: int = 1000
    bitstream_length: int = 256
    num_neurons: int = 128
    lfsr_seed: int = 0xACE1

    def convert(self, events: List[AEREvent]) -> Dict[int, np.ndarray]:
        """Convert AER events to per-neuron SC bitstreams."""
        # Count events per neuron in the window
        counts: Dict[int, int] = {}
        for e in events:
            if e.valid:
                counts[e.neuron_id] = counts.get(e.neuron_id, 0) + 1

        max_count = max(counts.values()) if counts else 1
        bitstreams = {}
        for nid, count in counts.items():
            prob = count / max_count
            bitstreams[nid] = self._lfsr_encode(prob, nid)
        return bitstreams

    def _lfsr_encode(self, probability: float, neuron_id: int) -> np.ndarray:
        """LFSR-16 encoding (bit-compatible with core_engine)."""
        threshold = int(np.clip(probability, 0.0, 1.0) * 65535)
        seed = (self.lfsr_seed + neuron_id * 7919) & 0xFFFF
        if seed == 0:
            seed = 1
        reg = seed
        bits = np.zeros(self.bitstream_length, dtype=np.uint8)
        for i in range(self.bitstream_length):
            bits[i] = 1 if reg < threshold else 0
            feedback = ((reg >> 15) ^ (reg >> 13) ^ (reg >> 12) ^ (reg >> 10)) & 1
            reg = ((reg << 1) | feedback) & 0xFFFF
        return bits


# ── SC → Optogenetic Stimulation ─────────────────────────────────────


class StimProtocol(Enum):
    """Optogenetic stimulation protocols."""

    CONSTANT = "constant"
    PULSED = "pulsed"
    GRADED = "graded"
    PATTERN = "pattern"


@dataclass
class OptogeneticPulse:
    """One optical stimulation pulse."""

    channel: int
    onset_ms: float
    duration_ms: float
    intensity_mw_mm2: float = 1.0
    wavelength_nm: int = 470  # blue (ChR2)


@dataclass
class SCToOptoEncoder:
    """Encodes SC bitstream output as optogenetic pulse sequences.

    Maps SC bitstream density to optical stimulation intensity,
    enabling closed-loop feedback from in-silico → biological.
    Enforces total power budget for tissue safety.
    """

    max_intensity_mw_mm2: float = 5.0
    min_pulse_ms: float = 1.0
    max_pulse_ms: float = 50.0
    wavelength_nm: int = 470
    clock_period_ms: float = 0.001  # 1 MHz
    max_total_power_mw: float = 50.0

    def encode(
        self,
        bitstreams: Dict[int, np.ndarray],
        t_start_ms: float = 0.0,
    ) -> List[OptogeneticPulse]:
        """Convert SC bitstreams to optogenetic pulses."""
        pulses = []
        total_power = 0.0
        for nid, bs in sorted(bitstreams.items()):
            density = float(np.sum(bs)) / len(bs) if len(bs) > 0 else 0.0
            if density < 0.01:
                continue

            intensity = density * self.max_intensity_mw_mm2
            if total_power + intensity > self.max_total_power_mw:
                break
            total_power += intensity

            duration = self.min_pulse_ms + density * (self.max_pulse_ms - self.min_pulse_ms)
            onset = t_start_ms + nid * self.clock_period_ms

            pulses.append(
                OptogeneticPulse(
                    channel=nid,
                    onset_ms=onset,
                    duration_ms=duration,
                    intensity_mw_mm2=intensity,
                    wavelength_nm=self.wavelength_nm,
                )
            )
        return pulses


# ── Biological Plasticity Adapters ───────────────────────────────────


@dataclass
class BiologicalSTDP:
    """Spike-Timing-Dependent Plasticity adapter for bio-hybrid loops.

    Bridges biological STDP time constants (∼20 ms) to SC clock
    rates (MHz) via a time-scaling factor. Computes ΔW from
    pre/post spike timing in biological time, then converts to
    Q8.8 weight updates for the SC domain.
    """

    tau_plus_ms: float = 20.0  # potentiation time constant
    tau_minus_ms: float = 20.0  # depression time constant
    a_plus: float = 0.01  # potentiation amplitude
    a_minus: float = 0.012  # depression amplitude (slightly > a_plus)
    w_max_q88: int = 512  # Q8.8 = 2.0
    w_min_q88: int = 0

    def compute_dw(self, dt_ms: float) -> float:
        """Compute weight change from spike timing difference.

        dt_ms = t_post - t_pre (positive = potentiation, negative = depression)
        """
        if dt_ms > 0:
            return self.a_plus * np.exp(-dt_ms / self.tau_plus_ms)
        elif dt_ms < 0:
            return -self.a_minus * np.exp(dt_ms / self.tau_minus_ms)
        return 0.0

    def update_weight(self, current_q88: int, dt_ms: float) -> int:
        """Update Q8.8 weight from spike timing."""
        dw = self.compute_dw(dt_ms)
        dw_q88 = int(dw * 256)  # Convert to Q8.8
        new_w = current_q88 + dw_q88
        return max(self.w_min_q88, min(self.w_max_q88, new_w))


@dataclass
class BCMPlasticity:
    """Bienenstock-Cooper-Munro plasticity adapter.

    Implements sliding-threshold BCM rule where the modification
    threshold θ tracks the postsynaptic firing rate. Converts
    biological firing rates to Q8.8 weight deltas.
    """

    tau_theta_ms: float = 1000.0  # threshold adaptation time constant
    learning_rate: float = 0.001
    theta: float = 0.0  # sliding threshold (internal state)
    w_max_q88: int = 512
    w_min_q88: int = 0

    def update_theta(self, post_rate_hz: float, dt_ms: float) -> float:
        """Update the sliding threshold from postsynaptic activity."""
        alpha = dt_ms / self.tau_theta_ms
        target = post_rate_hz**2
        self.theta += alpha * (target - self.theta)
        return self.theta

    def compute_dw(self, pre_rate_hz: float, post_rate_hz: float) -> float:
        """BCM weight change: ΔW = η * x * y * (y - θ)."""
        return self.learning_rate * pre_rate_hz * post_rate_hz * (post_rate_hz - self.theta)

    def update_weight(self, current_q88: int, pre_rate: float, post_rate: float) -> int:
        dw = self.compute_dw(pre_rate, post_rate)
        dw_q88 = int(dw * 256)
        new_w = current_q88 + dw_q88
        return max(self.w_min_q88, min(self.w_max_q88, new_w))


# ── Culture Health Monitor ───────────────────────────────────────────


@dataclass
class CultureHealth:
    """Monitor organoid/culture viability from MEA activity."""

    min_active_channels: int = 5
    min_firing_rate_hz: float = 0.1
    max_firing_rate_hz: float = 100.0
    burst_threshold_hz: float = 50.0

    def assess(self, spike_counts: np.ndarray, duration_s: float) -> Dict[str, float]:
        """Assess culture health from spike activity.

        spike_counts: per-channel spike counts over duration_s
        """
        rates = spike_counts / duration_s if duration_s > 0 else spike_counts
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


# ── Bio-Hybrid Session ──────────────────────────────────────────────


@dataclass
class BioHybridSession:
    """Manages a complete bio-hybrid experiment session.

    Orchestrates: MEA recording → spike detection → AER transcoding →
    SC processing → optogenetic feedback → plasticity update.
    """

    mea_config: MEAConfig
    detector: SpikeDetector
    transcoder: MEAToAERTranscoder
    sc_converter: AERToSCConverter
    opto_encoder: SCToOptoEncoder
    stdp: BiologicalSTDP = field(default_factory=BiologicalSTDP)
    health_monitor: CultureHealth = field(default_factory=CultureHealth)
    round_count: int = 0

    def process_frame(
        self,
        voltage_data: np.ndarray,
        t_start_s: float = 0.0,
    ) -> Dict:
        """Process one MEA data frame through the full pipeline.

        Returns dict with intermediate results at each stage.
        """
        t0 = time.perf_counter_ns()
        self.round_count += 1

        # 1. Detect spikes
        spikes = self.detector.detect(voltage_data)

        # 2. Transcode to AER
        aer_events = self.transcoder.transcode(spikes, t_start_s)

        # 3. Convert to SC bitstreams
        bitstreams = self.sc_converter.convert(aer_events)

        # 4. Generate optogenetic pulses
        opto_pulses = self.opto_encoder.encode(bitstreams)

        # 5. Health assessment
        n_channels = voltage_data.shape[1]
        spike_counts = np.zeros(n_channels)
        for s in spikes:
            if s.channel < n_channels:
                spike_counts[s.channel] += 1
        duration = voltage_data.shape[0] / self.mea_config.sample_rate_hz
        health = self.health_monitor.assess(spike_counts, duration)

        latency_us = (time.perf_counter_ns() - t0) / 1000.0

        return {
            "round": self.round_count,
            "num_spikes": len(spikes),
            "num_aer_events": len(aer_events),
            "num_bitstreams": len(bitstreams),
            "num_opto_pulses": len(opto_pulses),
            "latency_us": latency_us,
            "health": health,
            "spikes": spikes,
            "aer_events": aer_events,
            "bitstreams": bitstreams,
            "opto_pulses": opto_pulses,
        }


# ── Spike Sorter — template matching (Gap 1) ────────────────────────


@dataclass
class SpikeSorter:
    """Simple template-based spike sorter using amplitude clustering.

    Assigns detected spikes to unit IDs based on amplitude bins.
    For production use, replace with PCA + k-means or wavelet methods.
    """

    num_units: int = 4
    amplitude_bins: Optional[np.ndarray] = None

    def fit(self, spikes: List[DetectedSpike]) -> None:
        """Compute amplitude bin edges from training data."""
        amps = np.array([abs(s.amplitude_uv) for s in spikes])
        if len(amps) == 0:
            self.amplitude_bins = np.array([])
            return
        self.amplitude_bins = np.linspace(amps.min(), amps.max(), self.num_units + 1)

    def assign(self, spikes: List[DetectedSpike]) -> List[DetectedSpike]:
        """Assign unit_id to each spike based on amplitude bins."""
        if self.amplitude_bins is None or len(self.amplitude_bins) == 0:
            return spikes
        result = []
        for s in spikes:
            amp = abs(s.amplitude_uv)
            unit = int(
                np.clip(
                    np.searchsorted(self.amplitude_bins, amp) - 1,
                    0,
                    self.num_units - 1,
                )
            )
            result.append(
                DetectedSpike(
                    channel=s.channel,
                    timestamp_s=s.timestamp_s,
                    amplitude_uv=s.amplitude_uv,
                    unit_id=unit,
                    waveform=s.waveform,
                )
            )
        return result


# ── LFP / Oscillation Band Extraction (Gap 2) ───────────────────────


@dataclass
class LFPBand:
    """Frequency band definition for LFP extraction."""

    name: str
    low_hz: float
    high_hz: float


DEFAULT_LFP_BANDS = [
    LFPBand("delta", 0.5, 4.0),
    LFPBand("theta", 4.0, 8.0),
    LFPBand("alpha", 8.0, 13.0),
    LFPBand("beta", 13.0, 30.0),
    LFPBand("gamma", 30.0, 100.0),
]


def extract_lfp_power(
    voltage_data: np.ndarray,
    sample_rate_hz: float,
    bands: Optional[List[LFPBand]] = None,
) -> Dict[str, np.ndarray]:
    """Extract per-channel power in each LFP band.

    Uses FFT-based power spectral density estimation.
    Returns dict of band_name → per-channel power array.
    """
    if bands is None:
        bands = DEFAULT_LFP_BANDS

    n_samples, n_channels = voltage_data.shape
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate_hz)
    fft_mag = np.abs(np.fft.rfft(voltage_data, axis=0)) ** 2

    result = {}
    for band in bands:
        mask = (freqs >= band.low_hz) & (freqs < band.high_hz)
        power = np.sum(fft_mag[mask, :], axis=0) if mask.any() else np.zeros(n_channels)
        result[band.name] = power
    return result


# ── Closed-Loop Latency Budget (Gap 3) ───────────────────────────────


@dataclass
class LatencyBudget:
    """Tracks and enforces closed-loop latency requirements."""

    max_latency_us: float = 1000.0  # 1 ms default
    history: List[float] = field(default_factory=list)
    violations: int = 0

    def record(self, latency_us: float) -> bool:
        """Record a latency measurement. Returns True if within budget."""
        self.history.append(latency_us)
        if latency_us > self.max_latency_us:
            self.violations += 1
            return False
        return True

    @property
    def mean_latency_us(self) -> float:
        return float(np.mean(self.history)) if self.history else 0.0

    @property
    def p99_latency_us(self) -> float:
        return float(np.percentile(self.history, 99)) if self.history else 0.0

    @property
    def compliance_ratio(self) -> float:
        if not self.history:
            return 1.0
        return 1.0 - self.violations / len(self.history)


# ── Pharmacological Wash Model (Gap 4) ───────────────────────────────


@dataclass
class PharmModel:
    """Simulates effect of pharmacological agents on spike rate.

    Models excitatory (e.g., bicuculline) or inhibitory (e.g., TTX) agents
    as gain factors on firing rate.
    """

    agent_name: str = "none"
    gain: float = 1.0  # >1 = excitatory, <1 = inhibitory, 0 = silencing
    onset_delay_s: float = 30.0
    wash_time_s: float = 120.0
    _applied_at: float = -1.0

    def apply(self, t_current_s: float) -> None:
        self._applied_at = t_current_s

    def effective_gain(self, t_current_s: float) -> float:
        if self._applied_at < 0:
            return 1.0
        elapsed = t_current_s - self._applied_at
        if elapsed < self.onset_delay_s:
            frac = elapsed / self.onset_delay_s
            return 1.0 + frac * (self.gain - 1.0)
        return self.gain

    def modulate_spikes(self, spike_counts: np.ndarray, t_current_s: float) -> np.ndarray:
        """Modulate spike counts by pharmacological gain."""
        g = self.effective_gain(t_current_s)
        return np.round(spike_counts * g).astype(int)


# ── Multi-Well Plate Support (Gap 5) ─────────────────────────────────


@dataclass
class WellConfig:
    """One well in a multi-well MEA plate."""

    well_id: str
    mea_config: MEAConfig
    culture_type: str = "cortical"
    passage_number: int = 0

    @property
    def label(self) -> str:
        return f"{self.well_id}_{self.culture_type}_P{self.passage_number}"


@dataclass
class MultiWellPlate:
    """Multi-well plate (e.g., 6/24/48/96-well MEA plate)."""

    wells: List[WellConfig] = field(default_factory=list)

    def add_well(self, well: WellConfig) -> None:
        self.wells.append(well)

    @classmethod
    def standard_6_well(cls, layout: MEALayout = MEALayout.MEA_60) -> MultiWellPlate:
        plate = cls()
        for i in range(6):
            plate.add_well(
                WellConfig(
                    well_id=f"W{i + 1}",
                    mea_config=MEAConfig.from_layout(layout),
                )
            )
        return plate

    @property
    def num_wells(self) -> int:
        return len(self.wells)

    def get_well(self, well_id: str) -> Optional[WellConfig]:
        return next((w for w in self.wells if w.well_id == well_id), None)


# ── Network Burst Detection (Gap 6) ──────────────────────────────────


@dataclass
class NetworkBurst:
    """Detected network-wide synchronised burst event."""

    onset_s: float
    duration_s: float
    participating_channels: int
    total_spikes: int


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
    if not spikes:
        return []

    timestamps = np.array([s.timestamp_s for s in spikes])
    t_start, t_end = timestamps.min(), timestamps.max()
    if t_end <= t_start:
        return []

    n_bins = max(1, int((t_end - t_start) / bin_width_s) + 1)
    bin_counts = np.zeros(n_bins)
    bin_channels: List[set] = [set() for _ in range(n_bins)]

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


# ── Stimulation Artifact Rejection (Gap 7) ───────────────────────────


@dataclass
class ArtifactRejector:
    """Blanks stimulation artifacts from voltage data.

    Zeros the voltage trace in a window around each stimulation onset.
    """

    blanking_pre_ms: float = 0.5
    blanking_post_ms: float = 2.0

    def blank(
        self,
        voltage_data: np.ndarray,
        stim_times_s: List[float],
        sample_rate_hz: float,
    ) -> np.ndarray:
        """Return voltage data with stimulus artifacts blanked."""
        result = voltage_data.copy()
        pre_samples = int(self.blanking_pre_ms * sample_rate_hz / 1000.0)
        post_samples = int(self.blanking_post_ms * sample_rate_hz / 1000.0)

        for t_s in stim_times_s:
            center = int(t_s * sample_rate_hz)
            start = max(0, center - pre_samples)
            end = min(result.shape[0], center + post_samples)
            result[start:end, :] = 0.0
        return result


# ── Session Audit Log (Gap 8) ────────────────────────────────────────


@dataclass
class BioAuditEntry:
    """One audit entry for a bio-hybrid session."""

    round_number: int
    timestamp_iso: str
    num_spikes: int
    num_opto_pulses: int
    latency_us: float
    health_score: float
    notes: str = ""


@dataclass
class BioAuditLog:
    """Regulatory-grade audit log for bio-hybrid experiments."""

    entries: List[BioAuditEntry] = field(default_factory=list)
    experiment_id: str = ""

    def log(self, entry: BioAuditEntry) -> None:
        self.entries.append(entry)

    @property
    def total_rounds(self) -> int:
        return len(self.entries)

    def to_list(self) -> List[Dict]:
        return [
            {
                "round": e.round_number,
                "timestamp": e.timestamp_iso,
                "spikes": e.num_spikes,
                "opto_pulses": e.num_opto_pulses,
                "latency_us": e.latency_us,
                "health_score": e.health_score,
                "notes": e.notes,
            }
            for e in self.entries
        ]

    def checksum(self) -> str:
        """SHA-256 of log contents for tamper detection."""
        import json as _json

        data = _json.dumps(self.to_list(), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()


# ── SC Bitstream → Firing Rate Decoder (Gap 9) ──────────────────────


def decode_bitstream_rate(
    bitstreams: Dict[int, np.ndarray],
    sc_clock_hz: float = 1e6,
) -> Dict[int, float]:
    """Decode SC bitstreams back to biological firing rates (Hz).

    Interprets popcount/length as probability, scales by SC clock
    to get equivalent biological firing rate.
    """
    rates = {}
    for nid, bs in bitstreams.items():
        if len(bs) == 0:
            rates[nid] = 0.0
            continue
        prob = float(np.sum(bs)) / len(bs)
        rates[nid] = prob * sc_clock_hz
    return rates


# ── Homeostatic Plasticity (Gap 10) ──────────────────────────────────


@dataclass
class HomeostaticPlasticity:
    """Intrinsic excitability scaling to maintain target firing rate.

    Implements homeostatic plasticity: if a neuron fires too fast,
    reduce its excitability (threshold up); too slow, increase it.
    Operates on Q8.8 threshold values.
    """

    target_rate_hz: float = 10.0
    tau_homeo_ms: float = 10000.0  # slow timescale (seconds)
    max_threshold_q88: int = 512  # Q8.8 = 2.0
    min_threshold_q88: int = 64  # Q8.8 = 0.25

    def update_threshold(
        self,
        current_q88: int,
        observed_rate_hz: float,
        dt_ms: float,
    ) -> int:
        """Adjust threshold to drive firing rate toward target."""
        error = observed_rate_hz - self.target_rate_hz
        alpha = dt_ms / self.tau_homeo_ms
        delta_q88 = int(alpha * error * 2.56)  # scale to Q8.8
        new_q88 = current_q88 + delta_q88
        return max(self.min_threshold_q88, min(self.max_threshold_q88, new_q88))
