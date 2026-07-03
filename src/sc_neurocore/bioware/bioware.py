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
import math
import time
from dataclasses import dataclass, field, fields, replace
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from sc_neurocore.arcane_zenith import ArcaneZenithCognitiveCore

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
        presets: Dict[MEALayout, Dict[str, Any]] = {
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
    waveform: Optional[np.ndarray[Any, Any]] = None


@dataclass
class SpikeDetector:
    """Threshold-based spike detector for MEA voltage traces.

    Uses adaptive threshold: threshold = mean ± sigma * noise_estimate
    where noise_estimate = median(|x|) / 0.6745 (robust RMS).
    Supports configurable refractory period to prevent double-counting.
    """

    config: MEAConfig
    refractory_samples: int = 30
    _noise_estimates: Optional[np.ndarray[Any, Any]] = field(default=None, repr=False)

    def estimate_noise(self, voltage_data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Estimate per-channel noise from voltage data.

        Uses median absolute deviation (MAD) for robustness against spikes.
        voltage_data: shape (num_samples, num_channels)
        """
        mad: np.ndarray[Any, Any] = np.median(np.abs(voltage_data), axis=0) / 0.6745
        self._noise_estimates = mad
        return mad

    def detect(
        self, voltage_data: np.ndarray[Any, Any], snippet_ms: float = 2.0
    ) -> List[DetectedSpike]:
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

                # Extract waveform snippet
                half = int(snippet_ms * self.config.sample_rate_hz / 2000.0)
                start = max(0, idx - half)
                end = min(n_samples, idx + half)

                # Pad if too close to edges. The raw slice length is at most
                # 2*half == target_len (it is min(n, idx+half) - max(0, idx-half)),
                # and for an edge spike (idx < half) the slice starts at 0 so
                # pad_before = half - idx exactly closes the gap: pad_after is
                # never negative and the padded waveform is exactly target_len.
                raw_wave = voltage_data[start:end, ch].copy()
                target_len = int(2 * half)
                if len(raw_wave) < target_len:
                    pad_before = max(0, half - idx)
                    pad_after = max(0, target_len - len(raw_wave) - pad_before)
                    raw_wave = np.pad(raw_wave, (pad_before, pad_after), "constant")

                spikes.append(
                    DetectedSpike(
                        channel=ch,
                        timestamp_s=ts,
                        amplitude_uv=amp,
                        unit_id=ch,
                        waveform=raw_wave,
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

    def convert(self, events: List[AEREvent]) -> Dict[int, np.ndarray[Any, Any]]:
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

    def _lfsr_encode(self, probability: float, neuron_id: int) -> np.ndarray[Any, Any]:
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
        bitstreams: Dict[int, np.ndarray[Any, Any]],
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
            return float(self.a_plus * np.exp(-dt_ms / self.tau_plus_ms))
        elif dt_ms < 0:
            return float(-self.a_minus * np.exp(dt_ms / self.tau_minus_ms))
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

    def assess(self, spike_counts: np.ndarray[Any, Any], duration_s: float) -> Dict[str, float]:
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
class BioHybridFrameResult:
    """Strictly typed output packet detailing a full closed-loop step.

    Behaves both as a dataclass (``result.round``) and, for backward
    compatibility with pre-dataclass callers, as a mapping view of its
    fields (``result["round"]``, ``"latency_us" in result``,
    ``dict(result)``). The mapping surface is read-only.
    """

    round: int
    num_spikes: int
    num_aer_events: int
    num_bitstreams: int
    num_opto_pulses: int
    latency_us: float
    health: Dict[str, Any]
    spikes: List[DetectedSpike]
    aer_events: List[AEREvent]
    bitstreams: Dict[int, np.ndarray[Any, Any]]
    opto_pulses: List[OptogeneticPulse]

    def __getitem__(self, key: str) -> Any:
        if not isinstance(key, str) or key.startswith("_"):
            raise KeyError(key)
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        return key in {f.name for f in fields(self)}

    def keys(self) -> List[str]:
        return [f.name for f in fields(self)]


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
    artifact_rejector: Optional["ArtifactRejector"] = None
    pharm_model: Optional["PharmModel"] = None
    latency_budget: Optional["LatencyBudget"] = None
    homeostatic: Optional["HomeostaticPlasticity"] = None
    sorter: Optional["SpikeSorter"] = None
    zenith_core: Optional["ArcaneZenithCognitiveCore"] = None
    round_count: int = 0

    def process_frame(
        self,
        voltage_data: np.ndarray[Any, Any],
        t_start_s: float = 0.0,
        stim_times_s: Optional[List[float]] = None,
    ) -> BioHybridFrameResult:
        """Process one MEA data frame through the full pipeline."""
        t0 = time.perf_counter_ns()
        self.round_count += 1

        if self.artifact_rejector is not None and stim_times_s is not None:
            voltage_data = self.artifact_rejector.blank(
                voltage_data, stim_times_s, self.mea_config.sample_rate_hz
            )

        # 1. Detect spikes
        spikes = self.detector.detect(voltage_data)

        # 1.5 Core primitive wiring
        if self.sorter is not None:
            spikes = self.sorter.assign(spikes)

        if self.pharm_model is not None:
            spikes = self.pharm_model.modulate_spike_events(spikes, t_start_s)

        # 2. Transcode to AER
        aer_events = self.transcoder.transcode(spikes, t_start_s)

        # 3. Convert to SC bitstreams
        bitstreams = self.sc_converter.convert(aer_events)

        # 3.5 Zenith integration!
        if self.zenith_core is not None:
            rates = decode_bitstream_rate(bitstreams)
            self.zenith_core.step_from_bio_rates(rates)

        # 4. Generate optogenetic pulses
        opto_pulses = self.opto_encoder.encode(bitstreams)

        # 5. Health assessment
        n_channels = voltage_data.shape[1]
        spike_counts = np.zeros(n_channels)
        for s in spikes:
            if s.channel < n_channels:
                spike_counts[s.channel] += 1
        duration = voltage_data.shape[0] / self.mea_config.sample_rate_hz
        health = self.health_monitor.assess(spike_counts, duration_s=duration)

        latency_us = (time.perf_counter_ns() - t0) / 1000.0

        if self.latency_budget is not None:
            self.latency_budget.record(latency_us)

        return BioHybridFrameResult(
            round=self.round_count,
            num_spikes=len(spikes),
            num_aer_events=len(aer_events),
            num_bitstreams=len(bitstreams),
            num_opto_pulses=len(opto_pulses),
            latency_us=latency_us,
            health=health,
            spikes=spikes,
            aer_events=aer_events,
            bitstreams=bitstreams,
            opto_pulses=opto_pulses,
        )


# ── Spike Sorter — template matching (Gap 1) ────────────────────────


@dataclass
class SpikeSorter:
    """Research spike sorter using PCA feature extraction and K-Means clustering.

    Extracts the dominant principal components from the input raw waveforms, and cleanly
    separates units. Handles missing datasets explicitly natively. Requires `scikit-learn` to execute correctly.
    """

    num_units: int = 4
    n_components: int = 3
    _pca: Any = field(default=None, repr=False)
    _kmeans: Any = field(default=None, repr=False)

    def fit(self, spikes: List[DetectedSpike]) -> None:
        """Fit PCA and KMeans models sequentially on available waveforms.

        Silently no-ops (leaves ``_pca``/``_kmeans`` as ``None``) when
        fewer than ``num_units`` waveforms are present — sklearn is only
        imported in the path that actually needs it, so empty or
        amplitude-only spike lists don't require scikit-learn.
        """
        waveforms = [s.waveform for s in spikes if s.waveform is not None]
        if len(waveforms) < self.num_units:
            self._pca = None
            self._kmeans = None
            return

        try:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA
        except ImportError as exc:
            raise ImportError(
                "SpikeSorter.fit requires scikit-learn to cluster waveforms. "
                "Install with `pip install scikit-learn` or "
                "`pip install 'sc-neurocore[bioware]'`."
            ) from exc

        waves_array = np.vstack(waveforms)
        self._pca = PCA(n_components=min(self.n_components, len(waveforms), waves_array.shape[1]))
        features = self._pca.fit_transform(waves_array)

        self._kmeans = KMeans(n_clusters=self.num_units, n_init=10)
        self._kmeans.fit(features)

    def assign(self, spikes: List[DetectedSpike]) -> List[DetectedSpike]:
        """Assign cluster IDs based on PCA feature projections."""
        if self._pca is None or self._kmeans is None:
            return spikes

        result = []
        for s in spikes:
            if s.waveform is None:
                result.append(s)
                continue

            features = self._pca.transform(s.waveform.reshape(1, -1))
            unit = int(self._kmeans.predict(features)[0])

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

    def modulate_spikes(
        self, spike_counts: np.ndarray[Any, Any], t_current_s: float
    ) -> np.ndarray[Any, Any]:
        """Modulate spike counts by pharmacological gain."""
        g = self.effective_gain(t_current_s)
        return np.round(spike_counts * g).astype(int)

    def modulate_spike_events(
        self,
        spikes: List[DetectedSpike],
        t_current_s: float,
    ) -> List[DetectedSpike]:
        """Apply pharmacological rate gain to spike events.

        Inhibitory gains deterministically thin events across the observed
        response span instead of truncating the head of the frame. Excitatory
        gains preserve observed events and insert synthetic events inside the
        observed temporal support, using nearest observed spikes as channel,
        unit, amplitude, and waveform templates.
        """
        if not spikes:
            return []

        gain = self.effective_gain(t_current_s)
        if not math.isfinite(gain) or gain < 0.0:
            raise ValueError("pharmacological gain must be finite and >= 0")

        ordered = sorted(spikes, key=lambda s: (s.timestamp_s, s.channel, s.unit_id))
        target_count = int(round(len(ordered) * gain))
        if target_count <= 0:
            return []
        if target_count == len(ordered):
            return list(ordered)
        if target_count < len(ordered):
            indices = _quantile_indices(len(ordered), target_count)
            return [ordered[i] for i in indices]

        extra = target_count - len(ordered)
        timestamps = np.array([s.timestamp_s for s in ordered], dtype=float)
        if not np.all(np.isfinite(timestamps)):
            raise ValueError("detected spike timestamps must be finite")

        if len(ordered) == 1 or timestamps[-1] <= timestamps[0]:
            synthetic = [
                _clone_spike(ordered[0], timestamp_s=float(timestamps[0])) for _ in range(extra)
            ]
        else:
            insert_times = np.linspace(timestamps[0], timestamps[-1], extra + 2)[1:-1]
            synthetic = []
            for t in insert_times:
                # insert_times are strictly interior to (timestamps[0],
                # timestamps[-1]) — the linspace endpoints are dropped — so for
                # t < timestamps[-1] a left-side searchsorted always yields
                # idx <= len(ordered) - 1; no upper clamp is reachable.
                idx = int(np.searchsorted(timestamps, t, side="left"))
                if idx > 0 and abs(timestamps[idx - 1] - t) <= abs(timestamps[idx] - t):
                    idx -= 1
                synthetic.append(_clone_spike(ordered[idx], timestamp_s=float(t)))

        return sorted([*ordered, *synthetic], key=lambda s: (s.timestamp_s, s.channel, s.unit_id))


def _quantile_indices(n_items: int, target_count: int) -> List[int]:
    if target_count <= 0:
        return []
    if target_count >= n_items:
        return list(range(n_items))
    if target_count == 1:
        return [0]
    return [int(i) for i in np.rint(np.linspace(0, n_items - 1, target_count)).astype(int)]


def _clone_spike(template: DetectedSpike, *, timestamp_s: float) -> DetectedSpike:
    waveform = None if template.waveform is None else template.waveform.copy()
    return replace(template, timestamp_s=timestamp_s, waveform=waveform)


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
        voltage_data: np.ndarray[Any, Any],
        stim_times_s: List[float],
        sample_rate_hz: float,
    ) -> np.ndarray[Any, Any]:
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

    def to_list(self) -> List[Dict[str, Any]]:
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
        try:
            import orjson

            data = orjson.dumps(self.to_list(), option=orjson.OPT_SORT_KEYS)
        except ImportError:
            import json as _json

            data = _json.dumps(self.to_list(), sort_keys=True).encode("utf-8")
        return hashlib.sha256(data).hexdigest()


# ── SC Bitstream → Firing Rate Decoder (Gap 9) ──────────────────────

# Multi-language acceleration backends (Rust via PyO3, Julia, Mojo, Go)
# are planned for v4.0. The pure-Python implementation is the formal,
# fully-tested reference. All performance-critical paths are designed
# to be easily ported later.


def decode_bitstream_rate(
    bitstreams: Dict[int, np.ndarray[Any, Any]],
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
        """Adjust threshold to drive firing rate toward target.

        Proportional homeostatic controller on a Q8.8 fixed-point
        threshold. ``alpha = dt_ms / tau_homeo_ms`` is the integration
        weight over the time step; the rate error (``observed − target``)
        is scaled by ``alpha·256`` so that a 1 Hz error integrated over
        one full time-constant shifts the threshold by 1.0 Q8.8 unit
        (i.e. by ``256`` in integer representation). Result clamped to
        ``[min_threshold_q88, max_threshold_q88]``.
        """
        error = observed_rate_hz - self.target_rate_hz
        alpha = dt_ms / self.tau_homeo_ms
        delta_q88 = int(alpha * error * 256.0)
        new_q88 = current_q88 + delta_q88
        return max(self.min_threshold_q88, min(self.max_threshold_q88, new_q88))


# ── Evo Substrate Bridge (Gap 11) ───────────────────────────────────


def mea_fitness_hook(
    detected_spikes: List[DetectedSpike],
    target_rate: float = 10.0,
    *,
    duration_s: Optional[float] = None,
    stimulus_time_s: Optional[float] = None,
    measured_latency_ms: Optional[float] = None,
) -> Dict[str, float]:
    """Organism fitness metrics derived from MEA response dynamics.

    Designed to plug into the evo_substrate
    ``ReplicationEngine(metrics_fn=mea_fitness_hook)`` — returns the
    ``{"accuracy", "energy_mw", "latency_ms"}`` triple the engine scores.

    Accuracy is a bounded distance to the target mean per-channel firing
    rate when ``duration_s`` is supplied, or to the legacy per-channel
    spike count when it is omitted. ``energy_mw`` remains the documented
    spike-count proxy (0.5 mW / spike). ``latency_ms`` is either a caller
    supplied closed-loop measurement, the first response latency after
    ``stimulus_time_s``, or the first spike timestamp relative to frame
    start.
    """
    if duration_s is not None and (not math.isfinite(duration_s) or duration_s <= 0.0):
        raise ValueError("duration_s must be finite and > 0 when provided")
    if stimulus_time_s is not None and not math.isfinite(stimulus_time_s):
        raise ValueError("stimulus_time_s must be finite when provided")
    if measured_latency_ms is not None:
        if not math.isfinite(measured_latency_ms) or measured_latency_ms < 0.0:
            raise ValueError("measured_latency_ms must be finite and >= 0 when provided")

    if not detected_spikes:
        return {"accuracy": 0.1, "energy_mw": 0.0, "latency_ms": 0.0}

    counts: Dict[int, float] = {}
    for s in detected_spikes:
        counts[s.channel] = counts.get(s.channel, 0.0) + 1.0

    per_channel_activity = np.array(list(counts.values()), dtype=float)
    if duration_s is not None:
        per_channel_activity = per_channel_activity / duration_s
    mean_rate = float(np.mean(per_channel_activity)) if per_channel_activity.size else 0.0

    # Normalised distance to target rate → accuracy ∈ [0.1, 0.99].
    if target_rate > 0.0:
        accuracy = 1.0 - min(1.0, abs(mean_rate - target_rate) / target_rate)
    else:
        accuracy = 0.1

    latency_ms = _mea_response_latency_ms(
        detected_spikes,
        stimulus_time_s=stimulus_time_s,
        measured_latency_ms=measured_latency_ms,
    )
    return {
        "accuracy": float(np.clip(accuracy, 0.1, 0.99)),
        "energy_mw": float(len(detected_spikes) * 0.5),
        "latency_ms": latency_ms,
    }


def _mea_response_latency_ms(
    detected_spikes: List[DetectedSpike],
    *,
    stimulus_time_s: Optional[float],
    measured_latency_ms: Optional[float],
) -> float:
    if measured_latency_ms is not None:
        return float(measured_latency_ms)

    timestamps = np.array([s.timestamp_s for s in detected_spikes], dtype=float)
    if timestamps.size == 0:
        return 0.0
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("detected spike timestamps must be finite")

    if stimulus_time_s is not None:
        responses = timestamps[timestamps >= stimulus_time_s]
        if responses.size == 0:
            return 0.0
        return float((np.min(responses) - stimulus_time_s) * 1000.0)

    return float(max(0.0, np.min(timestamps)) * 1000.0)
