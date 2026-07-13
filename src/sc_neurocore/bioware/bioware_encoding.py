# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — AER, stochastic-bitstream, and optical encoding

"""AER, stochastic-bitstream, and optogenetic encoding boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from .bioware_contracts import AEREvent, DetectedSpike, OptogeneticPulse
from .bioware_validation import (
    require_finite,
    require_nonnegative,
    require_nonnegative_int,
    require_positive,
    require_positive_int,
    validate_binary_bitstream,
)


@dataclass
class MEAToAERTranscoder:
    """Converts MEA spike events to AER events for hardware.

    Maps biological electrode channels to AER neuron IDs,
    converting real-time timestamps to hardware clock ticks.
    """

    hw_clock_hz: float = 1e6  # 1 MHz default AER clock
    channel_map: Optional[Dict[int, int]] = None  # electrode → neuron_id

    def __post_init__(self) -> None:
        """Validate the hardware clock and optional channel mapping."""
        require_positive(self.hw_clock_hz, "hw_clock_hz")
        if self.channel_map is None:
            return
        for channel, neuron_id in self.channel_map.items():
            require_nonnegative_int(channel, "channel_map channel")
            require_nonnegative_int(neuron_id, "channel_map neuron_id")

    def transcode(
        self,
        spikes: List[DetectedSpike],
        t_start_s: float = 0.0,
    ) -> List[AEREvent]:
        """Convert spikes in one 16-bit hardware-clock epoch to AER events.

        ``DetectedSpike.timestamp_s`` and ``t_start_s`` must use the same time
        origin. The method rejects events outside the representable epoch;
        callers must split longer recordings instead of accepting timestamp
        wraparound and the resulting loss of temporal ordering.
        """
        require_nonnegative(t_start_s, "t_start_s")
        events = []
        for spike in spikes:
            neuron_id = self._map_channel(spike.channel)
            relative_s = spike.timestamp_s - t_start_s
            if relative_s < 0.0:
                raise ValueError("spike timestamp precedes t_start_s")
            ticks = relative_s * self.hw_clock_hz
            if not np.isfinite(ticks) or ticks > 0xFFFF:
                raise ValueError("spike timestamp does not fit the 16-bit AER window")
            ts_hw = int(ticks)
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


@dataclass
class AERToSCConverter:
    """Converts AER event streams to SC bitstreams.

    Uses a time-windowed rate code: count events per neuron per window,
    then LFSR-encode the resulting firing probabilities.
    """

    window_ticks: int = 0x10000
    bitstream_length: int = 256
    num_neurons: int = 128
    lfsr_seed: int = 0xACE1

    def __post_init__(self) -> None:
        """Validate window, bitstream, neuron, and LFSR boundaries."""
        require_positive_int(self.window_ticks, "window_ticks")
        require_positive_int(self.bitstream_length, "bitstream_length")
        require_positive_int(self.num_neurons, "num_neurons")
        require_nonnegative_int(self.lfsr_seed, "lfsr_seed")
        if self.lfsr_seed > 0xFFFF:
            raise ValueError("lfsr_seed must fit 16 bits")

    def convert(self, events: List[AEREvent]) -> Dict[int, np.ndarray[Any, Any]]:
        """Convert AER events to per-neuron SC bitstreams."""
        # Count events per neuron in the window
        counts: Dict[int, int] = {}
        for e in events:
            if e.valid:
                if e.neuron_id >= self.num_neurons:
                    raise ValueError(
                        f"AER neuron_id {e.neuron_id} is outside num_neurons={self.num_neurons}"
                    )
                if e.timestamp >= self.window_ticks:
                    raise ValueError(
                        f"AER timestamp {e.timestamp} is outside window_ticks={self.window_ticks}"
                    )
                counts[e.neuron_id] = counts.get(e.neuron_id, 0) + 1

        max_count = max(counts.values()) if counts else 1
        bitstreams = {}
        for nid, count in counts.items():
            prob = count / max_count
            bitstreams[nid] = self._lfsr_encode(prob, nid)
        return bitstreams

    def _lfsr_encode(self, probability: float, neuron_id: int) -> np.ndarray[Any, Any]:
        """LFSR-16 encoding (bit-compatible with core_engine)."""
        require_finite(probability, "probability")
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        require_nonnegative_int(neuron_id, "neuron_id")
        if neuron_id >= self.num_neurons:
            raise ValueError("neuron_id must be smaller than num_neurons")
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
    illuminated_area_mm2: float = 1.0

    def __post_init__(self) -> None:
        """Validate optical timing, irradiance, area, and power limits."""
        require_nonnegative(self.max_intensity_mw_mm2, "max_intensity_mw_mm2")
        require_positive(self.min_pulse_ms, "min_pulse_ms")
        require_positive(self.max_pulse_ms, "max_pulse_ms")
        if self.max_pulse_ms < self.min_pulse_ms:
            raise ValueError("max_pulse_ms must be >= min_pulse_ms")
        require_positive_int(self.wavelength_nm, "wavelength_nm")
        require_nonnegative(self.clock_period_ms, "clock_period_ms")
        require_positive(self.max_total_power_mw, "max_total_power_mw")
        require_positive(self.illuminated_area_mm2, "illuminated_area_mm2")

    def encode(
        self,
        bitstreams: Dict[int, np.ndarray[Any, Any]],
        t_start_ms: float = 0.0,
    ) -> List[OptogeneticPulse]:
        """Convert SC bitstreams to optogenetic pulses."""
        require_nonnegative(t_start_ms, "t_start_ms")
        pulses = []
        total_power = 0.0
        for nid, bs in sorted(bitstreams.items()):
            require_nonnegative_int(nid, "bitstream neuron_id")
            validate_binary_bitstream(bs, name=f"bitstreams[{nid}]", allow_empty=True)
            density = float(np.sum(bs)) / len(bs) if len(bs) > 0 else 0.0
            if density < 0.01:
                continue

            intensity = density * self.max_intensity_mw_mm2
            power_mw = intensity * self.illuminated_area_mm2
            if total_power + power_mw > self.max_total_power_mw:
                continue
            total_power += power_mw

            duration = self.min_pulse_ms + density * (self.max_pulse_ms - self.min_pulse_ms)
            onset = t_start_ms + nid * self.clock_period_ms

            pulses.append(
                OptogeneticPulse(
                    channel=nid,
                    onset_ms=onset,
                    duration_ms=duration,
                    intensity_mw_mm2=intensity,
                    wavelength_nm=self.wavelength_nm,
                    illuminated_area_mm2=self.illuminated_area_mm2,
                )
            )
        return pulses


def decode_bitstream_rate(
    bitstreams: Dict[int, np.ndarray[Any, Any]],
    sc_clock_hz: float = 1e6,
) -> Dict[int, float]:
    """Decode SC bitstreams back to biological firing rates (Hz).

    Interprets popcount/length as probability, scales by SC clock
    to get equivalent biological firing rate.
    """
    require_positive(sc_clock_hz, "sc_clock_hz")
    rates = {}
    for nid, bs in bitstreams.items():
        require_nonnegative_int(nid, "bitstream neuron_id")
        validate_binary_bitstream(bs, name=f"bitstreams[{nid}]", allow_empty=True)
        if len(bs) == 0:
            rates[nid] = 0.0
            continue
        prob = float(np.sum(bs)) / len(bs)
        rates[nid] = prob * sc_clock_hz
    return rates
