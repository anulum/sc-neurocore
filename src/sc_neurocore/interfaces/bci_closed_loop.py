# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Closed-loop BCI HIL template

"""Deterministic closed-loop BCI template for HIL prototyping.

The template wires raw electrode windows through WaveformCodec compression,
AER event payload generation, spike-rate decoding, feedback emission, and
runtime telemetry.  It deliberately uses an in-process implant emulator by
default so tests and examples can exercise the closed loop without claiming
access to a physical implant.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from sc_neurocore.edge.telemetry import DeviceTelemetry
from sc_neurocore.spike_codec.aer_codec import AERCompressionResult, AERSpikeCodec
from sc_neurocore.spike_codec.waveform_codec import (
    WaveformCodec,
    WaveformCompressionResult,
)


@dataclass(frozen=True)
class ClosedLoopBCIConfig:
    """Configuration for one raw-waveform to feedback HIL loop."""

    n_channels: int
    sampling_rate_hz: int = 30_000
    threshold_sigma: float = 4.5
    snippet_samples: int = 48
    waveform_mode: str = "spike"
    quantize_bits: int = 6
    timestamp_bits: int = 16
    feedback_gain: float = 1.0
    max_feedback: float = 1.0
    input_layer_id: str = "implant_input"
    feedback_layer_id: str = "implant_feedback"


@dataclass(frozen=True)
class FeedbackFrame:
    """Feedback vector emitted to an implant emulator or hardware adapter."""

    values: tuple[float, ...]
    timestamp_us: int
    active_count: int


@dataclass(frozen=True)
class ClosedLoopBCIResult:
    """One processed BCI/HIL loop window."""

    compressed_waveform: bytes
    waveform: WaveformCompressionResult
    spike_raster: np.ndarray[Any, Any]
    aer_payload: bytes
    aer: AERCompressionResult
    decoded_rates: np.ndarray[Any, Any]
    feedback: FeedbackFrame
    telemetry: dict[str, Any]


class SpikeDecoder(Protocol):
    """Decoder interface for closed-loop spike windows."""

    def decode(self, spike_raster: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Decode a binary spike raster into feedback control values."""


class FeedbackSink(Protocol):
    """Feedback interface for an implant emulator or hardware adapter."""

    def apply_feedback(self, values: np.ndarray[Any, Any], timestamp_us: int) -> FeedbackFrame:
        """Apply decoded feedback values and return the emitted frame."""


@dataclass
class RateSpikeDecoder:
    """Decode spike rasters as per-channel firing rates."""

    sampling_rate_hz: int

    def decode(self, spike_raster: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        raster = np.asarray(spike_raster, dtype=np.float32)
        if raster.ndim != 2:
            raise ValueError("spike_raster must have shape (samples, channels)")
        duration_s = max(raster.shape[0] / self.sampling_rate_hz, 1e-9)
        firing_rates: np.ndarray[Any, Any] = raster.sum(axis=0) / duration_s
        return firing_rates


@dataclass
class ImplantEmulator:
    """Deterministic feedback sink used by the closed-loop template."""

    gain: float = 1.0
    max_feedback: float = 1.0
    active_threshold: float = 1e-9
    frames: list[FeedbackFrame] = field(default_factory=list)

    def apply_feedback(self, values: np.ndarray[Any, Any], timestamp_us: int) -> FeedbackFrame:
        scaled = np.asarray(values, dtype=np.float32) * self.gain
        clipped = np.clip(scaled, -self.max_feedback, self.max_feedback)
        active = int(np.count_nonzero(np.abs(clipped) > self.active_threshold))
        frame = FeedbackFrame(
            values=tuple(float(v) for v in clipped.tolist()),
            timestamp_us=int(timestamp_us),
            active_count=active,
        )
        self.frames.append(frame)
        return frame


@dataclass
class ClosedLoopBCITemplate:
    """WaveformCodec + AER + telemetry closed-loop BCI scaffold."""

    config: ClosedLoopBCIConfig
    decoder: SpikeDecoder | None = None
    feedback_sink: FeedbackSink | None = None
    telemetry: DeviceTelemetry = field(default_factory=DeviceTelemetry)

    def __post_init__(self) -> None:
        self.waveform_codec = WaveformCodec(
            threshold_sigma=self.config.threshold_sigma,
            snippet_samples=self.config.snippet_samples,
            quantize_bits=self.config.quantize_bits,
            mode=self.config.waveform_mode,
        )
        self.aer_codec = AERSpikeCodec(timestamp_bits=self.config.timestamp_bits)
        if self.decoder is None:
            self.decoder = RateSpikeDecoder(self.config.sampling_rate_hz)
        if self.feedback_sink is None:
            self.feedback_sink = ImplantEmulator(
                gain=self.config.feedback_gain,
                max_feedback=self.config.max_feedback,
            )

    def process_window(
        self, waveform: np.ndarray[Any, Any], *, window_start_us: int = 0
    ) -> ClosedLoopBCIResult:
        """Process one raw electrode window through the closed-loop template."""
        window = self._validate_waveform(waveform)
        compressed, waveform_result = self.waveform_codec.compress(window)
        spike_raster = self._detect_spike_raster(window)
        aer_payload, aer_result = self.aer_codec.compress(spike_raster)

        if self.decoder is None or self.feedback_sink is None:
            raise RuntimeError("closed-loop BCI template was not initialised")
        decoded = self.decoder.decode(spike_raster)
        feedback = self.feedback_sink.apply_feedback(decoded, window_start_us)

        self.telemetry.record(
            self.config.input_layer_id,
            int(spike_raster.sum()),
            self.config.n_channels,
        )
        self.telemetry.record(
            self.config.feedback_layer_id,
            feedback.active_count,
            self.config.n_channels,
        )

        return ClosedLoopBCIResult(
            compressed_waveform=compressed,
            waveform=waveform_result,
            spike_raster=spike_raster,
            aer_payload=aer_payload,
            aer=aer_result,
            decoded_rates=decoded,
            feedback=feedback,
            telemetry=self.telemetry.summary(),
        )

    def _validate_waveform(self, waveform: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        window = np.asarray(waveform, dtype=np.float32)
        if window.ndim != 2:
            raise ValueError("waveform must have shape (samples, channels)")
        if window.shape[1] != self.config.n_channels:
            raise ValueError(
                f"waveform has {window.shape[1]} channels, expected {self.config.n_channels}"
            )
        if window.shape[0] == 0:
            raise ValueError("waveform must contain at least one sample")
        return window

    def _detect_spike_raster(self, waveform: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        noise_sigma = np.median(np.abs(waveform), axis=0) / 0.6745
        noise_sigma = np.maximum(noise_sigma, 1e-6)
        thresholds = -self.config.threshold_sigma * noise_sigma
        samples, channels = waveform.shape
        raster = np.zeros((samples, channels), dtype=np.int8)
        refractory = max(1, self.config.snippet_samples // 2)

        for channel in range(channels):
            last_spike = -refractory - 1
            for sample in range(1, samples):
                if (
                    waveform[sample, channel] < thresholds[channel]
                    and waveform[sample, channel] < waveform[sample - 1, channel]
                    and (sample - last_spike) > refractory
                ):
                    raster[sample, channel] = 1
                    last_spike = sample

        return raster
