# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — BCI Closed-Loop Primitives

"""Deterministic BCI closed-loop primitives for HIL prototyping.

This module provides bounded raw-signal processing, reward-modulated
adaptation, feedback packetisation, and an audit trace for each frame.  It is
research/HIL infrastructure, not medical-device control software.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    from sc_neurocore._native.learning_bridge import (
        RULE_ELIGENT,
        RustRuleLayer,
        is_available as _rust_learning_available,
    )

    FFI_ENABLED = _rust_learning_available()
except ImportError:
    FFI_ENABLED = False


SCHEMA_VERSION = "sc-neurocore.bci-closed-loop-primitive.v1"


@dataclass(frozen=True)
class BCIPrimitiveConfig:
    """Configuration for the deterministic closed-loop primitive."""

    channels: int = 1024
    sampling_rate_hz: int = 30_000
    threshold_sigma: float = 4.5
    legacy_derivative_threshold: float = 0.5
    refractory_samples: int = 16
    command_threshold_hz: float = 75.0
    legacy_active_fraction_threshold: float = 0.10
    learning_rate: float = 0.01
    weight_decay: float = 0.999
    min_weight: float = 0.01
    max_weight: float = 10.0
    feedback_gain: float = 1.0
    max_feedback_amplitude: float = 1.0
    latency_budget_ms: float = 10.0
    enable_native_learning: bool = True

    def __post_init__(self) -> None:
        if self.channels <= 0:
            raise ValueError("channels must be positive")
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if self.threshold_sigma <= 0:
            raise ValueError("threshold_sigma must be positive")
        if self.legacy_derivative_threshold <= 0:
            raise ValueError("legacy_derivative_threshold must be positive")
        if self.refractory_samples < 1:
            raise ValueError("refractory_samples must be >= 1")
        if self.command_threshold_hz < 0:
            raise ValueError("command_threshold_hz must be non-negative")
        if not 0.0 <= self.legacy_active_fraction_threshold <= 1.0:
            raise ValueError("legacy_active_fraction_threshold must be in [0, 1]")
        if self.learning_rate < 0:
            raise ValueError("learning_rate must be non-negative")
        if not 0.0 < self.weight_decay <= 1.0:
            raise ValueError("weight_decay must be in (0, 1]")
        if not 0.0 < self.min_weight <= self.max_weight:
            raise ValueError("min_weight must be positive and <= max_weight")
        if self.max_feedback_amplitude <= 0:
            raise ValueError("max_feedback_amplitude must be positive")
        if self.latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms must be positive")


@dataclass(frozen=True)
class BCIFrame:
    """One raw neural signal frame."""

    samples: np.ndarray[Any, Any]
    reward: float = 0.0
    timestamp_us: int = 0
    frame_id: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BCIFeedbackCommand:
    """Feedback command emitted by the primitive.

    Packet layout is 24 bytes: `[schema:u16, command:u8, flags:u8,
    channel:u16, reserved:u16, amplitude:f32, timestamp_us:u64, score:f32]`.
    """

    COMMAND_NOP = 0
    COMMAND_STIM = 1

    command: int
    channel: int
    amplitude: float
    timestamp_us: int
    score: float
    safety_limited: bool = False

    def to_packet(self) -> bytes:
        flags = 1 if self.safety_limited else 0
        return struct.pack(
            "<HBBHHfQf",
            1,
            self.command,
            flags,
            self.channel,
            0,
            self.amplitude,
            self.timestamp_us,
            self.score,
        )

    @classmethod
    def from_packet(cls, packet: bytes) -> "BCIFeedbackCommand":
        if len(packet) < 24:
            raise ValueError("BCI feedback packet must be at least 24 bytes")
        schema, command, flags, channel, _reserved, amplitude, timestamp_us, score = struct.unpack(
            "<HBBHHfQf", packet[:24]
        )
        if schema != 1:
            raise ValueError(f"unsupported BCI feedback packet schema {schema}")
        return cls(
            command=command,
            channel=channel,
            amplitude=float(amplitude),
            timestamp_us=int(timestamp_us),
            score=float(score),
            safety_limited=bool(flags & 1),
        )


@dataclass(frozen=True)
class BCIClosedLoopTrace:
    """Audit trace for one processed frame."""

    schema_version: str
    frame_id: int
    input_shape: tuple[int, ...]
    spike_count: int
    active_channels: int
    score: float
    command: int
    latency_ms: float
    latency_budget_ms: float
    latency_budget_met: bool
    adaptation_applied: bool
    ffi_accelerated: bool
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "frame_id": self.frame_id,
            "input_shape": list(self.input_shape),
            "spike_count": self.spike_count,
            "active_channels": self.active_channels,
            "score": self.score,
            "command": self.command,
            "latency_ms": self.latency_ms,
            "latency_budget_ms": self.latency_budget_ms,
            "latency_budget_met": self.latency_budget_met,
            "adaptation_applied": self.adaptation_applied,
            "ffi_accelerated": self.ffi_accelerated,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class BCIPrimitiveResult:
    """Result from one closed-loop primitive step."""

    command: BCIFeedbackCommand
    feedback_packet: bytes
    spikes: np.ndarray[Any, Any]
    channel_spike_counts: np.ndarray[Any, Any]
    score: float
    latency_ms: float
    trace: BCIClosedLoopTrace

    def as_legacy_dict(self) -> dict[str, Any]:
        return {
            "command": self.command.command,
            "latency_ms": self.latency_ms,
            "spikes": int(self.spikes.sum()),
            "active_channels": int(np.count_nonzero(self.channel_spike_counts)),
            "score": self.score,
            "feedback_bytes": len(self.feedback_packet),
            "latency_budget_met": self.trace.latency_budget_met,
            "trace": self.trace.as_dict(),
        }


class BCIClosedLoopPrimitive:
    """Deterministic raw-signal to feedback primitive with audit trace."""

    def __init__(
        self,
        config: BCIPrimitiveConfig | None = None,
        *,
        initial_weights: np.ndarray[Any, Any] | None = None,
    ) -> None:
        self.config = config or BCIPrimitiveConfig()
        if initial_weights is None:
            self.weights = np.ones(self.config.channels, dtype=np.float32)
        else:
            weights = np.asarray(initial_weights, dtype=np.float32)
            if weights.shape != (self.config.channels,):
                raise ValueError(
                    f"initial_weights shape {weights.shape} does not match "
                    f"({self.config.channels},)"
                )
            self.weights = weights.copy()
        self._frame_counter = 0

        if FFI_ENABLED and self.config.enable_native_learning:
            self.layer: RustRuleLayer | None = RustRuleLayer(
                self.config.channels,
                RULE_ELIGENT,
                weight=1.0,
                param_a=self.config.learning_rate,
                param_b=1.0,
            )
        else:
            self.layer = None

    def process_frame(self, frame: BCIFrame) -> BCIPrimitiveResult:
        start_time = time.perf_counter()
        samples, notes = self._validate_samples(frame.samples)
        spikes, channel_counts = self._extract_spikes(samples)
        score = self._score(channel_counts, samples.shape[0])
        command = self._build_command(score, frame.timestamp_us)
        adaptation = self._adapt(channel_counts, command.command, frame.reward)
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        frame_id = self._next_frame_id(frame.frame_id)

        trace = BCIClosedLoopTrace(
            schema_version=SCHEMA_VERSION,
            frame_id=frame_id,
            input_shape=tuple(int(v) for v in samples.shape),
            spike_count=int(spikes.sum()),
            active_channels=int(np.count_nonzero(channel_counts)),
            score=score,
            command=command.command,
            latency_ms=latency_ms,
            latency_budget_ms=self.config.latency_budget_ms,
            latency_budget_met=latency_ms <= self.config.latency_budget_ms,
            adaptation_applied=adaptation,
            ffi_accelerated=self.layer is not None,
            notes=tuple(notes),
        )
        return BCIPrimitiveResult(
            command=command,
            feedback_packet=command.to_packet(),
            spikes=spikes,
            channel_spike_counts=channel_counts,
            score=score,
            latency_ms=latency_ms,
            trace=trace,
        )

    def _next_frame_id(self, explicit: int | None) -> int:
        if explicit is not None:
            self._frame_counter = max(self._frame_counter, explicit + 1)
            return explicit
        frame_id = self._frame_counter
        self._frame_counter += 1
        return frame_id

    def _validate_samples(
        self, samples: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], list[str]]:
        data = np.asarray(samples, dtype=np.float32)
        if not np.all(np.isfinite(data)):
            raise ValueError("BCI frame contains non-finite values")
        notes: list[str] = []
        if data.ndim == 1:
            if data.shape[0] != self.config.channels:
                raise ValueError(
                    f"1D BCI frame has {data.shape[0]} channels, expected {self.config.channels}"
                )
            notes.append("legacy_vector_frame")
            return data.reshape(1, self.config.channels), notes
        if data.ndim != 2:
            raise ValueError("BCI frame samples must have shape (channels,) or (samples, channels)")
        if data.shape[0] == 0:
            raise ValueError("BCI frame must contain at least one sample")
        if data.shape[1] != self.config.channels:
            raise ValueError(
                f"BCI frame has {data.shape[1]} channels, expected {self.config.channels}"
            )
        return data, notes

    def _extract_spikes(
        self, samples: np.ndarray[Any, Any]
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        if samples.shape[0] == 1:
            vector = samples[0]
            diffs = np.abs(np.diff(vector, prepend=0.0))
            spikes = (diffs > self.config.legacy_derivative_threshold).reshape(
                1, self.config.channels
            )
            return spikes.astype(np.int8), spikes.sum(axis=0).astype(np.float32)

        noise_sigma = np.median(np.abs(samples), axis=0) / 0.6745
        noise_sigma = np.maximum(noise_sigma, 1e-6)
        thresholds = -self.config.threshold_sigma * noise_sigma
        spikes = np.zeros(samples.shape, dtype=np.int8)
        for channel in range(self.config.channels):
            last_spike = -self.config.refractory_samples - 1
            for sample_idx in range(1, samples.shape[0]):
                if (
                    samples[sample_idx, channel] < thresholds[channel]
                    and samples[sample_idx, channel] < samples[sample_idx - 1, channel]
                    and (sample_idx - last_spike) > self.config.refractory_samples
                ):
                    spikes[sample_idx, channel] = 1
                    last_spike = sample_idx
        return spikes, spikes.sum(axis=0).astype(np.float32)

    def _score(self, channel_counts: np.ndarray[Any, Any], n_samples: int) -> float:
        if n_samples <= 1:
            return float(np.dot(channel_counts, self.weights) / self.config.channels)
        duration_s = n_samples / self.config.sampling_rate_hz
        rates_hz = channel_counts / max(duration_s, 1e-9)
        return float(np.dot(rates_hz, self.weights) / self.config.channels)

    def _build_command(self, score: float, timestamp_us: int) -> BCIFeedbackCommand:
        if score <= 0.0:
            return BCIFeedbackCommand(
                command=BCIFeedbackCommand.COMMAND_NOP,
                channel=0,
                amplitude=0.0,
                timestamp_us=timestamp_us,
                score=score,
            )
        if score <= 1.0:
            threshold = self.config.legacy_active_fraction_threshold
            command = int(score > threshold)
            raw_amplitude = score / max(threshold, 1e-9) if command else 0.0
        else:
            command = int(score >= self.config.command_threshold_hz)
            raw_amplitude = score / max(self.config.command_threshold_hz, 1e-9) if command else 0.0
        amplitude = raw_amplitude * self.config.feedback_gain
        clipped = float(np.clip(amplitude, 0.0, self.config.max_feedback_amplitude))
        return BCIFeedbackCommand(
            command=command,
            channel=0,
            amplitude=clipped,
            timestamp_us=timestamp_us,
            score=score,
            safety_limited=amplitude != clipped,
        )

    def _adapt(self, channel_counts: np.ndarray[Any, Any], command: int, reward: float) -> bool:
        if reward == 0.0 and self.config.weight_decay == 1.0:
            return False
        spike_mask = channel_counts > 0
        if self.layer is not None:
            pre_spikes = spike_mask.astype(np.bool_)
            post_spikes = np.full(self.config.channels, command > 0, dtype=np.bool_)
            rewards = np.full(self.config.channels, reward, dtype=np.float32)
            self.layer.step(pre_spikes, post_spikes, rewards)
            self.weights = self.layer.get_weights().astype(np.float32)
            return True

        old = self.weights.copy()
        self.weights *= self.config.weight_decay
        if reward != 0.0:
            self.weights[spike_mask] += self.config.learning_rate * reward
            self.weights[~spike_mask] -= self.config.learning_rate * reward * 0.1
        self.weights = np.clip(self.weights, self.config.min_weight, self.config.max_weight)
        return bool(np.any(np.abs(self.weights - old) > 1e-9))


class BCIClosedLoopEngine:
    """Backward-compatible wrapper around :class:`BCIClosedLoopPrimitive`."""

    def __init__(self, channels: int = 1024):
        self.channels = channels
        self.primitive = BCIClosedLoopPrimitive(BCIPrimitiveConfig(channels=channels))

    @property
    def weights(self) -> np.ndarray[Any, Any]:
        return self.primitive.weights

    def process_bci_frame(self, raw_ephys: np.ndarray[Any, Any], reward: float) -> dict[str, Any]:
        result = self.primitive.process_frame(BCIFrame(samples=raw_ephys, reward=reward))
        return result.as_legacy_dict()


if __name__ == "__main__":
    engine = BCIClosedLoopEngine()
    data = np.random.default_rng(42).normal(size=1024).astype(np.float32)
    result = engine.process_bci_frame(data, reward=1.0)
    print(
        f"BCI frame: command={result['command']}, "
        f"latency={result['latency_ms']:.4f} ms, "
        f"budget_met={result['latency_budget_met']}"
    )
