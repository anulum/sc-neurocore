# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — BCI Studio

"""BCI Studio orchestrator for real-time closed-loop brain-computer interfaces.

Pipeline: raw_ephys → codec → spike_extract → SC_decode → learner → feedback
Includes SC-domain lossy compression, online STDP learning, FPGA feedback
serialization, and latency profiling.
"""

from __future__ import annotations

import struct
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class SessionMetrics:
    total_frames: int = 0
    total_spikes: int = 0
    latency_history: List[float] = field(default_factory=list)
    adaptation_events: int = 0

    @property
    def mean_latency_ms(self) -> float:
        return float(np.mean(self.latency_history)) if self.latency_history else 0.0

    @property
    def p95_latency_ms(self) -> float:
        return float(np.percentile(self.latency_history, 95)) if self.latency_history else 0.0

    @property
    def spike_rate(self) -> float:
        return self.total_spikes / max(1, self.total_frames)

    def summary(self) -> str:
        return (
            f"Frames: {self.total_frames}, "
            f"Spikes: {self.total_spikes}, "
            f"Rate: {self.spike_rate:.2f}/frame, "
            f"Latency: {self.mean_latency_ms:.3f} ms (p95={self.p95_latency_ms:.3f} ms), "
            f"Adaptations: {self.adaptation_events}"
        )


class SpikeCodec:
    """SC-domain lossy compression for neural data streams.

    Uses run-length encoding on spike trains with delta-time encoding.
    """

    def encode(self, spikes: np.ndarray) -> bytes:
        """Compress boolean spike array to RLE byte stream.

        Format: [total_len:u32_le] + N × [value:u8, count:u8]
        """
        if len(spikes) == 0:
            return b""
        runs: List[Tuple[int, int]] = []
        current = int(spikes[0])
        count = 1
        for i in range(1, len(spikes)):
            if int(spikes[i]) == current and count < 255:
                count += 1
            else:
                runs.append((current, count))
                current = int(spikes[i])
                count = 1
        runs.append((current, count))

        data = bytearray()
        data.extend(struct.pack("<I", len(spikes)))
        for val, cnt in runs:
            data.append(val & 0x01)
            data.append(cnt & 0xFF)
        return bytes(data)

    def decode(self, data: bytes) -> np.ndarray:
        """Decompress RLE byte stream back to spike array."""
        if len(data) < 4:
            return np.array([], dtype=np.uint8)
        total_len = struct.unpack("<I", data[:4])[0]
        spikes: List[int] = []
        i = 4
        while i + 1 < len(data) and len(spikes) < total_len:
            val = data[i]
            cnt = data[i + 1]
            spikes.extend([val] * cnt)
            i += 2
        return np.array(spikes[:total_len], dtype=np.uint8)

    def compression_ratio(self, original: np.ndarray) -> float:
        """Return compression ratio (original_bytes / compressed_bytes)."""
        compressed = self.encode(original)
        if len(compressed) == 0:
            return 1.0
        return len(original) / len(compressed)


class OnlineLearner:
    """Local STDP-inspired weight update rule (pure Python fallback)."""

    def __init__(
        self,
        num_weights: int,
        lr: float = 0.01,
        decay: float = 0.999,
    ) -> None:
        self.weights = np.ones(num_weights, dtype=np.float32)
        self.lr = lr
        self.decay = decay
        self.updates = 0

    def step(
        self,
        spikes: np.ndarray,
        reward: float,
    ) -> np.ndarray:
        """Apply reward-modulated STDP update.

        Spikes that contributed to a positive reward get potentiated;
        non-spiking channels get depressed toward baseline.
        """
        self.weights *= self.decay

        spike_mask = spikes.astype(bool)
        self.weights[spike_mask] += self.lr * reward
        self.weights[~spike_mask] -= self.lr * reward * 0.1

        self.weights = np.clip(self.weights, 0.01, 10.0)
        self.updates += 1
        return self.weights


class FPGAFeedbackController:
    """Serializes BCI commands for DMA push to FPGA feedback register."""

    COMMAND_NOP = 0
    COMMAND_STIM = 1
    COMMAND_INHIBIT = 2

    def serialize(
        self,
        command: int,
        channel: int = 0,
        amplitude: float = 1.0,
        timestamp_us: float = 0.0,
    ) -> bytes:
        """Pack a feedback command into a 16-byte DMA-aligned struct.

        Layout: [cmd:u8, chan:u16, amp:f32, ts:f64, pad:1]
        """
        return struct.pack("<BHfdx", command, channel, amplitude, timestamp_us)

    def deserialize(self, data: bytes) -> Dict:
        """Unpack a feedback command."""
        cmd, chan, amp, ts = struct.unpack("<BHfdx", data[:16])
        return {"command": cmd, "channel": chan, "amplitude": amp, "timestamp_us": ts}


class LatencyProfiler:
    """Rolling window latency tracker with percentile reporting."""

    def __init__(self, window_size: int = 1000) -> None:
        self.window: deque[float] = deque(maxlen=window_size)

    def record(self, latency_ms: float) -> None:
        self.window.append(latency_ms)

    @property
    def mean(self) -> float:
        return float(np.mean(list(self.window))) if self.window else 0.0

    @property
    def p50(self) -> float:
        return float(np.percentile(list(self.window), 50)) if self.window else 0.0

    @property
    def p95(self) -> float:
        return float(np.percentile(list(self.window), 95)) if self.window else 0.0

    @property
    def p99(self) -> float:
        return float(np.percentile(list(self.window), 99)) if self.window else 0.0

    @property
    def budget_met(self) -> bool:
        """True if p95 latency is under 10 ms BCI hard real-time target."""
        return self.p95 < 10.0


class BCIStudio:
    """End-to-end BCI closed-loop orchestrator."""

    def __init__(
        self,
        channels: int = 1024,
        lr: float = 0.01,
    ) -> None:
        self.channels = channels
        self.codec = SpikeCodec()
        self.learner = OnlineLearner(channels, lr=lr)
        self.feedback = FPGAFeedbackController()
        self.profiler = LatencyProfiler()
        self.metrics = SessionMetrics()
        self._running = False

    def start_session(self) -> None:
        self._running = True
        self.metrics = SessionMetrics()

    def stop_session(self) -> SessionMetrics:
        self._running = False
        return self.metrics

    def process_frame(
        self,
        raw_ephys: np.ndarray,
        reward: float = 0.0,
    ) -> Dict:
        """Process a single BCI frame through the full pipeline."""
        t0 = time.perf_counter()

        # Spike extraction (threshold on diff)
        spikes = (np.abs(np.diff(raw_ephys, prepend=0)) > 0.5).astype(np.uint8)

        # Compression (for telemetry/logging)
        compressed = self.codec.encode(spikes)
        comp_ratio = len(raw_ephys) / max(1, len(compressed))

        # SC decode: weighted vote
        total_voltage = float(np.dot(spikes, self.learner.weights))

        # Online learning
        old_weights = self.learner.weights.copy()
        self.learner.step(spikes, reward)
        weight_delta = float(np.sum(np.abs(self.learner.weights - old_weights)))
        if weight_delta > 0.01 * self.channels:
            self.metrics.adaptation_events += 1

        # Command decision
        command = (
            FPGAFeedbackController.COMMAND_STIM
            if total_voltage > self.channels * 0.1
            else FPGAFeedbackController.COMMAND_NOP
        )

        # Feedback serialization
        feedback_packet = self.feedback.serialize(
            command, channel=0, amplitude=min(total_voltage / self.channels, 1.0)
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.profiler.record(latency_ms)

        # Update session metrics
        n_spikes = int(np.sum(spikes))
        self.metrics.total_frames += 1
        self.metrics.total_spikes += n_spikes
        self.metrics.latency_history.append(latency_ms)

        return {
            "command": command,
            "latency_ms": latency_ms,
            "spikes": n_spikes,
            "compression_ratio": comp_ratio,
            "weight_delta": weight_delta,
            "feedback_bytes": len(feedback_packet),
        }
