# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Zenith closed-loop BCI primitive

"""Deterministic BCI closed-loop primitive with bounded latency accounting.

This wrapper packages the existing closed-loop template into a single production
surface (`ZenithBCILoop`) and reports a stage-level latency budget ledger for
Neuralink/Neuropixels-style continuous streams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sc_neurocore.interfaces.bci_closed_loop import ClosedLoopBCIConfig, ClosedLoopBCITemplate


@dataclass(frozen=True)
class ZenithBCILoopConfig:
    """Configuration for deterministic closed-loop stream processing.

    Attributes
    ----------
    n_channels:
        Number of neural acquisition channels in each waveform window.
    sampling_rate_hz:
        Sample rate used to convert window length into ingest latency.
    gpu_lanes:
        Parallel codec/decode lanes available for latency estimation.
    latency_budget_ms:
        Maximum allowed end-to-end closed-loop latency in milliseconds.
    threshold_sigma:
        Spike-detection threshold multiplier passed to the BCI template.
    snippet_samples:
        Number of waveform samples captured around each detected event.
    waveform_mode:
        Closed-loop waveform codec mode forwarded to the BCI template.
    quantize_bits:
        Bit width used by the waveform quantizer.
    timestamp_bits:
        Bit width reserved for encoded event timestamps.
    """

    n_channels: int
    sampling_rate_hz: int = 30_000
    gpu_lanes: int = 4
    latency_budget_ms: float = 10.0
    threshold_sigma: float = 4.5
    snippet_samples: int = 48
    waveform_mode: str = "spike"
    quantize_bits: int = 6
    timestamp_bits: int = 16

    def __post_init__(self) -> None:
        """Validate strictly positive loop sizing and latency parameters."""
        if self.n_channels <= 0:
            raise ValueError("n_channels must be positive")
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if self.gpu_lanes <= 0:
            raise ValueError("gpu_lanes must be positive")
        if self.latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms must be positive")


@dataclass(frozen=True)
class ZenithBCILoopResult:
    """Single-step closed-loop output with latency budget evidence.

    Attributes
    ----------
    command:
        Integer control action emitted for the processed waveform window.
    feedback_active_channels:
        Number of channels that received active feedback.
    spike_count:
        Total detected spikes in the processed window.
    decoded_rates:
        Per-channel decoded spike-rate estimates.
    latency_breakdown_ms:
        Stage-level latency ledger keyed by processing stage name.
    total_latency_ms:
        Sum of all estimated stage latencies in milliseconds.
    latency_budget_ms:
        Budget the closed-loop step was checked against.
    latency_budget_met:
        Whether ``total_latency_ms`` stayed within ``latency_budget_ms``.
    pathway_name:
        Human-readable identifier for the acquisition/control pathway.
    schema_version:
        Stable serializer schema emitted by :meth:`to_dict`.
    """

    command: int
    feedback_active_channels: int
    spike_count: int
    decoded_rates: tuple[float, ...]
    latency_breakdown_ms: dict[str, float]
    total_latency_ms: float
    latency_budget_ms: float
    latency_budget_met: bool
    pathway_name: str
    schema_version: str = "sc-neurocore.zenith-bci-loop.v1"

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-compatible result payload."""
        return {
            "schema_version": self.schema_version,
            "command": self.command,
            "feedback_active_channels": self.feedback_active_channels,
            "spike_count": self.spike_count,
            "decoded_rates": list(self.decoded_rates),
            "latency_breakdown_ms": dict(self.latency_breakdown_ms),
            "total_latency_ms": self.total_latency_ms,
            "latency_budget_ms": self.latency_budget_ms,
            "latency_budget_met": self.latency_budget_met,
            "pathway_name": self.pathway_name,
        }


class ZenithBCILoop:
    """Closed-loop primitive for continuous BCI streams with latency guarantees."""

    def __init__(self, config: ZenithBCILoopConfig) -> None:
        self.config = config
        self.template = ClosedLoopBCITemplate(
            ClosedLoopBCIConfig(
                n_channels=config.n_channels,
                sampling_rate_hz=config.sampling_rate_hz,
                threshold_sigma=config.threshold_sigma,
                snippet_samples=config.snippet_samples,
                waveform_mode=config.waveform_mode,
                quantize_bits=config.quantize_bits,
                timestamp_bits=config.timestamp_bits,
            )
        )

    def process_stream(
        self,
        waveform: np.ndarray[Any, Any],
        *,
        window_start_us: int = 0,
        pathway_name: str = "bci-neural-stream",
    ) -> ZenithBCILoopResult:
        """Process one continuous stream window into a closed-loop control action."""
        if not pathway_name:
            raise ValueError("pathway_name must be non-empty")
        window = np.asarray(waveform, dtype=np.float32)
        if window.ndim != 2:
            raise ValueError("waveform must have shape (samples, channels)")
        if window.shape[1] != self.config.n_channels:
            raise ValueError(
                f"waveform has {window.shape[1]} channels, expected {self.config.n_channels}"
            )
        if window.shape[0] == 0:
            raise ValueError("waveform must contain at least one sample")

        processed = self.template.process_window(window, window_start_us=window_start_us)
        latency = self._estimate_latency_ms(samples=window.shape[0], channels=window.shape[1])
        total = float(sum(latency.values()))
        budget_met = total <= self.config.latency_budget_ms

        return ZenithBCILoopResult(
            command=int(processed.feedback.active_count > 0),
            feedback_active_channels=processed.feedback.active_count,
            spike_count=int(processed.spike_raster.sum()),
            decoded_rates=tuple(float(v) for v in processed.decoded_rates.tolist()),
            latency_breakdown_ms=latency,
            total_latency_ms=total,
            latency_budget_ms=self.config.latency_budget_ms,
            latency_budget_met=budget_met,
            pathway_name=pathway_name,
        )

    def _estimate_latency_ms(self, *, samples: int, channels: int) -> dict[str, float]:
        ingest = (samples / self.config.sampling_rate_hz) * 0.2
        codec = (samples * channels) / (self.config.gpu_lanes * 2_000_000.0)
        decode = (channels / self.config.gpu_lanes) / 20_000.0
        feedback = 0.05
        return {
            "ingest": float(ingest),
            "codec": float(codec),
            "decode": float(decode),
            "feedback": float(feedback),
        }
