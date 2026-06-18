# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — In-Sensor Multi-Modal Event Fusion

"""Multi-modal event fusion primitives for SC-domain sensor processing.

Fuses heterogeneous event streams (DVS, tactile, cochlea, proprioceptive)
using SC-domain cross-modal attention kernels with on-the-fly bitstream
decorrelation.  All operations use stochastic computing arithmetic
(AND = multiply, MUX = scaled add) for sub-milliwatt fusion on FPGA.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np


class SensorModality(Enum):
    """Enumeration of the supported sensor modalities."""

    DVS = "dvs"
    TACTILE = "tactile"
    COCHLEA = "cochlea"
    PROPRIOCEPTIVE = "proprioceptive"
    CUSTOM = "custom"


@dataclass
class EventStream:
    """Timestamped event stream from a single sensor modality."""

    modality: SensorModality
    timestamps: np.ndarray[Any, Any]  # microsecond timestamps
    addresses: np.ndarray[Any, Any]  # spatial/channel addresses
    polarities: np.ndarray[Any, Any]  # +1 / -1 for ON/OFF events
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_events(self) -> int:
        """Return the number of events in the stream."""
        return len(self.timestamps)

    @property
    def duration_us(self) -> float:
        """Return the stream duration in microseconds."""
        if self.num_events < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def event_rate(self) -> float:
        """Return the mean event rate in events per microsecond."""
        dur = self.duration_us
        return self.num_events / (dur * 1e-6) if dur > 0 else 0.0

    def to_bitstream(self, length: int, num_channels: int) -> np.ndarray[Any, Any]:
        """Convert event stream to SC bitstream matrix (channels × length)."""
        bs = np.zeros((num_channels, length), dtype=np.uint8)
        if self.num_events == 0:
            return bs
        dur = max(1.0, self.duration_us)
        t0 = float(self.timestamps[0])
        for i in range(self.num_events):
            ch = int(self.addresses[i]) % num_channels
            pos = int((float(self.timestamps[i]) - t0) / dur * (length - 1))
            pos = max(0, min(length - 1, pos))
            if self.polarities[i] > 0:
                bs[ch, pos] = 1
        return bs


class BitstreamDecorrelator:
    """On-the-fly decorrelation for heterogeneous bitstreams.

    Uses per-stream LFSR-based scrambling to break inter-stream
    correlations introduced by shared clock domains or spatial
    proximity.
    """

    def __init__(self, seed: int = 0xACE1):
        self._base_seed = seed

    def decorrelate(
        self,
        streams: List[np.ndarray[Any, Any]],
        method: str = "lfsr",
    ) -> List[np.ndarray[Any, Any]]:
        """Decorrelate a list of bitstream matrices.

        Each matrix is (channels × length).
        """
        result = []
        for i, stream in enumerate(streams):
            seed = (self._base_seed + i * 7919) & 0xFFFF
            if seed == 0:
                seed = 1
            mask = self._generate_mask(stream.shape, seed, method)
            decorrelated = np.bitwise_xor(stream, mask).astype(np.uint8)
            result.append(decorrelated)
        return result

    def _generate_mask(
        self, shape: Tuple[int, ...], seed: int, method: str
    ) -> np.ndarray[Any, Any]:
        if method == "sobol":
            return self._sobol_mask(shape, seed)
        return self._lfsr_mask(shape, seed)

    def _lfsr_mask(self, shape: Tuple[int, ...], seed: int) -> np.ndarray[Any, Any]:
        rng = np.random.default_rng(seed)
        return rng.integers(0, 2, size=shape, dtype=np.uint8)

    def _sobol_mask(self, shape: Tuple[int, ...], seed: int) -> np.ndarray[Any, Any]:
        total = 1
        for s in shape:
            total *= s
        rng = np.random.default_rng(seed + 1000)
        flat = (rng.random(total) > 0.5).astype(np.uint8)
        return flat.reshape(shape)

    def measure_scc(self, a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> float:
        """Compute stochastic cross-correlation between two bitstreams."""
        a_flat = a.flatten().astype(np.float64)
        b_flat = b.flatten().astype(np.float64)
        pa = float(np.mean(a_flat))
        pb = float(np.mean(b_flat))
        p_and = float(np.mean(a_flat * b_flat))
        num = p_and - (pa * pb)
        if abs(num) < 1e-12:
            return 0.0
        denom = (min(pa, pb) - pa * pb) if num > 0 else (pa * pb - max(0.0, pa + pb - 1.0))
        if abs(denom) < 1e-12:
            return 0.0
        return max(-1.0, min(1.0, num / denom))


class CrossModalAttention:
    """SC-domain cross-modal attention kernel.

    Implements query-key-value attention using stochastic arithmetic:
      - Q·K similarity: SC-AND (bitwise AND = multiplication)
      - Weighted V: SC-MUX (bitwise multiplexer = scaled addition)
    """

    def __init__(self, num_channels: int, seed: int = 42):
        self.num_channels = num_channels
        rng = np.random.default_rng(seed)
        self.W_q = rng.integers(0, 2, (num_channels, num_channels), dtype=np.uint8)
        self.W_k = rng.integers(0, 2, (num_channels, num_channels), dtype=np.uint8)
        self.W_v = rng.integers(0, 2, (num_channels, num_channels), dtype=np.uint8)

    def _sc_and(self, a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        result: np.ndarray[Any, Any] = (a & b).astype(np.uint8)
        return result

    def _sc_mux(
        self, a: np.ndarray[Any, Any], b: np.ndarray[Any, Any], sel: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        result: np.ndarray[Any, Any] = ((a & sel) | (b & ~sel & 1)).astype(np.uint8)
        return result

    def attend(
        self,
        query_stream: np.ndarray[Any, Any],
        key_stream: np.ndarray[Any, Any],
        value_stream: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Compute cross-modal attention in SC domain.

        All inputs are (channels × bitstream_length).
        Returns attended value stream of same shape.
        """
        q = self._project(query_stream, self.W_q)
        k = self._project(key_stream, self.W_k)
        v = self._project(value_stream, self.W_v)

        similarity = self._sc_and(q, k)
        attended = self._sc_mux(v, np.zeros_like(v, dtype=np.uint8), similarity)
        return attended

    def _project(
        self, stream: np.ndarray[Any, Any], weights: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        ch, length = stream.shape
        result = np.zeros_like(stream, dtype=np.uint8)
        for c in range(ch):
            for c2 in range(ch):
                if weights[c, c2]:
                    result[c] |= stream[c2]
        return result


@dataclass
class FusionMetrics:
    """Metrics from a sensor fusion pass."""

    num_streams: int = 0
    total_events: int = 0
    fused_popcount: int = 0
    cross_modal_scc: float = 0.0
    latency_us: float = 0.0


class SensorFusionLayer:
    """Multi-stream sensor fusion with per-modality weighting."""

    def __init__(
        self,
        num_channels: int = 64,
        bitstream_length: int = 256,
        seed: int = 42,
    ):
        self.num_channels = num_channels
        self.bitstream_length = bitstream_length
        self.attention = CrossModalAttention(num_channels, seed)
        self.decorrelator = BitstreamDecorrelator(seed)
        self._modality_weights: Dict[SensorModality, float] = {}

    def set_weight(self, modality: SensorModality, weight: float) -> None:
        """Set the fusion weight for a modality, clipped to ``[0, 1]``."""
        self._modality_weights[modality] = max(0.0, min(1.0, weight))

    def fuse(
        self,
        streams: List[EventStream],
        use_attention: bool = True,
    ) -> Tuple[np.ndarray[Any, Any], FusionMetrics]:
        """Fuse multiple event streams into a single SC bitstream.

        Returns (fused_bitstream, metrics).
        """
        t0 = time.perf_counter()

        bitstreams = []
        for s in streams:
            bs = s.to_bitstream(self.bitstream_length, self.num_channels)
            w = self._modality_weights.get(s.modality, 1.0)
            if w < 1.0:
                mask = (
                    np.random.default_rng(hash(s.modality.value) & 0xFFFF).random(bs.shape) < w
                ).astype(np.uint8)
                bs = bs & mask
            bitstreams.append(bs)

        if not bitstreams:
            empty = np.zeros((self.num_channels, self.bitstream_length), dtype=np.uint8)
            return empty, FusionMetrics()

        decorrelated = self.decorrelator.decorrelate(bitstreams)

        if use_attention and len(decorrelated) >= 2:
            fused = decorrelated[0].copy()
            for i in range(1, len(decorrelated)):
                fused = self.attention.attend(fused, decorrelated[i], decorrelated[i])
        else:
            fused = decorrelated[0].copy()
            for bs in decorrelated[1:]:
                fused = (fused | bs).astype(np.uint8)

        cross_scc = 0.0
        if len(decorrelated) >= 2:
            cross_scc = self.decorrelator.measure_scc(
                decorrelated[0].flatten(), decorrelated[1].flatten()
            )

        elapsed = (time.perf_counter() - t0) * 1e6

        metrics = FusionMetrics(
            num_streams=len(streams),
            total_events=sum(s.num_events for s in streams),
            fused_popcount=int(np.sum(fused)),
            cross_modal_scc=cross_scc,
            latency_us=elapsed,
        )

        return fused, metrics


# ── Hyperdimensional / VSA Cross-Modal Binding ──────────────────────


class HDCBinding:
    """Hyperdimensional computing for cross-modal representation binding.

    Uses binary hypervectors for modality-independent representation.
    Binding: XOR (permutation-invariant association).
    Bundling: majority vote (superposition).
    """

    def __init__(self, dim: int = 1024, seed: int = 42):
        self.dim = dim
        self.rng = np.random.default_rng(seed)
        self._codebooks: Dict[str, np.ndarray[Any, Any]] = {}

    def get_hypervector(self, key: str) -> np.ndarray[Any, Any]:
        """Get or create a random hypervector for a key."""
        if key not in self._codebooks:
            self._codebooks[key] = self.rng.integers(0, 2, self.dim, dtype=np.uint8)
        return self._codebooks[key]

    def bind(self, a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """XOR binding: associate two representations."""
        bound: np.ndarray[Any, Any] = np.bitwise_xor(a, b).astype(np.uint8)
        return bound

    def bundle(self, vectors: List[np.ndarray[Any, Any]]) -> np.ndarray[Any, Any]:
        """Majority-vote bundling: superpose multiple vectors."""
        if not vectors:
            return np.zeros(self.dim, dtype=np.uint8)
        stacked = np.stack(vectors).astype(np.int32)
        majority: np.ndarray[Any, Any] = (np.sum(stacked, axis=0) > len(vectors) / 2).astype(
            np.uint8
        )
        return majority

    def similarity(self, a: np.ndarray[Any, Any], b: np.ndarray[Any, Any]) -> float:
        """Cosine-like similarity via Hamming distance."""
        matches = np.sum(a == b)
        return float(matches / len(a))

    def encode_stream(self, stream: EventStream, num_channels: int = 64) -> np.ndarray[Any, Any]:
        """Encode an event stream as a single hypervector."""
        modality_hv = self.get_hypervector(stream.modality.value)
        bs = stream.to_bitstream(min(self.dim, 256), num_channels)
        stream_hv = np.zeros(self.dim, dtype=np.uint8)
        flat = bs.flatten()
        stream_hv[: len(flat)] = flat[: self.dim]
        return self.bind(modality_hv, stream_hv)


# ── Per-Modality Sensor Adapters ────────────────────────────────────


class DVSAdapter:
    """Adapter for Dynamic Vision Sensor (event camera)."""

    @staticmethod
    def encode_events(
        timestamps: np.ndarray[Any, Any],
        x: np.ndarray[Any, Any],
        y: np.ndarray[Any, Any],
        polarities: np.ndarray[Any, Any],
        resolution: Tuple[int, int] = (128, 128),
    ) -> EventStream:
        """Encode DVS camera events into an EventStream of AER addresses."""
        addresses = (y.astype(np.int64) * resolution[0] + x.astype(np.int64)) % (
            resolution[0] * resolution[1]
        )
        return EventStream(
            modality=SensorModality.DVS,
            timestamps=timestamps,
            addresses=addresses,
            polarities=polarities,
            metadata={"resolution": resolution},
        )


class CochleaAdapter:
    """Adapter for silicon cochlea (frequency-to-channel mapping)."""

    def __init__(
        self, num_channels: int = 64, freq_min_hz: float = 20.0, freq_max_hz: float = 20000.0
    ):
        self.num_channels = num_channels
        self.freq_min = freq_min_hz
        self.freq_max = freq_max_hz

    def freq_to_channel(self, freq_hz: float) -> int:
        """Log-scale frequency to channel mapping (tonotopic)."""
        if freq_hz <= self.freq_min:
            return 0
        if freq_hz >= self.freq_max:
            return self.num_channels - 1
        log_pos = (np.log2(freq_hz) - np.log2(self.freq_min)) / (
            np.log2(self.freq_max) - np.log2(self.freq_min)
        )
        return int(log_pos * (self.num_channels - 1))

    def encode_spikes(
        self, timestamps: np.ndarray[Any, Any], frequencies: np.ndarray[Any, Any]
    ) -> EventStream:
        """Encode cochlear spike timestamps and frequencies into an EventStream."""
        channels = np.array([self.freq_to_channel(f) for f in frequencies])
        return EventStream(
            modality=SensorModality.COCHLEA,
            timestamps=timestamps,
            addresses=channels,
            polarities=np.ones(len(timestamps), dtype=np.int8),
            metadata={"freq_range": (self.freq_min, self.freq_max)},
        )


class TactileAdapter:
    """Adapter for e-skin / tactile sensor arrays."""

    @staticmethod
    def encode_pressure(
        timestamps: np.ndarray[Any, Any],
        taxel_ids: np.ndarray[Any, Any],
        pressures: np.ndarray[Any, Any],
        threshold: float = 0.1,
    ) -> EventStream:
        """Convert pressure readings to ON/OFF events."""
        polarities = np.where(pressures > threshold, 1, -1).astype(np.int8)
        return EventStream(
            modality=SensorModality.TACTILE,
            timestamps=timestamps,
            addresses=taxel_ids,
            polarities=polarities,
            metadata={"threshold": threshold},
        )


class IMUAdapter:
    """Adapter for IMU / proprioceptive streams."""

    @staticmethod
    def encode_angular_rate(
        timestamps: np.ndarray[Any, Any],
        axis_id: np.ndarray[Any, Any],
        rates_dps: np.ndarray[Any, Any],
        deadzone_dps: float = 5.0,
    ) -> EventStream:
        """Convert angular rate to events (above deadzone)."""
        polarities = np.where(rates_dps > 0, 1, -1).astype(np.int8)
        mask = np.abs(rates_dps) > deadzone_dps
        return EventStream(
            modality=SensorModality.PROPRIOCEPTIVE,
            timestamps=timestamps[mask],
            addresses=axis_id[mask],
            polarities=polarities[mask],
            metadata={"deadzone_dps": deadzone_dps},
        )


# ── Temporal Alignment ──────────────────────────────────────────────


class TemporalAligner:
    """Aligns heterogeneous event streams to a common time window."""

    def __init__(self, window_us: float = 1000.0):
        self.window_us = window_us

    def align(self, streams: List[EventStream]) -> List[EventStream]:
        """Slice all streams to their overlapping time window."""
        if not streams:
            return []
        t_min = max(float(s.timestamps[0]) for s in streams if s.num_events > 0)
        t_max = min(float(s.timestamps[-1]) for s in streams if s.num_events > 0)
        if t_min >= t_max:
            return streams

        aligned = []
        for s in streams:
            mask = (s.timestamps >= t_min) & (s.timestamps <= t_max)
            aligned.append(
                EventStream(
                    modality=s.modality,
                    timestamps=s.timestamps[mask],
                    addresses=s.addresses[mask],
                    polarities=s.polarities[mask],
                    metadata=s.metadata,
                )
            )
        return aligned

    def slice_windows(self, stream: EventStream) -> List[EventStream]:
        """Slice a stream into fixed-width time windows."""
        if stream.num_events < 2:
            return [stream]
        t0 = float(stream.timestamps[0])
        t_end = float(stream.timestamps[-1])
        windows = []
        while t0 < t_end:
            t1 = t0 + self.window_us
            mask = (stream.timestamps >= t0) & (stream.timestamps < t1)
            if np.any(mask):
                windows.append(
                    EventStream(
                        modality=stream.modality,
                        timestamps=stream.timestamps[mask],
                        addresses=stream.addresses[mask],
                        polarities=stream.polarities[mask],
                        metadata=stream.metadata,
                    )
                )
            t0 = t1
        return windows if windows else [stream]


# ── Fusion Verilog Emitter ──────────────────────────────────────────


class FusionVerilogEmitter:
    """Generates SystemVerilog for configurable multi-modal fusion."""

    @staticmethod
    def emit(
        module_name: str = "sc_multimodal_fusion",
        num_streams: int = 4,
        bitstream_width: int = 16,
        use_attention: bool = True,
    ) -> str:
        """Emit configurable multi-modal fusion SystemVerilog as a string."""
        lines = [
            "// SC-NeuroCore — Auto-Generated Multi-Modal Fusion",
            f"// Streams: {num_streams}, Bitstream: {bitstream_width}b",
            "",
            f"module {module_name} #(",
            f"    parameter STREAMS      = {num_streams},",
            f"    parameter BITSTREAM_W  = {bitstream_width}",
            ")(",
            "    input  logic clk,",
            "    input  logic rst_n,",
            "    input  logic [STREAMS-1:0]     aer_valid_in,",
            "    input  logic [BITSTREAM_W-1:0] aer_data_in [0:STREAMS-1],",
            "    output logic                   aer_valid_out,",
            "    output logic [BITSTREAM_W-1:0] fused_data_out",
            ");",
            "",
            "    // Per-stream LFSR decorrelation",
            "    logic [15:0] lfsr [0:STREAMS-1];",
            "    logic [BITSTREAM_W-1:0] decorr [0:STREAMS-1];",
            "",
            "    integer i;",
            "    always_ff @(posedge clk or negedge rst_n) begin",
            "        if (!rst_n) begin",
            "            for (i = 0; i < STREAMS; i++) lfsr[i] <= 16'hACE1 + i[15:0];",
            "            aer_valid_out <= 1'b0;",
            "            fused_data_out <= '0;",
            "        end else begin",
            "            // LFSR update",
            "            for (i = 0; i < STREAMS; i++)",
            "                lfsr[i] <= {lfsr[i][14:0], lfsr[i][15] ^ lfsr[i][13] ^ lfsr[i][12] ^ lfsr[i][10]};",
            "",
            "            // Decorrelate",
            "            for (i = 0; i < STREAMS; i++)",
            "                decorr[i] <= aer_data_in[i] ^ lfsr[i][BITSTREAM_W-1:0];",
            "",
        ]

        if use_attention:
            lines.extend(
                [
                    "            // Cross-modal attention (SC-AND coincidence)",
                    "            if (&aer_valid_in) begin",
                    "                aer_valid_out <= 1'b1;",
                    "                fused_data_out <= decorr[0];",
                    "                for (i = 1; i < STREAMS; i++)",
                    "                    fused_data_out <= fused_data_out & decorr[i];",
                    "            end else begin",
                    "                aer_valid_out <= 1'b0;",
                    "            end",
                ]
            )
        else:
            lines.extend(
                [
                    "            // Simple OR fusion",
                    "            aer_valid_out <= |aer_valid_in;",
                    "            fused_data_out <= decorr[0];",
                    "            for (i = 1; i < STREAMS; i++)",
                    "                fused_data_out <= fused_data_out | decorr[i];",
                ]
            )

        lines.extend(
            [
                "        end",
                "    end",
                "",
                "endmodule",
            ]
        )
        return "\n".join(lines)


# ── Energy Estimator ────────────────────────────────────────────────


@dataclass
class FusionEnergyEstimate:
    """Sub-mW energy estimate for fusion pipeline."""

    decorrelation_uw: float = 0.0
    attention_uw: float = 0.0
    routing_uw: float = 0.0
    total_uw: float = 0.0

    @property
    def total_mw(self) -> float:
        """Return the total estimated power in milliwatts."""
        return self.total_uw / 1000.0


class FusionEnergyEstimator:
    """Estimates per-inference energy for SC fusion on FPGA."""

    def __init__(self, tech_node_nm: int = 28, vdd_v: float = 0.9):
        self.tech_node_nm = tech_node_nm
        self.vdd_v = vdd_v
        # Energy per LUT switch (fJ) — scales with tech node
        self._efj_per_lut = 0.5 * (tech_node_nm / 7.0)

    def estimate(
        self,
        num_streams: int,
        num_channels: int,
        bitstream_length: int,
        use_attention: bool = True,
        clock_mhz: float = 100.0,
    ) -> FusionEnergyEstimate:
        """Estimate fusion energy from stream, channel, and timing parameters."""
        # LFSR: 16-bit per stream, 1 toggle/cycle over bitstream_length cycles
        lfsr_toggles = num_streams * 16 * bitstream_length
        decorr_fj = lfsr_toggles * self._efj_per_lut

        # Attention: AND per channel pair per bit
        if use_attention:
            attn_ops = num_channels * num_streams * bitstream_length
            attn_fj = attn_ops * self._efj_per_lut * 2
        else:
            attn_fj = 0.0

        # AER routing: 1 mux per stream per channel
        routing_fj = num_streams * num_channels * self._efj_per_lut

        total_fj = decorr_fj + attn_fj + routing_fj

        # Inference time = bitstream_length cycles at clock_mhz
        inference_time_us = bitstream_length / clock_mhz

        # Average power during inference: E / t
        decorr_uw = (
            (decorr_fj * 1e-15) / (inference_time_us * 1e-6) * 1e6 if inference_time_us > 0 else 0.0
        )
        attn_uw = (
            (attn_fj * 1e-15) / (inference_time_us * 1e-6) * 1e6 if inference_time_us > 0 else 0.0
        )
        routing_uw = (
            (routing_fj * 1e-15) / (inference_time_us * 1e-6) * 1e6
            if inference_time_us > 0
            else 0.0
        )
        total_uw = decorr_uw + attn_uw + routing_uw

        return FusionEnergyEstimate(
            decorrelation_uw=decorr_uw,
            attention_uw=attn_uw,
            routing_uw=routing_uw,
            total_uw=total_uw,
        )
