# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed codec acquisitions and round trips

"""Exercise the real public packet producers and consumers across the catalogue."""

from __future__ import annotations

import wave
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from sc_neurocore.neurons import StochasticLIFNeuron
from sc_neurocore.spike_codec import (
    AERSpikeCodec,
    CompressionResult,
    DeltaSpikeCodec,
    LosslessStreamDecoder,
    LosslessStreamEncoder,
    PredictiveSpikeCodec,
    SpikeCodec,
    SpikeEventDecoder,
    SpikeEventEncoder,
    StreamingSpikeCodec,
    WaveformCodec,
)

Raster = NDArray[np.int8]
Samples = NDArray[np.int16]
Packets = tuple[bytes, ...]
Output = tuple[NDArray[np.generic] | None, ...]


@dataclass(frozen=True)
class Outcome:
    """Transmitted bytes, consumer output and actual endpoint selection."""

    packets: Packets
    output: Output
    backend: str
    decoder_backend: str
    native_scope: str | None = None


@dataclass(frozen=True)
class Case:
    """A named workload exercising one catalogue variant."""

    family: str
    variant: str
    workload: str
    run: Callable[[str, Packets | None], Outcome]


class RasterCodec(Protocol):
    """Common whole-raster consumer contract of the public codec classes."""

    @property
    def backend(self) -> str:
        """Return the selected implementation."""
        ...

    def compress(self, spikes: Raster) -> tuple[bytes, CompressionResult]:
        """Encode a time-major binary raster and report its compression statistics."""
        ...

    def decompress(self, data: bytes, T: int, N: int) -> Raster:
        """Decode one raster with its dimensions."""
        ...


def acquisition(recording: Path | None) -> tuple[Raster, Samples, int]:
    """Capture a public LIF simulation, with optional waveform input from PCM16."""
    neurons = [
        StochasticLIFNeuron(v_threshold=1.0, dt=0.05, noise_std=0.0, seed=channel)
        for channel in range(3)
    ]
    raster = np.empty((1027, len(neurons)), dtype=np.int8)
    voltage = np.empty_like(raster, dtype=np.float64)
    for row in range(len(raster)):
        for channel, neuron in enumerate(neurons):
            current = 0.25 if (row + 31 * channel) % (173 + 23 * channel) < 40 else 1.5
            raster[row, channel] = neuron.step(current)
            voltage[row, channel] = neuron.v
    if not np.any(raster) or np.all(raster):
        raise ValueError("simulated acquisition lacks both active and inactive samples")
    if recording is None:
        return raster, np.rint(voltage * 16000).astype(np.int16), 20000
    with wave.open(str(recording), "rb") as reader:
        channels = reader.getnchannels()
        rate = reader.getframerate()
        samples = reader.readframes(reader.getnframes())
        if reader.getsampwidth() != 2 or not samples or rate < 3000:
            raise ValueError("expected a nonempty high-rate PCM16 recording")
    return raster, np.frombuffer(samples, dtype="<i2").reshape(-1, channels), rate


def _quantised(raster: Raster, precision: int) -> Raster:
    """Collapse complete bins and retain the original output dimensions."""
    if precision == 1:
        return raster
    rows, channels = raster.shape
    bins = rows // precision
    output = np.zeros_like(raster)
    output[:bins] = raster[: bins * precision].reshape(bins, precision, channels).max(axis=1)
    return output


def _delta_quantised(raster: Raster, precision: int, group_size: int) -> Raster:
    """Compute the documented DSCX result with timing quantisation after residual coding."""
    expected = np.empty_like(raster)
    for first in range(0, raster.shape[1], group_size):
        group = raster[:, first : first + group_size]
        reference = int(group.sum(axis=0).argmax())
        residual = group ^ group[:, reference : reference + 1]
        observed = _quantised(residual, precision) ^ _quantised(
            group[:, reference : reference + 1], precision
        )
        expected[:, first : first + group.shape[1]] = observed
    return expected


def _raster(
    make: Callable[..., RasterCodec],
    raster: Raster,
    expected: Raster,
    backend: str,
    packets: Packets | None,
) -> Outcome:
    """Cross-read raster packets and check exact original or quantised samples."""
    codec = make(backend=backend)
    rows, channels = raster.shape
    encoded = (codec.compress(raster)[0],) if packets is None else packets
    output = codec.decompress(encoded[0], rows, channels)
    np.testing.assert_array_equal(output, expected)
    return Outcome(encoded, (output,), codec.backend, codec.backend)


def _frames(raster: Raster, backend: str, packets: Packets | None) -> Outcome:
    """Stream independent frames and retain an unpadded final raster window."""
    codec = StreamingSpikeCodec(37, backend=backend)
    windows = tuple(raster[pos : pos + 37] for pos in range(0, len(raster), 37))
    encoded = (
        tuple(codec.compress_frame(window) for window in windows) if packets is None else packets
    )
    output = np.concatenate([codec.decompress_frame(packet) for packet in encoded])
    np.testing.assert_array_equal(output, raster)
    return Outcome(encoded, (output,), codec.backend, codec.backend)


def _waveform(samples: Samples, mode: str, backend: str, packets: Packets | None) -> Outcome:
    """Exercise every mode with its own WFC2 reconstruction contract."""
    codec = WaveformCodec(mode=mode, backend=backend)
    encoded = (codec.compress(samples)[0],) if packets is None else packets
    result = codec.decompress(encoded[0], backend=backend)
    if result.mode != mode:
        raise ValueError(f"expected waveform mode {mode}, received {result.mode}")
    if mode.startswith("lossless"):
        np.testing.assert_array_equal(result.waveform, samples)
    else:
        if result.spike_raster is None or result.spike_raster.shape != samples.shape:
            raise ValueError(f"{mode} did not return a full raster")
        if mode != "spike" and (result.waveform is None or result.waveform.shape != samples.shape):
            raise ValueError(f"{mode} did not return a full waveform")
    return Outcome(
        encoded,
        (result.spike_raster, result.waveform),
        codec.effective_backend,
        codec.effective_backend,
        codec.native_scope,
    )


def _lossless_stream(samples: Samples, backend: str, packets: Packets | None) -> Outcome:
    """Read a stateful stream with unequal groups and an incomplete final window."""
    channels = samples.shape[1]
    encoder = LosslessStreamEncoder(channels, groups=min(2, channels), backend=backend)
    decoder = LosslessStreamDecoder(channels, groups=min(2, channels), backend=backend)
    windows = tuple(samples[pos : pos + 37] for pos in range(0, len(samples), 37))
    encoded = tuple(encoder.encode(window) for window in windows) if packets is None else packets
    output = np.concatenate(
        [
            decoder.decode(packet, len(window))
            for packet, window in zip(encoded, windows, strict=True)
        ]
    )
    np.testing.assert_array_equal(output, samples)
    return Outcome(encoded, (output,), encoder.backend, decoder.backend)


def _events(
    samples: Samples, rate: int, timing: str, backend: str, packets: Packets | None
) -> Outcome:
    """Exercise finished bins, offsets, power periods and carried partial intervals."""
    channels = samples.shape[1]
    groups = min(2, channels)
    bin_samples = max(round(rate / 1000), 1)
    encoder = SpikeEventEncoder(
        channels, rate, timing=timing, power_bins=3, groups=groups, backend=backend
    )
    decoder = SpikeEventDecoder(
        channels, bin_samples, timing=timing, power_bins=3, groups=groups, backend=backend
    )
    windows = tuple(samples[pos : pos + 997] for pos in range(0, len(samples), 997))
    encoded = tuple(encoder.encode(window) for window in windows) if packets is None else packets
    output: list[NDArray[np.generic] | None] = []
    first_bin = first_period = 0
    for packet, window in zip(encoded, windows, strict=True):
        result = decoder.decode(packet, len(window))
        if (result.first_bin, result.first_period) != (first_bin, first_period):
            raise ValueError("event and bandpower indices lost stream state")
        first_bin += len(result.events)
        first_period += len(result.power)
        output.extend((result.events, result.offsets, result.power))
    if (first_bin, first_period) != (
        len(samples) // bin_samples,
        len(samples) // (3 * bin_samples),
    ):
        raise ValueError("event stream lost a completed bin or power period")
    return Outcome(encoded, tuple(output), encoder.backend, decoder.backend)


def cases(recording: Path | None) -> list[Case]:
    """Build inputs and explicit selections for every public catalogue variant."""
    raster, samples, rate = acquisition(recording)
    made = []
    for mode, precision in (("lossless", 1), ("lossy", 4)):
        for entropy in ("auto", "varint", "huffman"):
            codec = partial(SpikeCodec, mode=mode, timing_precision=precision, entropy=entropy)
            made.append(
                Case(
                    "isi",
                    mode,
                    entropy,
                    partial(_raster, codec, raster, _quantised(raster, precision)),
                )
            )
        delta = partial(DeltaSpikeCodec, group_size=2, mode=mode, timing_precision=precision)
        made.append(
            Case(
                "delta",
                mode,
                mode,
                partial(_raster, delta, raster, _delta_quantised(raster, precision, 2)),
            )
        )
    for predictor in ("ema", "lfsr", "context", "world_model"):
        for mode, precision in (("lossless", 1), ("lossy", 4)):
            predictive = partial(
                PredictiveSpikeCodec,
                predictor=predictor,
                base_mode=mode,
                timing_precision=precision,
            )
            made.append(
                Case(
                    "predictive",
                    predictor,
                    mode,
                    partial(_raster, predictive, raster, _quantised(raster, precision)),
                )
            )
    streaming = partial(StreamingSpikeCodec, window_size=37)
    made.append(
        Case("streaming", "lossless", "container", partial(_raster, streaming, raster, raster))
    )
    made.append(Case("streaming", "lossless", "frames", partial(_frames, raster)))
    for polarity, data in (("events", raster), ("silences", 1 - raster)):
        made.append(Case("aer", "lossless", polarity, partial(_raster, AERSpikeCodec, data, data)))
    for mode in ("spike", "waveform", "full", "lossless", "lossless_rice"):
        made.append(Case("waveform", mode, mode, partial(_waveform, samples, mode)))
    made.append(
        Case("lossless_stream", "lossless", "partial tail", partial(_lossless_stream, samples))
    )
    for timing in ("bin", "sample"):
        made.append(Case("spike_events", timing, timing, partial(_events, samples, rate, timing)))
    return made
