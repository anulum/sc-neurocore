# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Codec runtime routes for files

"""Dispatch framed file data through the public Python and Rust codec APIs."""

from __future__ import annotations

import struct
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

from sc_neurocore.spike_codec import (
    CompressionResult,
    LosslessStreamDecoder,
    LosslessStreamEncoder,
    SpikeEventDecoder,
    SpikeEventEncoder,
    WaveformCodec,
    describe_codec,
    get_codec,
    probe_codec,
)

from .file import MAX_FRAME_VALUES, CodecFileHeader
from .samples import OutputField, Samples

_OPTIONS = {
    "isi": {"entropy", "timing_precision"},
    "delta": {"group_size", "timing_precision"},
    "predictive": {"timing_precision"},
    "streaming": {"window_size"},
    "aer": set(),
    "waveform": {"adc_bits"},
    "lossless_stream": {"adc_bits", "groups"},
    "spike_events": {"bin_samples", "power_bins", "groups"},
}
_RASTER_SHAPES = {"aer": struct.Struct("!4sII"), "streaming": struct.Struct("!4sHIH")}


class _RasterCodec(Protocol):
    @property
    def backend(self) -> str: ...
    def compress(self, samples: Samples) -> tuple[bytes, CompressionResult]: ...
    def decompress(self, packet: bytes, rows: int, channels: int) -> Samples: ...


@dataclass(frozen=True, slots=True)
class FileEncoder:
    """Selected encoder and callable for the next source window."""

    backend: str
    native_scope: str
    encode: Callable[[Samples], bytes]


@dataclass(frozen=True, slots=True)
class FileDecoder:
    """Selected decoder and callable for the next packet and its source rows."""

    backend: str
    native_scope: str
    decode: Callable[[bytes, int], dict[str, Samples]]


def check_file_options(header: CodecFileHeader) -> None:
    """Reject unsupported or incompatible settings before constructing any codec."""
    descriptor = describe_codec(header.codec)
    descriptor.variant(header.variant)
    unknown = header.options.keys() - _OPTIONS[header.codec]
    if unknown:
        raise ValueError(f"options unavailable for {header.codec}: {', '.join(sorted(unknown))}")
    for key, value in header.options.items():
        if key == "entropy":
            if value not in ("auto", "varint", "huffman"):
                raise ValueError("entropy must be auto, varint or huffman")
        elif type(value) is not int or not 1 <= value <= (1 << 31) - 1:
            raise ValueError(f"{key} must be a positive integer")
    if header.codec in ("isi", "delta") and header.variant == "lossless":
        if header.options.get("timing_precision", 1) != 1:
            raise ValueError("lossless rasters require timing_precision=1")
    if header.codec == "streaming":
        limit = min(65535, MAX_FRAME_VALUES // header.channels)
        if cast(int, header.options.get("window_size", 20)) > limit:
            raise ValueError("SSCF window exceeds the frame sample limit")
    if header.codec == "waveform" and header.options and not header.variant.startswith("lossless"):
        raise ValueError("waveform adc_bits is available only in lossless modes")


def _scope(header: CodecFileHeader, backend: str) -> str:
    return (
        "none"
        if backend == "python"
        else describe_codec(header.codec).variant(header.variant).native_scope
    )


def _backend(header: CodecFileHeader, backend: str, *, allow_gpu: bool = False) -> None:
    choices = probe_codec(header.codec, header.variant)
    allowed = choices.backend_choices + (choices.encoder_only_backends if allow_gpu else ())
    if backend not in allowed:
        detail = "; ".join(choices.diagnostics)
        raise ValueError(
            f"{header.codec}/{header.variant} cannot use backend {backend!r}. {detail}".rstrip()
        )


def _raster(header: CodecFileHeader, backend: str) -> _RasterCodec:
    descriptor = describe_codec(header.codec)
    kwargs = dict(header.options)
    if descriptor.variant_parameter is not None:
        kwargs[descriptor.variant_parameter] = header.variant
    if header.codec == "predictive" and kwargs.get("timing_precision", 1) != 1:
        kwargs["base_mode"] = "lossy"
    return cast(_RasterCodec, get_codec(header.codec, backend=backend, **kwargs))


def _event_layout(header: CodecFileHeader) -> tuple[int, int, int, int]:
    rate = header.sample_rate_hz
    if rate is None:
        raise ValueError("spike_events requires the input sample rate")
    return (
        rate,
        cast(int, header.options.get("bin_samples", max(round(rate / 1000.0), 1))),
        cast(int, header.options.get("power_bins", 20)),
        cast(int, header.options.get("groups", 1)),
    )


def make_file_encoder(header: CodecFileHeader, backend: str) -> FileEncoder:
    """Create an encoder that retains stream state only for stateful families."""
    check_file_options(header)
    _backend(header, backend, allow_gpu=True)
    if header.codec == "spike_events":
        rate, bins, periods, groups = _event_layout(header)
        events = SpikeEventEncoder(
            header.channels,
            rate,
            bin_samples=bins,
            timing=header.variant,
            power_bins=periods,
            groups=groups,
            backend=backend,
        )
        return FileEncoder(events.backend, _scope(header, events.backend), events.encode)
    if header.codec == "lossless_stream":
        lossless = LosslessStreamEncoder(
            header.channels,
            adc_bits=cast(int, header.options.get("adc_bits", 16)),
            groups=cast(int, header.options.get("groups", 1)),
            backend=backend,
            gpu_max_samples=header.frame_rows if backend == "gpu" else None,
        )
        return FileEncoder(lossless.backend, _scope(header, lossless.backend), lossless.encode)
    if header.codec == "waveform":
        waveform = WaveformCodec(
            mode=header.variant,
            backend=backend,
            adc_bits=cast(int | None, header.options.get("adc_bits")),
        )
        return FileEncoder(
            waveform.effective_backend,
            waveform.native_scope,
            lambda data: waveform.compress(data)[0],
        )
    raster = _raster(header, backend)
    return FileEncoder(
        raster.backend, _scope(header, raster.backend), lambda data: raster.compress(data)[0]
    )


def _raster_shape(packet: bytes, codec: str, shape: tuple[int, int]) -> None:
    header = _RASTER_SHAPES.get(codec)
    if header is None:
        return
    if len(packet) < header.size or header.unpack_from(packet)[-2:] != shape:
        raise ValueError(f"{codec} packet geometry disagrees with its file frame")


def _field(values: Samples | None, rows: int, channels: int) -> Samples:
    if values is None or values.shape != (rows, channels):
        raise ValueError("decoded frame has incompatible dimensions")
    return values


def make_file_decoder(header: CodecFileHeader, backend: str) -> FileDecoder:
    """Create a file decoder and match decoded window dimensions to metadata."""
    check_file_options(header)
    _backend(header, backend)
    channels = header.channels
    if header.codec == "spike_events":
        _, bins, periods, groups = _event_layout(header)
        events = SpikeEventDecoder(
            channels,
            bins,
            timing=header.variant,
            power_bins=periods,
            groups=groups,
            backend=backend,
        )
        consumed = 0

        def decode_events(packet: bytes, rows: int) -> dict[str, Samples]:
            nonlocal consumed
            window = events.decode(packet, rows)
            first_bin, first_period = consumed // bins, consumed // (bins * periods)
            consumed += rows
            count = consumed // bins - first_bin
            fields = {
                "events": _field(window.events, count, channels),
                "power": _field(
                    window.power, consumed // (bins * periods) - first_period, channels
                ),
            }
            if header.variant == "sample":
                fields["offsets"] = _field(window.offsets, count, channels)
            return fields

        return FileDecoder(events.backend, _scope(header, events.backend), decode_events)
    if header.codec == "lossless_stream":
        lossless = LosslessStreamDecoder(
            channels,
            adc_bits=cast(int, header.options.get("adc_bits", 16)),
            groups=cast(int, header.options.get("groups", 1)),
            backend=backend,
        )

        def decode_lossless(packet: bytes, rows: int) -> dict[str, Samples]:
            return {"waveform": _field(lossless.decode(packet, rows), rows, channels)}

        return FileDecoder(lossless.backend, _scope(header, lossless.backend), decode_lossless)
    if header.codec == "waveform":
        waveform = WaveformCodec(mode=header.variant, backend=backend)
        fields = tuple(field.name for field in decoded_fields(header))

        def decode_waveform(packet: bytes, rows: int) -> dict[str, Samples]:
            window = WaveformCodec.decompress(packet, max_samples=rows * channels, backend=backend)
            if window.mode != header.variant:
                raise ValueError("waveform packet mode disagrees with the file metadata")
            values = {"waveform": window.waveform, "spikes": window.spike_raster}
            return {name: _field(values[name], rows, channels) for name in fields}

        return FileDecoder(waveform.effective_backend, waveform.native_scope, decode_waveform)
    raster = _raster(header, backend)

    def decode_raster(packet: bytes, rows: int) -> dict[str, Samples]:
        _raster_shape(packet, header.codec, (rows, channels))
        return {"raster": _field(raster.decompress(packet, rows, channels), rows, channels)}

    return FileDecoder(raster.backend, _scope(header, raster.backend), decode_raster)


def decoded_fields(header: CodecFileHeader) -> tuple[OutputField, ...]:
    """Describe all decoded fields using file geometries and canonical output dtypes."""
    rows, channels = header.rows, header.channels
    if header.codec == "spike_events":
        _, bins, periods, _ = _event_layout(header)
        fields = (
            OutputField("events", rows // bins, channels, "|u1"),
            OutputField("power", rows // (bins * periods), channels, "<i4"),
        )
        if header.variant == "sample":
            return (*fields, OutputField("offsets", rows // bins, channels, "<i2"))
        return fields
    if header.codec == "lossless_stream" or (
        header.codec == "waveform" and header.variant.startswith("lossless")
    ):
        return (OutputField("waveform", rows, channels, "<i2"),)
    if header.codec == "waveform":
        spikes = OutputField("spikes", rows, channels, "|i1")
        if header.variant == "spike":
            return (spikes,)
        return (OutputField("waveform", rows, channels, "<f4"), spikes)
    return (OutputField("raster", rows, channels, "|i1"),)
