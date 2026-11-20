# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Recording codec operations

"""Encode, verify and export recordings through the shared file adapters."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from sc_neurocore.spike_codec import describe_codec

from .file import (
    MAX_FRAME_VALUES,
    CodecFileHeader,
    atomic_codec_output,
    iter_codec_packets,
    read_codec_header,
    write_codec_file,
)

if TYPE_CHECKING:
    from .samples import OutputField, SampleInput, Samples

Progress = Callable[[int, int], None]
Preview = Callable[[int, Mapping[str, "Samples"]], None]


@dataclass(frozen=True, slots=True)
class RecordingOptions:
    """Select a codec and describe acquisition geometry.

    Attributes
    ----------
    codec, variant
        Catalogue name and optional named variant.
    frame_rows
        Maximum number of source rows per packet. The default is bounded by
        the source channel count.
    sample_rate_hz
        Acquisition rate in Hz. A WAV header supplies its own rate.
    rows, channels
        Total declared geometry for sorted event CSV or legacy BCI RLE.
    codec_options
        Settings to store in the packet file and validate against the codec.
    legacy_rle
        Interpret the source as explicitly selected legacy BCI raster data.
    """

    codec: str
    variant: str | None = None
    frame_rows: int | None = None
    sample_rate_hz: int | None = None
    rows: int | None = None
    channels: int | None = None
    codec_options: Mapping[str, str | int] = field(default_factory=dict)
    legacy_rle: bool = False


@dataclass(frozen=True, slots=True)
class EncodedRecording:
    """Describe a published recording, including its framed byte count and digest."""

    header: CodecFileHeader
    backend: str
    native_scope: str
    size_bytes: int
    content_sha256: str


@dataclass(frozen=True, slots=True)
class DecodedRecording:
    """Describe verified fields returned by the decoder or written to an export."""

    header: CodecFileHeader
    backend: str
    native_scope: str
    fields: tuple[OutputField, ...]


def recording_header(
    path: Path, samples: SampleInput, options: RecordingOptions
) -> CodecFileHeader:
    """Combine the source shape and codec options into bounded recording metadata."""
    descriptor = describe_codec(options.codec)
    variant = descriptor.variant(options.variant).name
    if options.legacy_rle and descriptor.input_kind != "binary_raster":
        raise ValueError("legacy RLE can only be encoded by a raster codec")
    if path.suffix.lower() == ".csv" and descriptor.input_kind != "binary_raster":
        raise ValueError("event CSV can only be encoded by a raster codec")
    frame_rows = options.frame_rows
    if frame_rows is None:
        frame_rows = min(4096, MAX_FRAME_VALUES // max(samples.channels, 1))
    return CodecFileHeader(
        descriptor.name,
        variant,
        samples.rows,
        samples.channels,
        frame_rows,
        samples.sample_rate_hz,
        dict(options.codec_options),
    )


def open_recording_source(
    path: Path, options: RecordingOptions
) -> AbstractContextManager[SampleInput]:
    """Open the declared recording format with its own geometry rules."""
    if options.legacy_rle:
        from .legacy_rle import open_legacy_raster

        return open_legacy_raster(path, options.rows, options.channels, options.sample_rate_hz)
    from .samples import open_sample_input

    return open_sample_input(
        path,
        options.sample_rate_hz,
        event_rows=options.rows,
        event_channels=options.channels,
    )


def encode_recording(
    source: Path,
    destination: Path,
    options: RecordingOptions,
    *,
    backend: str = "auto",
    overwrite: bool = False,
    progress: Progress | None = None,
) -> EncodedRecording:
    """Encode bounded source windows and publish the complete packet file.

    Parameters
    ----------
    source, destination
        Input acquisition and distinct output file on the local filesystem.
    options
        Codec, variant and source geometry.
    backend
        Runtime backend selector supported by the selected codec and direction.
    overwrite
        Permit atomic replacement of an existing destination after success.
    progress
        Optional callback with processed and total input rows, called after
        each window is encoded. Raising abandons the pending destination.

    Returns
    -------
    EncodedRecording
        Metadata and content digest of the published file.
    """
    from .routes import make_file_encoder

    with open_recording_source(source, options) as samples:
        header = recording_header(source, samples, options)
        encoder = make_file_encoder(header, backend)

        def packets() -> Iterator[bytes]:
            processed = 0
            for window in samples.windows(header.frame_sizes()):
                packet = encoder.encode(window)
                processed += window.shape[0]
                if progress is not None:
                    progress(processed, header.rows)
                yield packet

        with (
            atomic_codec_output(destination, source, overwrite) as pending,
            pending.open("xb") as stream,
        ):
            digest = write_codec_file(stream, header, packets())
            size = stream.tell()
    return EncodedRecording(header, encoder.backend, encoder.native_scope, size, digest)


def decode_recording(
    source: Path,
    destination: Path | None = None,
    *,
    backend: str = "auto",
    field_name: str = "all",
    overwrite: bool = False,
    progress: Progress | None = None,
    preview: Preview | None = None,
) -> DecodedRecording:
    """Verify every packet and optionally export decoded fields atomically.

    Parameters
    ----------
    source, destination
        Framed packet file and optional NPY or NPZ output. Omit the output
        to verify the file without writing an export.
    backend
        Runtime backend selector for decoding.
    field_name
        Exported field or "all" for every available field.
    overwrite
        Permit atomic replacement of an existing destination after success.
    progress
        Optional callback with processed and total input rows.
    preview
        Optional callback with the input row offset and read-only views of
        the decoded fields. Collect only the preview needed. The file is
        unverified until this function returns. Raising in either callback
        abandons the export.

    Returns
    -------
    DecodedRecording
        Verified recording metadata and the selected fields.
    """
    from .routes import decoded_fields, make_file_decoder
    from .samples import write_decoded_samples

    with source.open("rb") as stream:
        header, prefix = read_codec_header(stream)
        decoder = make_file_decoder(header, backend)
        fields = decoded_fields(header)

        def windows() -> Iterator[dict[str, Samples]]:
            processed = 0
            for rows, packet in iter_codec_packets(stream, header, prefix):
                values = decoder.decode(packet, rows)
                if preview is not None:
                    views = {name: data.view() for name, data in values.items()}
                    for view in views.values():
                        view.setflags(write=False)
                    preview(processed, views)
                processed += rows
                if progress is not None:
                    progress(processed, header.rows)
                yield values

        if destination is None:
            for _ in windows():
                pass
        else:
            if field_name != "all":
                fields = tuple(field for field in fields if field.name == field_name)
                if not fields:
                    raise ValueError(f"decoded file has no {field_name!r} field")
            suffix = destination.suffix.lower()
            if suffix not in (".npy", ".npz") or (suffix == ".npy" and len(fields) != 1):
                raise ValueError("use .npz for multiple fields or select one field for .npy output")
            with atomic_codec_output(destination, source, overwrite) as pending:
                write_decoded_samples(pending, fields, windows(), archive=suffix == ".npz")
    return DecodedRecording(header, decoder.backend, decoder.native_scope, fields)
