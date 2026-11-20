# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed codec commands

"""Discover, encode, decode, verify and time codecs on real recording files."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
import sys
import wave
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from time import perf_counter_ns

from sc_neurocore.spike_codec import describe_codec, list_codec_descriptors, probe_codec

from ..codec.file import read_codec_header
from ..codec.recording import (
    RecordingOptions,
    decode_recording,
    encode_recording,
    open_recording_source,
    recording_header,
)

_BACKENDS = ("auto", "python", "rust")


def _source_options(args: argparse.Namespace) -> RecordingOptions:
    options: dict[str, str | int] = {}
    for item in args.option:
        name, equals, value = item.partition("=")
        if not equals or not name or not value or name in options:
            raise ValueError("options must be unique NAME=VALUE pairs")
        options[name] = value if name == "entropy" else int(value)
    return RecordingOptions(
        codec=args.codec,
        variant=args.variant,
        frame_rows=args.frame_rows,
        sample_rate_hz=args.sample_rate,
        rows=args.rows,
        channels=args.channels,
        codec_options=options,
        legacy_rle=args.legacy_rle,
    )


def _discover(args: argparse.Namespace) -> dict[str, object]:
    return {"codecs": [asdict(item) for item in list_codec_descriptors()]}


def _inspect(args: argparse.Namespace) -> dict[str, object]:
    if args.file:
        if args.variant is not None:
            raise ValueError("--variant is only valid when inspecting a catalogue codec")
        with args.file.open("rb") as stream:
            header, _ = read_codec_header(stream)
        return {"file": asdict(header), "verified": False}
    return {
        "codec": asdict(describe_codec(args.codec)),
        "availability": asdict(probe_codec(args.codec, args.variant)),
    }


def _encode(args: argparse.Namespace) -> dict[str, object]:
    result = encode_recording(
        args.input, args.output, _source_options(args), backend=args.backend, overwrite=args.force
    )
    return {
        "file": asdict(result.header),
        "backend": result.backend,
        "native_scope": result.native_scope,
        "bytes": result.size_bytes,
        "content_sha256": result.content_sha256,
    }


def _read(args: argparse.Namespace) -> dict[str, object]:
    output: Path | None = args.output
    if output is None:
        result = decode_recording(args.input, backend=args.backend)
    else:
        result = decode_recording(
            args.input, output, backend=args.backend, field_name=args.field, overwrite=args.force
        )
    return {
        "file": asdict(result.header),
        "backend": result.backend,
        "native_scope": result.native_scope,
        "fields": [asdict(field) for field in result.fields],
        "verified": True,
    }


def _measure(args: argparse.Namespace, backend: str) -> dict[str, object]:
    from ..codec.routes import make_file_decoder, make_file_encoder

    options = _source_options(args)
    with open_recording_source(args.input, options) as samples:
        header = recording_header(args.input, samples, options)
        encoder = make_file_encoder(header, backend)
        decoder = make_file_decoder(header, "auto" if backend == "gpu" else backend)
        payload_hash, output_hash = hashlib.sha256(), hashlib.sha256()
        encode_ns = decode_ns = payload_bytes = 0
        for window in samples.windows(header.frame_sizes()):
            before = perf_counter_ns()
            packet = encoder.encode(window)
            encoded = perf_counter_ns()
            decoded = decoder.decode(packet, window.shape[0])
            finished = perf_counter_ns()
            encode_ns += encoded - before
            decode_ns += finished - encoded
            payload_bytes += len(packet)
            payload_hash.update(struct.pack("!I", len(packet)))
            payload_hash.update(packet)
            for name, values in sorted(decoded.items()):
                output_hash.update(name.encode("ascii"))
                output_hash.update(values.tobytes())
    return {
        "file": asdict(header),
        "backend": encoder.backend,
        "decoder_backend": decoder.backend,
        "native_scope": encoder.native_scope,
        "encode_ns": encode_ns,
        "decode_ns": decode_ns,
        "payload_bytes": payload_bytes,
        "packet_sha256": payload_hash.hexdigest(),
        "decoded_sha256": output_hash.hexdigest(),
    }


def _benchmark(args: argparse.Namespace) -> dict[str, object]:
    if not 1 <= args.repetitions <= 100:
        raise ValueError("benchmark repetitions must be between 1 and 100")
    backends = ("python", "rust") if args.backend == "compare" else (args.backend,)
    with args.input.open("rb") as source:
        input_hash = hashlib.file_digest(source, "sha256").hexdigest()
    runs = []
    expected = None
    for index in range(args.repetitions):
        for backend in backends if index % 2 == 0 else reversed(backends):
            result = _measure(args, backend)
            hashes = (result["packet_sha256"], result["decoded_sha256"])
            if expected is not None and hashes != expected:
                raise ValueError("benchmark backends or repetitions disagree on packets or output")
            expected = hashes
            runs.append(result)
    with args.input.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != input_hash:
            raise ValueError("benchmark input changed during measurement")
    return {"platform": platform.platform(), "input_sha256": input_hash, "runs": runs}


def _report(
    args: argparse.Namespace, execute: Callable[[argparse.Namespace], dict[str, object]]
) -> int:
    try:
        print(json.dumps(execute(args), sort_keys=True))
    except KeyboardInterrupt:
        print(json.dumps({"error": "codec command interrupted"}), file=sys.stderr)
        return 130
    except (
        OSError,
        ValueError,
        RuntimeError,
        ImportError,
        EOFError,
        OverflowError,
        wave.Error,
    ) as error:
        print(json.dumps({"error": str(error), "type": type(error).__name__}), file=sys.stderr)
        return 1
    return 0


def _add_source(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "input", type=Path, help="numeric .npy, PCM16 .wav, event .csv or legacy RLE"
    )
    parser.add_argument("--codec", required=True, help="catalogue codec name")
    parser.add_argument("--variant", help="catalogue variant (defaults to the codec's default)")
    parser.add_argument("--frame-rows", type=int, help="maximum source rows per packet")
    parser.add_argument("--sample-rate", type=int, help="sample rate in Hz for an NPY recording")
    parser.add_argument("--rows", type=int, help="total sample rows for event CSV or legacy RLE")
    parser.add_argument("--channels", type=int, help="total channels for event CSV or legacy RLE")
    parser.add_argument("--legacy-rle", action="store_true", help="import legacy BCI spike bytes")
    parser.add_argument(
        "--option",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="repeatable codec option",
    )


def add_codec_commands(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the installed recording codec commands and their file options."""
    codec = subparsers.add_parser("codec", help="discover codecs and process recording files")
    codec.set_defaults(handler=lambda args: _report(args, _discover))
    commands = codec.add_subparsers(dest="codec_action", metavar="ACTION")
    discover = commands.add_parser("discover", help="list catalogue codecs and their variants")
    discover.set_defaults(handler=lambda args: _report(args, _discover))

    inspect = commands.add_parser("inspect", help="inspect catalogue availability or file metadata")
    target = inspect.add_mutually_exclusive_group(required=True)
    target.add_argument("codec", nargs="?", help="catalogue codec name")
    target.add_argument("--file", type=Path, help="file written by codec encode")
    inspect.add_argument("--variant", help="variant to probe")
    inspect.set_defaults(handler=lambda args: _report(args, _inspect))

    encode = commands.add_parser("encode", help="encode a recording to framed codec packets")
    _add_source(encode)
    encode.add_argument("output", type=Path)
    encode.add_argument(
        "--force", action="store_true", help="replace the destination after success"
    )
    encode.add_argument("--backend", choices=(*_BACKENDS, "gpu"), default="auto")
    encode.set_defaults(handler=lambda args: _report(args, _encode))

    decode = commands.add_parser("decode", help="verify and decode packet data to NumPy arrays")
    decode.add_argument("input", type=Path)
    decode.add_argument("output", type=Path, help=".npy for one field or .npz for named arrays")
    decode.add_argument(
        "--field",
        choices=("all", "raster", "waveform", "spikes", "events", "power", "offsets"),
        default="all",
    )
    decode.add_argument(
        "--force", action="store_true", help="replace the destination after success"
    )
    decode.add_argument("--backend", choices=_BACKENDS, default="auto")
    decode.set_defaults(handler=lambda args: _report(args, _read))

    verify = commands.add_parser("verify", help="check integrity and decode every packet")
    verify.add_argument("input", type=Path)
    verify.add_argument("--backend", choices=_BACKENDS, default="auto")
    verify.set_defaults(handler=lambda args: _report(args, _read), output=None)

    benchmark = commands.add_parser("benchmark", help="time matched codec round trips")
    _add_source(benchmark)
    benchmark.add_argument("--repetitions", type=int, default=3)
    benchmark.add_argument("--backend", choices=("compare", *_BACKENDS, "gpu"), default="compare")
    benchmark.set_defaults(handler=lambda args: _report(args, _benchmark))
