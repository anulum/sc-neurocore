# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Framed spike-raster packet benchmark

"""Compare Python and Rust on complete SSCF packets and bounded pull streams."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
import json
import statistics
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from typing import Any

import numpy as np

from bench_isi_codec import Raster, _hash, _module_hash
from bench_predictive_codec import _host, _recordings
from isi_codec_timing import _peak_rss, summarize_samples
from sc_neurocore.spike_codec import StreamingSpikeCodec

_BACKENDS = ("python", "rust")
_OPERATIONS = ("encode", "decode", "roundtrip", "frame_roundtrip", "stream_roundtrip")


@dataclass(frozen=True)
class _Case:
    raster: Raster
    window: int
    chunk_size: int

    def chunks(self) -> Iterator[Raster]:
        for start in range(0, len(self.raster), self.chunk_size):
            yield self.raster[start : start + self.chunk_size]

    @property
    def first_frame(self) -> Raster:
        return self.raster[: self.window]


def _digest(raster: Raster) -> str:
    return sha256(raster.tobytes(order="C")).hexdigest()


def _frames_digest(frames: list[bytes]) -> str:
    digest = sha256()
    for frame in frames:
        digest.update(len(frame).to_bytes(8, "big"))
        digest.update(frame)
    return digest.hexdigest()


def _validate(codec: StreamingSpikeCodec, case: _Case, packet: bytes, frames: list[bytes]) -> None:
    decoded = codec.decompress(packet)
    if decoded.dtype != case.raster.dtype or not np.array_equal(decoded, case.raster):
        raise ValueError("whole SSCF output changed the raster")
    offset = 0
    for frame in codec.iter_decompress(frames):
        finish = offset + frame.shape[0]
        if frame.dtype != case.raster.dtype or not np.array_equal(
            frame, case.raster[offset:finish]
        ):
            raise ValueError("independent frame changed the raster")
        offset = finish
    if offset != case.raster.shape[0]:
        raise ValueError("stream did not conserve rows")


def _sample(codec: StreamingSpikeCodec, case: _Case, packet: bytes) -> tuple[int, ...]:
    start = perf_counter_ns()
    codec.compress(case.raster)
    encoded = perf_counter_ns()
    codec.decompress(packet)
    decoded = perf_counter_ns()
    codec.decompress(codec.compress(case.raster)[0])
    roundtrip = perf_counter_ns()
    codec.decompress_frame(codec.compress_frame(case.first_frame))
    frame = perf_counter_ns()
    restored = sum(len(part) for part in codec.iter_decompress(codec.iter_compress(case.chunks())))
    end = perf_counter_ns()
    if restored != len(case.raster):
        raise ValueError("timed bounded pull did not conserve rows")
    return encoded - start, decoded - encoded, roundtrip - decoded, frame - roundtrip, end - frame


def _first_call(args: argparse.Namespace) -> None:
    start = perf_counter_ns()
    raster = np.load(args.raster, allow_pickle=False)
    codec = StreamingSpikeCodec(args.window_size, backend=args.backend)
    loaded = perf_counter_ns()
    packet, _ = codec.compress(raster)
    encoded = perf_counter_ns()
    decoded = codec.decompress(packet)
    finished = perf_counter_ns()
    print(
        json.dumps(
            {
                "load_and_construct_ns": loaded - start,
                "encode_ns": encoded - loaded,
                "decode_ns": finished - encoded,
                "roundtrip_ns": finished - loaded,
                "peak_rss_bytes": _peak_rss(),
                "packet_sha256": sha256(packet).hexdigest(),
                "decoded_sha256": _digest(decoded),
            }
        )
    )


def _cold(
    case: _Case, raster_file: Path, packet_hash: str, count: int
) -> dict[str, list[dict[str, object]]]:
    expected = _digest(case.raster)
    results: dict[str, list[dict[str, object]]] = {name: [] for name in _BACKENDS}
    for trial in range(count):
        order = _BACKENDS if trial % 2 == 0 else tuple(reversed(_BACKENDS))
        for name in order:
            command = [
                sys.executable,
                "-B",
                __file__,
                "first-call",
                str(raster_file),
                name,
                str(case.window),
            ]
            start = perf_counter_ns()
            result: dict[str, object] = json.loads(
                subprocess.check_output(command, text=True, timeout=240)
            )
            result["process_wall_ns"] = perf_counter_ns() - start
            if result["packet_sha256"] != packet_hash or result["decoded_sha256"] != expected:
                raise ValueError(f"fresh {name} process changed packet or raster")
            results[name].append(result)
    return results


def _compare_row(
    case: _Case, label: str, raster_file: Path, args: argparse.Namespace
) -> dict[str, Any]:
    codecs = {name: StreamingSpikeCodec(case.window, backend=name) for name in _BACKENDS}
    reference = codecs["python"].compress(case.raster)
    frames = list(codecs["python"].iter_compress(case.chunks()))
    for codec in codecs.values():
        if codec.compress(case.raster) != reference:
            raise ValueError("SSCF packet or statistics parity failed")
        if list(codec.iter_compress(case.chunks())) != frames:
            raise ValueError("standalone frame parity failed")
        _validate(codec, case, reference[0], frames)

    samples: dict[str, dict[str, list[int]]] = {
        name: {op: [] for op in _OPERATIONS} for name in _BACKENDS
    }
    for trial in range(args.warmups + args.repetitions):
        order = _BACKENDS if trial % 2 == 0 else tuple(reversed(_BACKENDS))
        for name in order:
            durations = _sample(codecs[name], case, reference[0])
            if trial >= args.warmups:
                for op, elapsed in zip(_OPERATIONS, durations, strict=True):
                    samples[name][op].append(elapsed)

    for codec in codecs.values():
        if (
            codec.compress(case.raster) != reference
            or list(codec.iter_compress(case.chunks())) != frames
        ):
            raise ValueError("packet bytes changed during measurement")
        _validate(codec, case, reference[0], frames)

    packet_hash = sha256(reference[0]).hexdigest()
    return {
        "recording": label,
        "shape": list(case.raster.shape),
        "window_size": case.window,
        "chunk_size": case.chunk_size,
        "acquisition_window_ns": case.window / args.sample_rate_hz * 1e9,
        "raster_sha256": _digest(case.raster),
        "sscf_packet_sha256": packet_hash,
        "sscf_packet_bytes": len(reference[0]),
        "standalone_stream_sha256": _frames_digest(frames),
        "standalone_stream_bytes": sum(map(len, frames)),
        "standalone_frame_count": len(frames),
        "final_frame_rows": ((len(case.raster) - 1) % case.window + 1 if frames else 0),
        "max_sscf_frame_bytes": reference[1].max_frame_bytes,
        "samples_ns": samples,
        "summary": {
            name: {
                op: summarize_samples(
                    times, case.first_frame.size if op == "frame_roundtrip" else case.raster.size
                )
                for op, times in series.items()
            }
            for name, series in samples.items()
        },
        "python_over_rust_paired_median": {
            op: statistics.median(
                py / rust
                for py, rust in zip(samples["python"][op], samples["rust"][op], strict=True)
            )
            for op in _OPERATIONS
        },
        "fresh_process": _cold(case, raster_file, packet_hash, args.cold_repetitions),
    }


def _compare(args: argparse.Namespace) -> None:
    start = datetime.now(UTC).isoformat()
    before = _host()
    rows = []
    with TemporaryDirectory(prefix="raster-stream-benchmark-") as temporary:
        raster_file = Path(temporary) / "raster.npy"
        for label, raster in _recordings(args.shd, args.recording):
            np.save(raster_file, raster, allow_pickle=False)
            for window in args.window_size:
                case = _Case(raster, window, args.chunk_size)
                rows.append(_compare_row(case, label, raster_file, args))
    modules = (
        "sc_neurocore.spike_codec.streaming_codec",
        "sc_neurocore.spike_codec.raster_frame",
        "sc_neurocore_engine.sc_neurocore_engine",
    )
    harness = (
        "bench_raster_stream.py",
        "bench_predictive_codec.py",
        "bench_isi_codec.py",
        "isi_codec_timing.py",
    )
    result = {
        "schema_version": 1,
        "started_utc": start,
        "finished_utc": datetime.now(UTC).isoformat(),
        "hardware_before": before,
        "hardware_after": _host(),
        "dataset": "synthetic" if args.shd is None else "Spiking Heidelberg Digits",
        "dataset_sha256": None if args.shd is None else _hash(args.shd),
        "recording_indices": sorted(args.recording),
        "sample_rate_hz": args.sample_rate_hz,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "cold_repetitions": args.cold_repetitions,
        "modules_sha256": {name: _module_hash(name) for name in modules},
        "harness_sha256": {name: _hash(Path(__file__).with_name(name)) for name in harness},
        "rows": rows,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    """Run the comparison or one fresh-interpreter round trip."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--window-size", type=int, action="append", required=True)
    compare.add_argument("--chunk-size", type=int, default=37)
    compare.add_argument("--sample-rate-hz", type=int, required=True)
    compare.add_argument("--warmups", type=int, default=2)
    compare.add_argument("--repetitions", type=int, default=9)
    compare.add_argument("--cold-repetitions", type=int, default=5)
    compare.add_argument("--shd", type=Path)
    compare.add_argument("--recording", type=int, action="append", default=[])
    compare.add_argument("--output", type=Path, required=True)
    first = commands.add_parser("first-call")
    first.add_argument("raster", type=Path)
    first.add_argument("backend", choices=_BACKENDS)
    first.add_argument("window_size", type=int)
    args = parser.parse_args()
    if args.command == "first-call":
        _first_call(args)
        return
    if len(set(args.window_size)) != len(args.window_size) or not all(
        1 <= window <= 65535 for window in args.window_size
    ):
        parser.error("select distinct window sizes in 1..65535")
    if args.chunk_size < 1 or args.sample_rate_hz < 1:
        parser.error("chunk size and sample rate must be positive")
    if bool(args.shd) != bool(args.recording) or min(args.recording, default=0) < 0:
        parser.error("use --shd with explicit nonnegative --recording indices")
    if len(set(args.recording)) != len(args.recording):
        parser.error("select each recording once")
    if args.warmups < 1 or min(args.repetitions, args.cold_repetitions) < 5:
        parser.error("use at least one warmup and five repetitions per timing group")
    if args.output.exists():
        parser.error("output already exists")
    _compare(args)


if __name__ == "__main__":
    main()
