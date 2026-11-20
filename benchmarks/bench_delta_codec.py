# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Delta and ISI packet benchmark

"""Measure complete delta and ISI packets on the same rasters with Python and Rust."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns
from typing import Any

import numpy as np

from bench_isi_codec import Raster, _expected, _hash, _module_hash
from bench_predictive_codec import _host, _recordings
from isi_codec_timing import OPERATIONS, _peak_rss, summarize_samples
from sc_neurocore.spike_codec import DeltaSpikeCodec, SpikeCodec

Codec = DeltaSpikeCodec | SpikeCodec
_LANES = ("delta-python", "delta-rust", "isi-python", "isi-rust")


def _codec(lane: str, size: int, precision: int) -> Codec:
    family, backend = lane.split("-")
    mode = "lossless" if precision == 1 else "lossy"
    if family == "delta":
        return DeltaSpikeCodec(size, mode, precision, backend=backend)
    return SpikeCodec(mode=mode, timing_precision=precision, backend=backend)


def _expected_outputs(raster: Raster, size: int, precision: int) -> dict[str, Raster]:
    if precision == 1:
        return {"delta": raster, "isi": raster}
    binned = _expected(raster, precision)
    restored = binned.copy()
    for start in range(0, raster.shape[1], size):
        group = raster[:, start : start + size]
        reference = int(group.sum(axis=0).argmax())
        predictor = group[:, reference : reference + 1]
        binned_predictor = binned[:, start + reference]
        for offset, samples in enumerate(group.T):
            if offset != reference:
                residual = _expected((samples[:, None] ^ predictor), precision)[:, 0]
                restored[:, start + offset] = residual ^ binned_predictor
    return {"delta": restored, "isi": binned}


def _distortion(reference: Raster, actual: Raster) -> dict[str, int | float | None]:
    missing = int(np.count_nonzero((reference == 1) & (actual == 0)))
    extra = int(np.count_nonzero((reference == 0) & (actual == 1)))
    return {
        "reference_events": int(reference.sum()),
        "decoded_events": int(actual.sum()),
        "missing_events": missing,
        "extra_events": extra,
        "rmse_binary_occupancy": ((missing + extra) / actual.size) ** 0.5 if actual.size else None,
    }


def _validated_packets(
    codecs: dict[str, Codec], raster: Raster, outputs: dict[str, Raster]
) -> dict[str, bytes]:
    encoded = {name: codec.compress(raster) for name, codec in codecs.items()}
    packets = {name: result[0] for name, result in encoded.items()}
    for family in ("delta", "isi"):
        first, second = f"{family}-python", f"{family}-rust"
        if encoded[first] != encoded[second]:
            raise ValueError(f"{family} packet or statistics parity failed")
        expected = outputs[family]
        for reader, writer in ((first, second), (second, first)):
            actual = codecs[reader].decompress(packets[writer], *expected.shape)
            if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
                raise ValueError(f"{reader} did not match the expected raster")
    return packets


def _sample(
    codec: Codec, raster: Raster, packet: bytes, rows: int, channels: int
) -> tuple[int, int, int]:
    start = perf_counter_ns()
    codec.compress(raster)
    encoded = perf_counter_ns()
    codec.decompress(packet, rows, channels)
    decoded = perf_counter_ns()
    output, _ = codec.compress(raster)
    codec.decompress(output, rows, channels)
    return encoded - start, decoded - encoded, perf_counter_ns() - decoded


def _first_call(args: argparse.Namespace) -> None:
    start = perf_counter_ns()
    raster = np.load(args.raster, allow_pickle=False)
    codec = _codec(args.lane, args.group_size, args.precision)
    constructed = perf_counter_ns()
    packet, _ = codec.compress(raster)
    encoded = perf_counter_ns()
    decoded = codec.decompress(packet, raster.shape[0] // args.precision, raster.shape[1])
    finished = perf_counter_ns()
    print(
        json.dumps(
            {
                "load_and_construct_ns": constructed - start,
                "encode_ns": encoded - constructed,
                "decode_ns": finished - encoded,
                "roundtrip_ns": finished - constructed,
                "peak_rss_bytes": _peak_rss(),
                "packet_sha256": hashlib.sha256(packet).hexdigest(),
                "raster_sha256": hashlib.sha256(decoded.tobytes(order="C")).hexdigest(),
            }
        )
    )


def _cold_samples(
    raster_file: Path,
    size: int,
    packets: dict[str, bytes],
    output_hashes: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, list[dict[str, object]]]:
    samples: dict[str, list[dict[str, object]]] = {lane: [] for lane in _LANES}
    packet_hashes = {lane: hashlib.sha256(packet).hexdigest() for lane, packet in packets.items()}
    for iteration in range(args.cold_repetitions):
        pivot = iteration % len(_LANES)
        for lane in _LANES[pivot:] + _LANES[:pivot]:
            command = [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "first-call",
                str(raster_file),
                lane,
                str(size),
                str(args.precision),
            ]
            start = perf_counter_ns()
            result: dict[str, object] = json.loads(
                subprocess.check_output(command, text=True, timeout=240)
            )
            result["process_wall_ns"] = perf_counter_ns() - start
            if (
                result["packet_sha256"] != packet_hashes[lane]
                or result["raster_sha256"] != output_hashes[lane.split("-")[0]]
            ):
                raise ValueError(f"fresh {lane} process changed packet or raster")
            samples[lane].append(result)
    return samples


def _benchmark_row(
    raster: Raster, raster_file: Path, label: str, size: int, args: argparse.Namespace
) -> dict[str, Any]:
    outputs = _expected_outputs(raster, size, args.precision)
    codecs = {lane: _codec(lane, size, args.precision) for lane in _LANES}
    packets = _validated_packets(codecs, raster, outputs)
    series: dict[str, dict[str, list[int]]] = {
        lane: {op: [] for op in OPERATIONS} for lane in _LANES
    }
    for iteration in range(args.warmups + args.repetitions):
        pivot = iteration % len(_LANES)
        for lane in _LANES[pivot:] + _LANES[:pivot]:
            expected = outputs[lane.split("-")[0]]
            observed = _sample(codecs[lane], raster, packets[lane], *expected.shape)
            if iteration >= args.warmups:
                for op, duration in zip(OPERATIONS, observed, strict=True):
                    series[lane][op].append(duration)
    if packets != _validated_packets(codecs, raster, outputs):
        raise ValueError("packet bytes changed during measurement")
    raster_hash = hashlib.sha256(raster.tobytes(order="C")).hexdigest()
    output_hashes = {
        family: hashlib.sha256(data.tobytes(order="C")).hexdigest()
        for family, data in outputs.items()
    }
    return {
        "recording": label,
        "shape": list(raster.shape),
        "group_size": size,
        "raster_sha256": raster_hash,
        "decoded_shape": list(outputs["isi"].shape),
        "decoded_sha256": output_hashes,
        "occupancy_error_vs_direct_binning": {
            family: _distortion(outputs["isi"], data) for family, data in outputs.items()
        },
        "packet_bytes": {name: len(packet) for name, packet in packets.items()},
        "packet_sha256": {
            name: hashlib.sha256(packet).hexdigest() for name, packet in packets.items()
        },
        "samples_ns": series,
        "summary": {
            name: {op: summarize_samples(values, raster.size) for op, values in times.items()}
            for name, times in series.items()
        },
        "delta_python_over_rust_median": {
            op: statistics.median(
                python / rust
                for python, rust in zip(
                    series["delta-python"][op], series["delta-rust"][op], strict=True
                )
            )
            for op in OPERATIONS
        },
        "fresh_process": _cold_samples(raster_file, size, packets, output_hashes, args),
    }


def _compare(args: argparse.Namespace) -> None:
    started = datetime.now(UTC).isoformat()
    before = _host()
    rows = []
    with TemporaryDirectory(prefix="delta-codec-benchmark-") as directory:
        raster_file = Path(directory) / "raster.npy"
        for label, raster in _recordings(args.shd, args.recording):
            np.save(raster_file, raster, allow_pickle=False)
            for size in args.group_size:
                rows.append(_benchmark_row(raster, raster_file, label, size, args))
    modules = (
        "sc_neurocore.spike_codec.delta_codec",
        "sc_neurocore.spike_codec.codec",
        "sc_neurocore.spike_codec.isi_packet",
        "sc_neurocore.spike_codec.entropy",
        "sc_neurocore_engine.sc_neurocore_engine",
    )
    harness = (
        "bench_delta_codec.py",
        "bench_isi_codec.py",
        "bench_predictive_codec.py",
        "isi_codec_timing.py",
    )
    report = {
        "schema_version": 2,
        "started_utc": started,
        "finished_utc": datetime.now(UTC).isoformat(),
        "hardware_before": before,
        "hardware_after": _host(),
        "host_profile": args.host_profile,
        "dataset": "synthetic" if args.shd is None else "Spiking Heidelberg Digits",
        "dataset_sha256": None if args.shd is None else _hash(args.shd),
        "recording_indices": sorted(args.recording),
        "mode": "lossless" if args.precision == 1 else "lossy",
        "timing_precision": args.precision,
        "entropy": "auto",
        "raster_dtype": "int8",
        "raster_layout": "C",
        "time_unit": "synthetic sample" if args.shd is None else "1 ms source bin",
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "cold_repetitions": args.cold_repetitions,
        "modules_sha256": {name: _module_hash(name) for name in modules},
        "harness_sha256": {name: _hash(Path(__file__).with_name(name)) for name in harness},
        "rows": rows,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    """Run paired whole-packet measurements or one fresh-process call."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--group-size", type=int, action="append", required=True)
    compare.add_argument("--precision", type=int, default=1)
    compare.add_argument("--warmups", type=int, default=2)
    compare.add_argument("--repetitions", type=int, default=9)
    compare.add_argument("--cold-repetitions", type=int, default=5)
    compare.add_argument("--shd", type=Path)
    compare.add_argument("--recording", type=int, action="append", default=[])
    compare.add_argument("--host-profile", choices=("isolated", "shared"), required=True)
    compare.add_argument("--output", type=Path, required=True)
    first = commands.add_parser("first-call")
    first.add_argument("raster", type=Path)
    first.add_argument("lane", choices=_LANES)
    first.add_argument("group_size", type=int)
    first.add_argument("precision", type=int)
    args = parser.parse_args()
    if args.precision < 1:
        parser.error("precision must be positive")
    if args.command == "first-call":
        _first_call(args)
        return
    if len(set(args.group_size)) != len(args.group_size) or not all(
        1 <= n <= 256 for n in args.group_size
    ):
        parser.error("select distinct group sizes in 1..256")
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
