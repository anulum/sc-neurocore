# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python and Rust ISI raster codec benchmark

"""Measure complete ISI encoding and decoding on identical resident rasters."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from sc_neurocore.spike_codec import SpikeCodec

from isi_codec_timing import OPERATIONS, fresh_process_samples, summarize_samples, warmed_samples

Raster = NDArray[np.int8]
_SEED = 185


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _module_hash(name: str) -> str:
    origin = import_module(name).__file__
    if origin is None:
        raise ValueError(f"benchmark module has no file to hash: {name}")
    return _hash(Path(origin))


def _synthetic() -> Iterator[tuple[str, Raster]]:
    rng = np.random.default_rng(_SEED)
    for label, shape, density in (
        ("sparse_short", (32, 8), 0.01),
        ("sparse_long", (8192, 64), 0.008),
        ("dense", (1024, 64), 0.3),
    ):
        yield label, (rng.random(shape) < density).astype(np.int8)
    bursts = (rng.random((1024, 1)) < 0.065) & (rng.random((1024, 64)) < 0.75)
    yield "correlated_bursts", bursts.astype(np.int8)


def _shd(path: Path) -> Iterator[tuple[str, Raster]]:
    h5py = import_module("h5py")
    with h5py.File(path, "r") as dataset:
        times, units = dataset["spikes/times"], dataset["spikes/units"]
        if len(times) != len(units) or len(times) == 0:
            raise ValueError("SHD times and units must describe the same nonempty split")
        for recording in range(len(times)):
            seconds = np.asarray(times[recording], dtype=np.float64)
            channels = np.asarray(units[recording], dtype=np.intp)
            if (
                seconds.shape != channels.shape
                or not np.isfinite(seconds).all()
                or np.any(seconds < 0)
                or np.any((channels < 0) | (channels >= 700))
            ):
                raise ValueError(f"invalid SHD event in recording {recording}")
            bins = np.floor(seconds * 1000).astype(np.intp)
            rows = int(bins.max()) + 1 if bins.size else 0
            raster = np.zeros((rows, 700), dtype=np.int8)
            raster[bins, channels] = 1
            yield str(recording), raster


def _expected(raster: Raster, precision: int) -> Raster:
    rows, channels = raster.shape
    bins = rows // precision
    return raster[: bins * precision].reshape(bins, precision, channels).any(axis=1).astype(np.int8)


def _validate(
    codecs: tuple[SpikeCodec, SpikeCodec], raster: Raster, expected: Raster
) -> tuple[bytes, str]:
    python_packet, python_stats = codecs[0].compress(raster)
    rust_packet, rust_stats = codecs[1].compress(raster)
    if python_packet != rust_packet or python_stats != rust_stats:
        raise ValueError("Python and Rust emitted different ISI packets")
    for decoder, packet in zip(codecs, (rust_packet, python_packet), strict=True):
        decoded = decoder.decompress(packet, *expected.shape)
        if decoded.dtype != expected.dtype or not np.array_equal(decoded, expected):
            raise ValueError("ISI cross-decoding changed the expected raster")
    return python_packet, hashlib.sha256(expected).hexdigest()


def _cohort_timing(rows: list[dict[str, Any]]) -> dict[str, object]:
    cells = sum(row["source_bytes"] for row in rows)
    totals = {
        backend: {
            operation: list(
                map(sum, zip(*(row["samples_ns"][backend][operation] for row in rows), strict=True))
            )
            for operation in OPERATIONS
        }
        for backend in ("python", "rust")
    }
    comparisons: dict[str, object] = {}
    for operation in OPERATIONS:
        pairs = [
            python_ns / rust_ns
            for python_ns, rust_ns in zip(
                totals["python"][operation], totals["rust"][operation], strict=True
            )
            if rust_ns
        ]
        if not pairs:
            raise ValueError("a benchmark operation has no measurable duration")
        quartiles = statistics.quantiles(pairs, n=4, method="inclusive")
        comparisons[operation] = {
            "paired_python_over_rust_median": statistics.median(pairs),
            "paired_python_over_rust_quartiles": quartiles,
        }
    return {
        "source_bytes": cells,
        "encoded_bytes": sum(row["packet_bytes"] for row in rows),
        "recordings": len(rows),
        "sum_of_resident_calls_ns": totals,
        "comparison": comparisons,
        "summary": {
            backend: {
                operation: summarize_samples(series, cells)
                for operation, series in operations.items()
            }
            for backend, operations in totals.items()
        },
    }


def main() -> None:
    """Write raw samples, per-record identity and hardware metadata to a new JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shd", type=Path, help="uncompressed SHD test HDF5; every recording is used"
    )
    parser.add_argument("--entropy", choices=("varint", "huffman", "auto"), default="auto")
    parser.add_argument("--precision", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--cold-repetitions", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.precision < 1 or args.warmups < 1 or min(args.repetitions, args.cold_repetitions) < 5:
        parser.error(
            "precision and warmups must be positive; at least five repetitions are required"
        )
    if args.output.exists():
        parser.error("output already exists")

    started = datetime.now(UTC).isoformat()
    mode = "lossless" if args.precision == 1 else "lossy"
    codecs = (
        SpikeCodec(mode, args.precision, args.entropy, backend="python"),
        SpikeCodec(mode, args.precision, args.entropy, backend="rust"),
    )
    inputs = _synthetic() if args.shd is None else _shd(args.shd)
    rows: list[dict[str, Any]] = []
    cold: dict[str, object] = {}
    for name, raster in inputs:
        expected = _expected(raster, args.precision)
        packet, output_hash = _validate(codecs, raster, expected)
        if not rows:
            cold = {
                "representative_recording": name,
                "samples": fresh_process_samples(
                    raster, expected, packet, args.entropy, args.precision, args.cold_repetitions
                ),
            }
        samples = warmed_samples(
            codecs, raster, packet, expected.shape, args.warmups, args.repetitions
        )
        checked, checked_hash = _validate(codecs, raster, expected)
        if checked != packet or checked_hash != output_hash:
            raise ValueError(f"timed computations changed the packet or decoded data for {name}")
        rows.append(
            {
                "name": name,
                "source_sha256": hashlib.sha256(raster).hexdigest(),
                "shape": raster.shape,
                "decoded_shape": expected.shape,
                "source_bytes": raster.nbytes,
                "source_events": int(np.count_nonzero(raster)),
                "decoded_events": int(np.count_nonzero(expected)),
                "packet_sha256": hashlib.sha256(packet).hexdigest(),
                "packet_bytes": len(packet),
                "decoded_sha256": output_hash,
                "samples_ns": samples,
                "ratio_bits_to_packet": raster.size / (8 * len(packet)),
                "warm_summary": {
                    lane: {
                        operation: summarize_samples(series, raster.size)
                        for operation, series in times.items()
                    }
                    for lane, times in samples.items()
                },
            }
        )

    report = {
        "schema_version": 2,
        "started_utc": started,
        "finished_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "python": sys.version,
        "numpy": np.__version__,
        "engine_sha256": _module_hash("sc_neurocore_engine.sc_neurocore_engine"),
        "harness_sha256": _hash(Path(__file__)),
        "timing_source_sha256": _hash(Path(__file__).with_name("isi_codec_timing.py")),
        "python_codec_sha256": {
            name: _module_hash(f"sc_neurocore.spike_codec.{name}")
            for name in ("codec", "entropy", "isi_packet")
        },
        "dataset": "synthetic" if args.shd is None else "Spiking Heidelberg Digits",
        "dataset_sha256": None if args.shd is None else _hash(args.shd),
        "rasterization": "int8 binary, 700 channels and floor(seconds * 1000)"
        if args.shd
        else None,
        "synthetic_seed": _SEED if args.shd is None else None,
        "entropy": args.entropy,
        "timing_precision": args.precision,
        "mode": mode,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "cold_repetitions": args.cold_repetitions,
        "fresh_process": cold,
        "warm_cohort": _cohort_timing(rows),
        "rows": rows,
    }
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
