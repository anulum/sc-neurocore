# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ISI timing and fresh process measurements

"""Measure public ISI packets with fresh interpreters and resident input."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from numpy.typing import NDArray

from sc_neurocore.spike_codec import SpikeCodec

Raster = NDArray[np.int8]
Timings = dict[str, dict[str, list[int]]]
OPERATIONS = ("encode", "decode", "roundtrip")


def _sample(work: Callable[[], object]) -> int:
    start = perf_counter_ns()
    work()
    return perf_counter_ns() - start


def warmed_samples(
    codecs: tuple[SpikeCodec, SpikeCodec],
    raster: Raster,
    packet: bytes,
    decoded_shape: tuple[int, int],
    warmups: int,
    repetitions: int,
) -> Timings:
    """Time complete public calls and alternate backend order on one raster."""

    def roundtrip(codec: SpikeCodec) -> Raster:
        encoded, _ = codec.compress(raster)
        return codec.decompress(encoded, *decoded_shape)

    def sample(codec: SpikeCodec) -> tuple[int, int, int]:
        return (
            _sample(lambda: codec.compress(raster)),
            _sample(lambda: codec.decompress(packet, *decoded_shape)),
            _sample(lambda: roundtrip(codec)),
        )

    timings: Timings = {codec.backend: {name: [] for name in OPERATIONS} for codec in codecs}
    for repetition in range(warmups + repetitions):
        pair = codecs if repetition % 2 == 0 else codecs[::-1]
        for codec in pair:
            elapsed = sample(codec)
            if repetition >= warmups:
                for series, duration in zip(timings[codec.backend].values(), elapsed, strict=True):
                    series.append(duration)
    return timings


def summarize_samples(series: list[int], cells: int) -> dict[str, float | int | None]:
    """Report quantiles supported by sample count and throughput of the median call."""
    median = statistics.median(series)
    percentiles = (
        statistics.quantiles(series, n=100, method="inclusive") if len(series) >= 40 else []
    )
    return {
        "p50_ns": median,
        "p95_ns": percentiles[94] if percentiles else None,
        "p99_ns": percentiles[98] if len(series) >= 200 else None,
        "median_absolute_deviation_ns": statistics.median(
            abs(sample - median) for sample in series
        ),
        "raster_values_per_second": cells * 1e9 / median if median else None,
        "source_bytes_per_second": cells * 1e9 / median if median else None,
    }


def fresh_process_samples(
    raster: Raster,
    expected: Raster,
    packet: bytes,
    entropy: str,
    precision: int,
    repetitions: int,
) -> dict[str, list[dict[str, object]]]:
    """Measure import and first calls using a separate installed consumer per sample."""
    samples: dict[str, list[dict[str, object]]] = {"python": [], "rust": []}
    packet_hash = hashlib.sha256(packet).hexdigest()
    raster_hash = hashlib.sha256(expected).hexdigest()
    with tempfile.TemporaryDirectory(prefix="isi-benchmark-") as directory:
        source = Path(directory) / "raster.npy"
        np.save(source, raster, allow_pickle=False)
        for repetition in range(repetitions):
            order = ("python", "rust") if repetition % 2 == 0 else ("rust", "python")
            for backend in order:
                command = [
                    sys.executable,
                    "-I",
                    str(Path(__file__).resolve()),
                    str(source),
                    backend,
                    entropy,
                    str(precision),
                ]
                started = perf_counter_ns()
                output = subprocess.check_output(command, text=True, timeout=120)
                elapsed = perf_counter_ns() - started
                sample = json.loads(output)
                if (
                    sample["packet_sha256"] != packet_hash
                    or sample["decoded_sha256"] != raster_hash
                ):
                    raise ValueError(f"fresh {backend} process produced different outputs")
                sample["process_wall_ns"] = elapsed
                samples[backend].append(sample)
    return samples


def _peak_rss() -> int | None:
    try:
        resource = import_module("resource")
    except ModuleNotFoundError:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def main() -> None:
    """Measure the first encode and decode in this process and write their digests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raster", type=Path)
    parser.add_argument("backend", choices=("python", "rust"))
    parser.add_argument("entropy", choices=("varint", "huffman", "auto"))
    parser.add_argument("precision", type=int)
    args = parser.parse_args()

    raster = np.load(args.raster, allow_pickle=False)
    mode = "lossless" if args.precision == 1 else "lossy"
    start = perf_counter_ns()
    codec = SpikeCodec(mode, args.precision, args.entropy, backend=args.backend)
    setup_ns = perf_counter_ns() - start
    start = perf_counter_ns()
    packet, _ = codec.compress(raster)
    encode_ns = perf_counter_ns() - start
    start = perf_counter_ns()
    decoded = codec.decompress(packet, raster.shape[0] // args.precision, raster.shape[1])
    decode_ns = perf_counter_ns() - start
    print(
        json.dumps(
            {
                "backend": codec.backend,
                "setup_ns": setup_ns,
                "encode_ns": encode_ns,
                "decode_ns": decode_ns,
                "roundtrip_ns": encode_ns + decode_ns,
                "peak_rss_bytes": _peak_rss(),
                "packet_sha256": hashlib.sha256(packet).hexdigest(),
                "decoded_sha256": hashlib.sha256(decoded).hexdigest(),
                "cpu_affinity": sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else None,
            },
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
