#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NumPy LUT packed-bit popcount benchmark
"""Measure an actual NumPy byte-LUT popcount over packed bitstreams."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


REPO_ROOT = Path(__file__).resolve().parents[1]
UTC = timezone.utc
DEFAULT_BIT_SIZES = (1 << 20, 1 << 26, 1 << 30)
POPCOUNT_LUT = np.array([value.bit_count() for value in range(256)], dtype=np.uint8)


def numpy_lut_popcount(packed_bytes: NDArray[np.generic]) -> int:
    """Count set bits through the real NumPy byte lookup-table baseline."""
    if packed_bytes.dtype != np.uint8 or packed_bytes.ndim != 1:
        raise ValueError("packed_bytes must be a one-dimensional uint8 array")
    return int(POPCOUNT_LUT[packed_bytes].sum(dtype=np.uint64))


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one source file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_optional(path: Path) -> str | None:
    """Read a one-line host fact when the Linux source exists."""
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def _git_head() -> str | None:
    """Return the exact source commit without making Git state changes."""
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _patterned_expected_popcount(byte_count: int) -> int:
    """Return the independent bit count for ``arange(..., dtype=uint8)``."""
    cycles, remainder = divmod(byte_count, 256)
    return cycles * 1024 + sum(value.bit_count() for value in range(remainder))


def _validate_bit_sizes(bit_sizes: tuple[int, ...]) -> None:
    """Reject empty, non-positive, or non-byte-aligned workloads."""
    if not bit_sizes:
        raise ValueError("at least one bit size is required")
    if any(bit_count <= 0 or bit_count % 8 for bit_count in bit_sizes):
        raise ValueError("bit sizes must be positive multiples of eight")


def run_benchmark(bit_sizes: tuple[int, ...], *, warmups: int, repeats: int) -> dict[str, Any]:
    """Run the packed-byte NumPy LUT benchmark and return evidence metadata."""
    _validate_bit_sizes(bit_sizes)
    if warmups < 0 or repeats < 1:
        raise ValueError("warmups must be non-negative and repeats must be positive")

    results: list[dict[str, Any]] = []
    for bit_count in bit_sizes:
        packed = np.arange(bit_count // 8, dtype=np.uint8)
        expected = _patterned_expected_popcount(bit_count // 8)
        assert numpy_lut_popcount(packed) == expected

        for _ in range(warmups):
            numpy_lut_popcount(packed)

        samples_ns: list[int] = []
        for _ in range(repeats):
            started = time.perf_counter_ns()
            observed = numpy_lut_popcount(packed)
            elapsed_ns = time.perf_counter_ns() - started
            if observed != expected:
                raise RuntimeError("NumPy LUT popcount changed during measurement")
            samples_ns.append(elapsed_ns)

        median_ns = statistics.median(samples_ns)
        results.append(
            {
                "bit_count": bit_count,
                "packed_bytes": bit_count // 8,
                "set_bits": expected,
                "samples_ns": samples_ns,
                "median_ns": median_ns,
                "throughput_gbit_s": bit_count / float(median_ns),
            }
        )

    source_paths = (
        Path("benchmarks/bench_bitstream_numpy_lut.py"),
        Path("engine/benches/bitstream_bench.rs"),
    )
    return {
        "schema": "sc-neurocore.bitstream-popcount-benchmark.v1",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_head(),
        "operation": "packed bitstream popcount",
        "implementation": "NumPy uint8 lookup table followed by uint64 sum",
        "input_representation": "one packed uint8 stores eight logical bits",
        "warmups": warmups,
        "repeats": repeats,
        "system": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cpu_affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
            "load_average": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
            "cpu_governor": _read_optional(
                Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
            ),
            "isolated_cpus": _read_optional(Path("/sys/devices/system/cpu/isolated")),
        },
        "source_hashes": {path.as_posix(): _sha256(REPO_ROOT / path) for path in source_paths},
        "results": results,
    }


def _parse_sizes(raw_sizes: str) -> tuple[int, ...]:
    """Parse a comma-separated list of integer bit counts."""
    try:
        return tuple(int(raw_size.strip()) for raw_size in raw_sizes.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("sizes must be comma-separated integers") from exc


def main() -> int:
    """Run the benchmark CLI and optionally write its structured result."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        type=_parse_sizes,
        default=DEFAULT_BIT_SIZES,
        help="comma-separated logical bit counts",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = run_benchmark(args.sizes, warmups=args.warmups, repeats=args.repeats)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
