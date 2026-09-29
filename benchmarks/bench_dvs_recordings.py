# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Guarded DVS reader measurements

"""Measure all five actual DVS read calls after exact stored-value parity.

File I/O, result allocation and native startup/transport are timed. Each Julia
call starts a fresh runtime and includes its compilation; warmup only warms host
caches. This is a reader measurement, not training or target latency/energy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from sc_neurocore.accel.backend_selection import current_cpu
from sc_neurocore.accel.dvs_native import native_executable
from sc_neurocore.accel.dvs_recordings import read_dvs_recording

Backend = Literal["numpy", "rust", "mojo", "julia", "go"]
_BACKENDS: tuple[Backend, ...] = ("numpy", "rust", "mojo", "julia", "go")


def benchmark_recording(
    recording: Path, *, corpus_kind: str, repetitions: int = 25, warmup: int = 3
) -> dict[str, Any]:
    """Measure complete public read calls and retain hashes of source, input and artifacts.

    Parameters
    ----------
    recording : Path
        Existing single four-column NPY recording; never generated here.
    corpus_kind : str
        Explicit caller provenance label, without publisher qualification.
    repetitions : int
        Positive measured call count per backend.
    warmup : int
        Non-negative untimed call count per backend.

    Returns
    -------
    dict
        Exact parity receipts, host/source provenance and every measured sample.

    Raises
    ------
    ValueError
        Invalid counts/provenance or invalid reference input.
    RuntimeError
        Any explicitly required native backend is undeclared or refuses input.
    AssertionError
        A measured backend differs in any returned float64 storage bit.
    """
    if repetitions < 1 or warmup < 0:
        raise ValueError("repetitions must be positive and warmup non-negative")
    if corpus_kind not in ("generated-format-fixture", "publisher-recording", "operator-converted"):
        raise ValueError("explicit recording provenance is required")
    recording = recording.resolve(strict=True)
    raw_digest = hashlib.sha256(recording.read_bytes()).hexdigest()
    golden = read_dvs_recording(recording, backend="numpy")
    artifacts = {}
    for backend in _BACKENDS[1:]:
        assert backend != "numpy"
        executable = native_executable(backend)
        if executable is None:
            raise RuntimeError(f"{backend} DVS executable must be explicitly declared")
        artifacts[backend] = {
            "path": str(executable),
            "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
        }
    root = Path(__file__).resolve().parents[1]
    sources = [
        Path(__file__),
        root / "src/sc_neurocore/accel/dvs_native.py",
        root / "src/sc_neurocore/accel/dvs_recordings.py",
    ]
    for pattern in (
        "go/services/loaders/**/*.go",
        "go/services/loaders/*.[ch]",
        "rust/safety/dvs_native/Cargo.*",
        "rust/safety/dvs_native/build.rs",
        "rust/safety/dvs_native/src/*",
        "go/go.mod",
        "go/go.sum",
        "julia/datasets/dvs*.jl",
        "mojo/kernels/dvs_*.mojo",
    ):
        sources.extend(p for p in (root / "src/sc_neurocore/accel").glob(pattern) if p.is_file())
    source_hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(set(sources))
    }
    backends = {}
    for backend in _BACKENDS:
        actual = read_dvs_recording(recording, backend=backend)
        np.testing.assert_array_equal(actual.view(np.uint64), golden.view(np.uint64))
        for _ in range(warmup):
            read_dvs_recording(recording, backend=backend)
        samples_ms = []
        for _ in range(repetitions):
            started = time.perf_counter_ns()
            result = read_dvs_recording(recording, backend=backend)
            samples_ms.append((time.perf_counter_ns() - started) / 1_000_000)
            np.testing.assert_array_equal(result.view(np.uint64), golden.view(np.uint64))
        backends[backend] = {
            "available": True,
            "used": True,
            "parity": "exact-float64-storage-bits",
            "median_call_ms": statistics.median(samples_ms),
            "samples_ms": samples_ms,
        }
    if hashlib.sha256(recording.read_bytes()).hexdigest() != raw_digest:
        raise RuntimeError("recording changed during measurement")
    if any(
        hashlib.sha256((root / name).read_bytes()).hexdigest() != digest
        for name, digest in source_hashes.items()
    ):
        raise RuntimeError("reader source changed during measurement")
    if any(
        hashlib.sha256(Path(receipt["path"]).read_bytes()).hexdigest() != receipt["sha256"]
        for receipt in artifacts.values()
    ):
        raise RuntimeError("native executable changed during measurement")
    return {
        "kernel": "dvs-recording",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "meta": {
            "cpu": current_cpu(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "load_average": os.getloadavg(),
            "host_isolated": False,
        },
        "recording": {
            "name": recording.name,
            "sha256": raw_digest,
            "bytes": recording.stat().st_size,
            "events": len(golden),
            "corpus_kind_declared": corpus_kind,
            "publisher_acceptance": False,
        },
        "measurement": {
            "scope": "public-read-call-including-file-startup-and-transport",
            "repetitions": repetitions,
            "warmup": warmup,
            "julia_compilation_in_each_call": True,
            "order": list(_BACKENDS),
        },
        "source_sha256": source_hashes,
        "native_artifacts": artifacts,
        "backends": backends,
    }


def main() -> None:
    """Write new benchmark evidence for explicitly supplied recordings and executables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--corpus-kind",
        required=True,
        choices=("generated-format-fixture", "publisher-recording", "operator-converted"),
    )
    parser.add_argument("--repetitions", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    result = benchmark_recording(
        args.recording,
        corpus_kind=args.corpus_kind,
        repetitions=args.repetitions,
        warmup=args.warmup,
    )
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
