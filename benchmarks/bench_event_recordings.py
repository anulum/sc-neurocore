# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Event-recording decoder comparison

"""Measure actual N-MNIST decoder calls after exact parity against NumPy.

The caller supplies a recording and declares its provenance. Reading the file,
parity checks and warmup are outside the timed region; output allocation and the
native boundary are included. These measurements do not qualify a dataset or
measure training, hardware latency or energy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from sc_neurocore.accel.backend_selection import current_cpu
from sc_neurocore.accel.event_recordings import decode_nmnist_recording


def benchmark_recording(
    recording: Path, *, corpus_kind: str, repetitions: int = 25, warmup: int = 3
) -> dict[str, Any]:
    """Compare the executable Rust, Mojo, Julia, Go and NumPy paths on identical file bytes.

    Parameters
    ----------
    recording : Path
        One complete published-format binary recording. Nothing is generated.
    corpus_kind : str
        Caller-declared provenance, retained without a publisher acceptance claim.
    repetitions : int
        Positive number of timed decoder calls for each backend.
    warmup : int
        Non-negative count of untimed calls after exact output parity.

    Returns
    -------
    dict
        Host, file digest and measured median call times for the supported paths.

    Raises
    ------
    ValueError
        A sample is incomplete or the repetition counts are invalid.
    RuntimeError
        An explicitly requested native library is unavailable or refuses input.
    AssertionError
        Any backend differs from the NumPy recording.
    """
    if repetitions < 1 or warmup < 0:
        raise ValueError("repetitions must be positive and warmup non-negative")
    raw = recording.read_bytes()
    golden = decode_nmnist_recording(raw, backend="numpy")
    backends: dict[str, Any] = {}
    for backend in ("numpy", "rust", "mojo", "julia", "go"):
        np.testing.assert_array_equal(decode_nmnist_recording(raw, backend=backend), golden)
        for _ in range(warmup):
            decode_nmnist_recording(raw, backend=backend)
        samples_ms = []
        for _ in range(repetitions):
            started = time.perf_counter_ns()
            decode_nmnist_recording(raw, backend=backend)
            samples_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
        backends[backend] = {
            "available": True,
            "used": True,
            "parity": "exact",
            "median_call_ms": statistics.median(samples_ms),
            "samples_ms": samples_ms,
        }
    return {
        "kernel": "nmnist-recording",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "meta": {
            "cpu": current_cpu(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "recording": {
            "name": recording.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "events": len(golden),
            "corpus_kind_declared": corpus_kind,
            "publisher_acceptance": False,
        },
        "measurement": {"scope": "decode-call", "repetitions": repetitions, "warmup": warmup},
        "backends": backends,
    }


def main() -> None:
    """Write a measured comparison for the supplied recording and native artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--corpus-kind",
        choices=("generated-format-fixture", "publisher-recording", "operator-converted"),
        required=True,
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
