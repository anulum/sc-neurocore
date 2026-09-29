# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Five-runtime QCFS quantisation and derivative comparison

"""Time public QCFS forward and backward calls in NumPy, Rust, Go, Mojo and Julia.

Each runtime runs in a fresh process pinned to one CPU. Every timed response
must equal the NumPy bits before its sample counts; the report binds the QCFS
source bytes and the configured library bytes and is published atomically only
after all five runtimes pass.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np

from sc_neurocore.conversion.qcfs_dispatch import QCFSBackend, qcfs_backward, qcfs_forward

BACKENDS: tuple[QCFSBackend, ...] = ("numpy", "rust", "go", "mojo", "julia")
SIZES = (1, 64, 4096, 262144)
LIBRARIES = ("SC_NEUROCORE_QCFS_RUST_LIB", "SC_NEUROCORE_QCFS_GO_LIB", "SC_NEUROCORE_QCFS_MOJO_LIB")


def source_digests() -> dict[str, str]:
    """Bind the imported QCFS Python owners and every native counterpart by SHA-256.

    Returns
    -------
    dict of str to str
        Package-relative source paths and digests.
    """
    import sc_neurocore

    package = Path(sc_neurocore.__file__).resolve().parent
    names = [
        "conversion/qcfs_kernel.py",
        "conversion/qcfs_native.py",
        "conversion/qcfs_dispatch.py",
        "accel/rust/safety/qcfs.rs",
        "accel/rust/safety/qcfs_native/src/lib.rs",
        "accel/go/conversion/qcfs.go",
        "accel/go/conversion/qcfscshared/main.go",
        "accel/mojo/kernels/qcfs.mojo",
        "accel/julia/conversion/qcfs.jl",
    ]
    return {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in names}


def artifact_digests() -> dict[str, str]:
    """Bind each configured QCFS library by SHA-256 without building anything.

    Returns
    -------
    dict of str to str
        Configuration variable name to library digest.

    Raises
    ------
    KeyError, OSError
        A library configuration or its file is absent.
    """
    return {
        name: hashlib.sha256(Path(os.environ[name]).resolve(strict=True).read_bytes()).hexdigest()
        for name in LIBRARIES
    }


def workloads() -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Return the seeded forward/backward workloads for every size.

    Returns
    -------
    list of tuple
        Workload name, activations and upstream gradients.
    """
    rng = np.random.default_rng(2022)
    return [(f"n{size}", rng.normal(0.4, 0.6, size), rng.normal(0.0, 1.0, size)) for size in SIZES]


def respond(backend: QCFSBackend, operation: str, x: np.ndarray, upstream: np.ndarray) -> bytes:
    """Return one complete public response as bytes.

    Parameters
    ----------
    backend : str
        Runtime to call.
    operation : {'forward', 'backward'}
        Quantisation, or both derivative arrays concatenated.
    x, upstream : numpy.ndarray
        Workload activations and upstream gradients.

    Returns
    -------
    bytes
        The float64 response bytes.
    """
    if operation == "forward":
        return qcfs_forward(x, 8, 1.0, backend=backend).tobytes()
    inputs, thresholds = qcfs_backward(x, upstream, 8, 1.0, backend=backend)
    return inputs.tobytes() + thresholds.tobytes()


def measure(backend: QCFSBackend, samples: int, warmup: int) -> dict[str, object]:
    """Time every workload through the public API after checking its bits against NumPy.

    Parameters
    ----------
    backend : str
        Runtime to time.
    samples, warmup : int
        Recorded warm samples and preceding repetitions.

    Returns
    -------
    dict
        Raw call times, response digests and median milliseconds per workload.

    Raises
    ------
    RuntimeError
        A response differs from the NumPy bits.
    """
    cases = []
    for name, x, upstream in workloads():
        for operation in ("forward", "backward"):
            expected = respond("numpy", operation, x, upstream)
            timings = []
            for repeat in range(1 + warmup + samples):
                started = time.perf_counter_ns()
                response = respond(backend, operation, x, upstream)
                elapsed = time.perf_counter_ns() - started
                if response != expected:
                    raise RuntimeError(f"{backend} {operation} {name} differs from NumPy")
                if repeat > warmup:
                    timings.append(elapsed)
            cases.append(
                {
                    "workload": f"{operation}-{name}",
                    "elements": int(x.size),
                    "response_sha256": hashlib.sha256(expected).hexdigest(),
                    "samples_ns": timings,
                    "median_ms": statistics.median(timings) / 1e6,
                }
            )
    return {"backend": backend, "bit_parity": True, "cases": cases}


def main(argv: Sequence[str] | None = None) -> int:
    """Run all five runtimes and publish one atomic comparison report.

    Parameters
    ----------
    argv : sequence of str or None
        Command arguments; None reads the actual command line.

    Returns
    -------
    int
        Zero after every runtime passed and the report was written.

    Raises
    ------
    RuntimeError
        A runtime failed, or sources or libraries changed during the capture.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--worker", choices=BACKENDS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.samples < 3 or args.warmup < 1:
        parser.error("at least three samples and one warmup required")
    if args.cpu not in os.sched_getaffinity(0):
        parser.error("CPU must belong to the current allowed affinity")
    os.sched_setaffinity(0, {args.cpu})
    if args.worker is not None:
        print(json.dumps(measure(cast(QCFSBackend, args.worker), args.samples, args.warmup)))
        return 0
    sources, artifacts = source_digests(), artifact_digests()
    before = list(os.getloadavg())
    results = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for backend in BACKENDS:
        command = [sys.executable, str(Path(__file__).resolve()), "--output", str(args.output)]
        command += ["--cpu", str(args.cpu), "--samples", str(args.samples)]
        command += ["--warmup", str(args.warmup), "--worker", backend]
        completed = subprocess.run(
            command,
            env=dict(os.environ, GOMAXPROCS="1"),
            capture_output=True,
            text=True,
            timeout=600,
        )
        args.output.with_suffix(f".{backend}.stdout").write_text(completed.stdout)
        args.output.with_suffix(f".{backend}.stderr").write_text(completed.stderr)
        if completed.returncode:
            raise RuntimeError(f"{backend} QCFS comparison failed; see retained stdout/stderr")
        results[backend] = json.loads(completed.stdout)
    if sources != source_digests() or artifacts != artifact_digests():
        raise RuntimeError("QCFS sources or configured libraries changed during capture")
    cpu = next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    )
    payload = {
        "schema_version": "sc-neurocore.qcfs-comparison.v1",
        "meta": {
            "cpu": cpu,
            "cpu_affinity": [args.cpu],
            "kernel": platform.release(),
            "python": platform.python_version(),
            "numpy": importlib.metadata.version("numpy"),
            "juliacall": importlib.metadata.version("juliacall"),
            "samples": args.samples,
            "warmup": args.warmup,
            "steps": 8,
            "theta": 1.0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "load_average": before,
            "load_average_after": list(os.getloadavg()),
            "cpu_isolation": "one CPU affinity; host load is not isolated",
            "coverage_process_start_requested": bool(os.environ.get("COVERAGE_PROCESS_START")),
            "latency_boundary": (
                "public qcfs_forward/qcfs_backward call: admission, float64 copy, runtime "
                "selection and native transport included; interpreter startup and builds excluded"
            ),
        },
        "source_sha256": sources,
        "artifact_sha256": artifacts,
        "backends": results,
    }
    with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent, delete=False) as temporary:
        json.dump(payload, temporary, indent=2)
        temporary.write("\n")
        temporary.flush()
        os.fsync(temporary.fileno())
    os.replace(temporary.name, args.output)
    print(f"All five QCFS runtimes verified; wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
