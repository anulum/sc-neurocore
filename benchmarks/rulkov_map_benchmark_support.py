# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shared Rulkov benchmark measurement support

"""Source-bound measurement mechanics shared by both Rulkov identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt


N_STEPS = 2_000_000
CURRENT = 0.5
N_REPEATS = 5
REPO_ROOT = Path(__file__).resolve().parents[1]
BACKENDS = ("python", "rust", "julia", "go", "mojo")


class BatchModel(Protocol):
    """Minimum public model surface exercised by the benchmark."""

    def simulate(
        self, n_steps: int, current: float, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]: ...


@dataclass(frozen=True)
class BenchmarkSpec:
    """One Rulkov identity's runtime and provenance bindings."""

    benchmark: str
    model: str
    title: str
    event_semantics: str
    model_factory: Callable[[], BatchModel]
    backend_probes: dict[str, Callable[[], bool]]
    unavailable_reasons: dict[str, str]
    source_hash_paths: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes(paths: tuple[str, ...]) -> dict[str, object]:
    flat: dict[str, object] = {path: _sha256(REPO_ROOT / path) for path in paths}
    nested: dict[str, object] = {}
    for path, digest in flat.items():
        stem, suffix = path.rsplit(".", 1)
        nested[stem] = {suffix: digest}
    return {**flat, **nested}


def _measure(
    model_factory: Callable[[], BatchModel], backend: str
) -> tuple[float, float, npt.NDArray[np.float64], int]:
    model_factory().simulate(N_STEPS, CURRENT, backend=backend)
    timings_ms: list[float] = []
    trace: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
    events = 0
    for _ in range(N_REPEATS):
        started = time.perf_counter()
        trace, events = model_factory().simulate(N_STEPS, CURRENT, backend=backend)
        timings_ms.append((time.perf_counter() - started) * 1000.0)
    timings_ms.sort()
    return timings_ms[len(timings_ms) // 2], timings_ms[0], trace, int(events)


def _backend_status(spec: BenchmarkSpec) -> dict[str, tuple[bool, str]]:
    status: dict[str, tuple[bool, str]] = {}
    for backend in BACKENDS:
        available = spec.backend_probes[backend]()
        reason = "" if available else spec.unavailable_reasons.get(backend, "backend unavailable")
        status[backend] = available, reason
    return status


def run_benchmark(argv: list[str], spec: BenchmarkSpec) -> int:
    """Measure one identity, enforce parity availability, and write evidence."""
    parser = argparse.ArgumentParser(description=f"Five-runtime benchmark: {spec.title}.")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--allow-unavailable-backends",
        action="store_true",
        help="Record unavailable optional backends instead of failing the parity benchmark.",
    )
    args = parser.parse_args(argv)

    print(f"# {spec.title}")
    print(f"# Workload: {N_STEPS:,} steps, default parameters, current={CURRENT}")
    print(f"# Repeats per backend: {N_REPEATS}")
    print(f"# Python: {platform.python_version()}, NumPy: {np.__version__}")
    print(f"# platform: {platform.platform()}")
    print("# isolation: non-isolated (loaded workstation) — functional/regression evidence")
    print()

    status = _backend_status(spec)
    missing = [backend for backend, (available, _reason) in status.items() if not available]
    if missing and not args.allow_unavailable_backends:
        print("Missing required backend(s): " + ", ".join(missing))
        return 2

    reference: npt.NDArray[np.float64] | None = None
    python_median: float | None = None
    rows: list[dict[str, object]] = []
    for backend, (available, reason) in status.items():
        if not available:
            rows.append({"backend": backend, "skipped": True, "unavailable_reason": reason})
            continue
        median_ms, min_ms, trace, events = _measure(spec.model_factory, backend)
        if backend == "python":
            reference = trace
            python_median = median_ms
            parity = 0.0
        else:
            if reference is None:
                raise RuntimeError("Python reference must run before accelerated backends")
            parity = float(np.max(np.abs(trace - reference)))
        speedup = python_median / median_ms if python_median is not None else 1.0
        rows.append(
            {
                "backend": backend,
                "median_ms": median_ms,
                "min_ms": min_ms,
                "parity_max_abs_diff": parity,
                "speedup_vs_python": speedup,
                "spikes": events,
            }
        )

    report = {
        "benchmark": spec.benchmark,
        "model": spec.model,
        "event_semantics": spec.event_semantics,
        "workload": {"n_steps": N_STEPS, "current": CURRENT, "repeats": N_REPEATS},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "isolation": "non-isolated (loaded workstation)",
        },
        "results": rows,
        "backend_summary": {
            str(row["backend"]): row for row in rows if isinstance(row.get("backend"), str)
        },
        "source_hashes": _source_hashes(spec.source_hash_paths),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {args.json}")
    return 0


__all__ = ["BenchmarkSpec", "run_benchmark"]
