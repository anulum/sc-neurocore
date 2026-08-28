#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SC clipped rational-recovery map multi-language benchmark

"""Measure the retained rational-recovery map across all five runtimes.

The result is functional local-regression evidence from a loaded workstation,
not an isolated-core performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os as _os
import platform
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt

from sc_neurocore.neurons.models import sc_clipped_rational_recovery_map
from sc_neurocore.neurons.models.sc_clipped_rational_recovery_map import (
    SCClippedRationalRecoveryMapNeuron,
)

N_STEPS = 2_000_000
CURRENT = 0.0
N_REPEATS = 5
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASH_PATHS = (
    "benchmarks/bench_sc_clipped_rational_recovery_map.py",
    "src/sc_neurocore/neurons/models/sc_clipped_rational_recovery_map.py",
    "engine/src/neurons/sc_clipped_rational_recovery_map.rs",
    "src/sc_neurocore/accel/go/neurons/sc_clipped_rational_recovery_map/sc_clipped_rational_recovery_map.go",
    "src/sc_neurocore/accel/julia/neurons/sc_clipped_rational_recovery_map.jl",
    "src/sc_neurocore/accel/mojo/neurons/sc_clipped_rational_recovery_map.mojo",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> dict[str, object]:
    flat: dict[str, object] = {path: _sha256(REPO_ROOT / path) for path in SOURCE_HASH_PATHS}
    nested: dict[str, object] = {}
    for path, digest in flat.items():
        stem, suffix = path.rsplit(".", 1)
        nested[stem] = {suffix: digest}
    return {**flat, **nested}


def _probe_rust() -> tuple[bool, str]:
    available = sc_clipped_rational_recovery_map._HAS_RUST
    return available, "" if available else "engine wheel lacks the symbol"


def _probe_julia() -> tuple[bool, str]:
    available = sc_clipped_rational_recovery_map._ensure_julia_loaded()
    return available, "" if available else "juliacall/.jl unavailable"


def _probe_go() -> tuple[bool, str]:
    available = sc_clipped_rational_recovery_map._ensure_go_loaded()
    reason = "accel/go rational-recovery shared library not built"
    return available, "" if available else reason


def _probe_mojo() -> tuple[bool, str]:
    if not _os.path.isfile(_os.path.expanduser("~/.pixi/bin/mojo")):
        return False, "mojo binary not at ~/.pixi/bin/mojo"
    available = sc_clipped_rational_recovery_map._ensure_mojo_loaded()
    reason = "accel/mojo rational-recovery shared library not built"
    return available, "" if available else reason


def _run(backend: str) -> tuple[float, float, npt.NDArray[np.float64], int]:
    SCClippedRationalRecoveryMapNeuron().simulate(N_STEPS, CURRENT, backend=backend)
    timings_ms: list[float] = []
    trace: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
    events = 0
    for _ in range(N_REPEATS):
        started = time.perf_counter()
        trace, events = SCClippedRationalRecoveryMapNeuron().simulate(
            N_STEPS, CURRENT, backend=backend
        )
        timings_ms.append((time.perf_counter() - started) * 1000.0)
    timings_ms.sort()
    return timings_ms[len(timings_ms) // 2], timings_ms[0], trace, int(events)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="SC rational-recovery map benchmark.")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--allow-unavailable-backends",
        action="store_true",
        help="Record unavailable backends instead of failing the parity benchmark.",
    )
    args = parser.parse_args(argv)

    print("# SC clipped rational-recovery map N-step benchmark")
    print(f"# Workload: {N_STEPS:,} retained-default steps, current={CURRENT}")
    print(f"# Repeats per backend: {N_REPEATS}")
    print(f"# Python: {platform.python_version()}, NumPy: {np.__version__}")
    print(f"# platform: {platform.platform()}")
    print("# isolation: non-isolated (loaded workstation) — functional/regression evidence")
    print()

    backends = {
        "python": (True, ""),
        "rust": _probe_rust(),
        "julia": _probe_julia(),
        "go": _probe_go(),
        "mojo": _probe_mojo(),
    }
    missing = [name for name, (available, _reason) in backends.items() if not available]
    if missing and not args.allow_unavailable_backends:
        print("Missing required backend(s): " + ", ".join(missing))
        return 2

    reference: npt.NDArray[np.float64] | None = None
    python_median: float | None = None
    rows: list[dict[str, object]] = []
    for name, (available, reason) in backends.items():
        if not available:
            rows.append({"backend": name, "skipped": True, "unavailable_reason": reason})
            continue
        median_ms, min_ms, trace, events = _run(name)
        if name == "python":
            reference = trace
            python_median = median_ms
            parity = 0.0
        else:
            if reference is None:
                raise RuntimeError("Python reference must run before accelerated backends")
            parity = float(np.max(np.abs(trace - reference)))
        speedup = python_median / median_ms if python_median and median_ms > 0 else float("nan")
        print(
            f"{name:<8} median={median_ms:>10.2f} ms min={min_ms:>10.2f} ms "
            f"delta={parity:.2e} speedup={speedup:.2f}x events={events}"
        )
        rows.append(
            {
                "backend": name,
                "median_ms": median_ms,
                "min_ms": min_ms,
                "parity_max_abs_diff": parity,
                "speedup_vs_python": speedup,
                "spikes": events,
            }
        )

    report = {
        "benchmark": "sc_clipped_rational_recovery_map_simulate",
        "workload": {
            "n_steps": N_STEPS,
            "profile": "retained defaults",
            "current": CURRENT,
            "repeats": N_REPEATS,
        },
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
        "source_hashes": _source_hashes(),
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote {args.json}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
