#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Ermentrout-Kopell theta map multi-language benchmark

"""Multi-language benchmark for ``ErmentroutKopellMapNeuron.simulate``.

Times the N-step theta-map recurrence across the polyglot backend chain
(python / rust / julia / go / mojo), records the parity gap against the NumPy
reference, and writes a JSON artefact.

Usage::

    python benchmarks/bench_ermentrout_kopell_map.py
    python benchmarks/bench_ermentrout_kopell_map.py --json benchmarks/results/bench_ermentrout_kopell_map.json

Measurement note: functional / local-regression benchmark on a loaded
workstation, explicitly **non-isolated** per
`BROADCAST_2026-06-04_benchmark_core_isolation`; do not promote the speed
numbers without an isolated-core rerun.
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

from sc_neurocore.neurons.models import ermentrout_kopell_map_neuron as ek
from sc_neurocore.neurons.models.ermentrout_kopell_map_neuron import ErmentroutKopellMapNeuron

N_STEPS = 2_000_000
CURRENT = 0.1
N_REPEATS = 5
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASH_PATHS = (
    "benchmarks/bench_ermentrout_kopell_map.py",
    "src/sc_neurocore/neurons/models/ermentrout_kopell_map_neuron.py",
    "engine/src/neurons/ermentrout_kopell_map.rs",
    "engine/src/bindings/ermentrout_kopell_map.rs",
    "src/sc_neurocore/accel/rust/safety/ermentrout_kopell_map_neuron.rs",
    "src/sc_neurocore/accel/go/neurons/ermentrout_kopell_map/ermentrout_kopell_map.go",
    "src/sc_neurocore/accel/julia/neurons/ermentrout_kopell_map_neuron.jl",
    "src/sc_neurocore/accel/mojo/neurons/ermentrout_kopell_map_neuron.mojo",
    "src/sc_neurocore/neurons/model_schemas/ermentrout_kopell_map_neuron.toml",
    "src/sc_neurocore/neurons/model_schemas/ermentrout_kopell_map_neuron.json",
    "src/sc_neurocore/neurons/reference_receipts/ermentrout_kopell_1986.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
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
    return (ek._HAS_RUST, "" if ek._HAS_RUST else "engine wheel lacks the symbol")


def _probe_julia() -> tuple[bool, str]:
    ok = ek._ensure_julia_loaded()
    return (ok, "" if ok else "juliacall/.jl unavailable")


def _probe_go() -> tuple[bool, str]:
    ok = ek._ensure_go_loaded()
    return (ok, "" if ok else "accel/go/neurons/ermentrout_kopell_map/libermentrout.so not built")


def _probe_mojo() -> tuple[bool, str]:
    if not _os.path.isfile(_os.path.expanduser("~/.pixi/bin/mojo")):
        return False, "mojo binary not at ~/.pixi/bin/mojo"
    ok = ek._ensure_mojo_loaded()
    return (ok, "" if ok else "accel/mojo/neurons/libermentrout.so not built")


def _run(backend: str) -> tuple[float, float, npt.NDArray[np.float64], int]:
    ErmentroutKopellMapNeuron().simulate(N_STEPS, CURRENT, backend=backend)  # warm-up (Julia JIT)
    times_ms: list[float] = []
    trace: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
    spikes = 0
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        trace, spikes = ErmentroutKopellMapNeuron().simulate(N_STEPS, CURRENT, backend=backend)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    times_ms.sort()
    return times_ms[len(times_ms) // 2], times_ms[0], trace, int(spikes)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Ermentrout-Kopell theta map benchmark.")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument(
        "--allow-unavailable-backends",
        action="store_true",
        help="Record unavailable optional backends instead of failing the parity benchmark.",
    )
    args = parser.parse_args(argv)

    print("# Ermentrout-Kopell theta map N-step benchmark")
    print(f"# Workload: {N_STEPS:,} steps, default params, current={CURRENT}")
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

    print(f"{'backend':<8}  {'available':<10}  reason")
    print(f"{'-' * 8}  {'-' * 10}  {'-' * 50}")
    for name, (avail, reason) in backends.items():
        print(f"{name:<8}  {'yes' if avail else 'no':<10}  {reason}")
    print()

    missing = [name for name, (avail, _reason) in backends.items() if not avail]
    if missing and not args.allow_unavailable_backends:
        print(
            "Missing required Ermentrout-Kopell backend(s): "
            + ", ".join(missing)
            + ". Build/install them or rerun with --allow-unavailable-backends for diagnostics only."
        )
        return 2

    reference: npt.NDArray[np.float64] | None = None
    python_median: float | None = None
    rows: list[dict[str, object]] = []

    print(f"{'backend':<8}  {'median ms':>12}  {'min ms':>12}  {'parity Δ':>12}  {'speedup':>9}")
    print(f"{'-' * 8}  {'-' * 12}  {'-' * 12}  {'-' * 12}  {'-' * 9}")
    for name, (avail, reason) in backends.items():
        if not avail:
            print(f"{name:<8}  {'(skip)':>12}  {'(skip)':>12}  {'-':>12}  {'-':>9}")
            rows.append({"backend": name, "skipped": True, "unavailable_reason": reason})
            continue
        median_ms, min_ms, trace, spikes = _run(name)
        if name == "python":
            reference = trace
            python_median = median_ms
            parity = 0.0
        else:
            if reference is None:
                raise RuntimeError("Python reference must run before accelerated backends")
            parity = float(np.max(np.abs(trace - reference)))
        speedup = (python_median / median_ms) if python_median and median_ms > 0 else float("nan")
        print(f"{name:<8}  {median_ms:>12.2f}  {min_ms:>12.2f}  {parity:>12.2e}  {speedup:>8.2f}x")
        rows.append(
            {
                "backend": name,
                "median_ms": median_ms,
                "min_ms": min_ms,
                "parity_max_abs_diff": parity,
                "speedup_vs_python": speedup,
                "spikes": spikes,
                "theta_final": float(trace[-1]) if trace.size else 0.0,
                "trace_sha256": hashlib.sha256(
                    np.asarray(trace, dtype="<f8").tobytes(order="C")
                ).hexdigest(),
            }
        )

    print()
    print("# Note: the theta neuron is non-chaotic, so cos differences do not amplify.")
    print("# Rust shares the reference libm and is bit-exact (parity 0); julia/go/mojo")
    print("# use their own cos and sit within a small non-amplifying ULP band, with")
    print("# identical spike counts. auto -> Rust (fastest bit-exact backend).")

    report = {
        "benchmark": "ermentrout_kopell_map_simulate",
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
        "source_hashes": _source_hashes(),
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
