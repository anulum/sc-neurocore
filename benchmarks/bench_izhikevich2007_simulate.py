#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Izhikevich 2007 RK4 simulator multi-language benchmark

"""Multi-language benchmark for ``Izhikevich2007Neuron.simulate`` (RK4).

Times the N-step RK4 recurrence across the polyglot backend chain
(python / rust / julia / go / mojo), records the parity gap against the NumPy
reference, and writes a JSON artefact.

Usage::

    python benchmarks/bench_izhikevich2007_simulate.py
    python benchmarks/bench_izhikevich2007_simulate.py --json benchmarks/results/bench_izhikevich2007_simulate.json

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

from sc_neurocore.neurons.models import izhikevich2007 as izh
from sc_neurocore.neurons.models.izhikevich2007 import Izhikevich2007Neuron

N_STEPS = 2_000_000
CURRENT = 300.0
N_REPEATS = 5
REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASH_PATHS = (
    "benchmarks/bench_izhikevich2007_simulate.py",
    "src/sc_neurocore/neurons/models/izhikevich2007.py",
    "engine/src/rk4_neurons.rs",
    "engine/src/bindings/izhikevich2007.rs",
    "src/sc_neurocore/accel/rust/safety/izhikevich2007.rs",
    "src/sc_neurocore/accel/go/neurons/izhikevich2007/izhikevich2007.go",
    "src/sc_neurocore/accel/julia/neurons/izhikevich2007.jl",
    "src/sc_neurocore/accel/mojo/neurons/izhikevich2007.mojo",
    "src/sc_neurocore/neurons/model_schemas/izhikevich2007.toml",
    "src/sc_neurocore/neurons/model_schemas/izhikevich2007.json",
    "src/sc_neurocore/neurons/reference_receipts/izhikevich_2007.json",
    "src/sc_neurocore/neurons/reference_receipts/izhikevich_2007_rk4.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_hashes() -> dict[str, object]:
    flat: dict[str, object] = {path: _sha256(REPO_ROOT / path) for path in SOURCE_HASH_PATHS}
    nested: dict[str, object] = {}
    for path, digest in flat.items():
        stem, suffix = path.rsplit(".", 1)
        by_suffix = nested.setdefault(stem, {})
        assert isinstance(by_suffix, dict)
        by_suffix[suffix] = digest
    return {**flat, **nested}


def _probe_rust() -> tuple[bool, str]:
    return (izh._HAS_RUST, "" if izh._HAS_RUST else "engine wheel lacks the symbol")


def _probe_julia() -> tuple[bool, str]:
    ok = izh._ensure_julia_loaded()
    return (ok, "" if ok else "juliacall/.jl unavailable")


def _probe_go() -> tuple[bool, str]:
    ok = izh._ensure_go_loaded()
    return (ok, "" if ok else "accel/go/neurons/izhikevich2007/libizh2007.so not built")


def _probe_mojo() -> tuple[bool, str]:
    if not _os.path.isfile(_os.path.expanduser("~/.pixi/bin/mojo")):
        return False, "mojo binary not at ~/.pixi/bin/mojo"
    ok = izh._ensure_mojo_loaded()
    return (ok, "" if ok else "accel/mojo/neurons/libizh2007.so not built")


def _run(backend: str) -> tuple[float, float, npt.NDArray[np.float64], int, float, float]:
    Izhikevich2007Neuron().simulate(N_STEPS, CURRENT, backend=backend)  # warm-up (Julia JIT)
    times_ms: list[float] = []
    trace: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
    spikes = 0
    final_v = -60.0
    final_u = 0.0
    for _ in range(N_REPEATS):
        neuron = Izhikevich2007Neuron()
        t0 = time.perf_counter()
        trace, spikes = neuron.simulate(N_STEPS, CURRENT, backend=backend)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
        final_v, final_u = neuron.v, neuron.u
    times_ms.sort()
    return times_ms[len(times_ms) // 2], times_ms[0], trace, spikes, final_v, final_u


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Izhikevich 2007 RK4 multi-language benchmark.")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    print("# Izhikevich 2007 RK4 N-step benchmark")
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
        median_ms, min_ms, trace, spikes, final_v, final_u = _run(name)
        if name == "python":
            reference = trace
            python_median = median_ms
            parity = 0.0
        else:
            assert reference is not None
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
                "spikes": int(spikes),
                "v_final": float(final_v),
                "u_final": float(final_u),
                "trace_sha256": hashlib.sha256(
                    np.asarray(trace, dtype="<f8").tobytes(order="C")
                ).hexdigest(),
            }
        )

    print()
    print("# Note: the NeuroML RHS k(v-vr)(v-vt)/C is exact arithmetic (no")
    print("# transcendentals), so rust/julia/go reproduce the trace bit-for-bit")
    print("# (parity 0). Mojo fuses some RK4 multiply-adds into FMAs; the hard")
    print("# vpeak reset re-anchors the trajectory, so the gap stays a tiny")
    print("# non-amplifying ULP band with identical spike counts. auto -> Rust.")

    report = {
        "benchmark": "izhikevich2007_simulate_rk4",
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
