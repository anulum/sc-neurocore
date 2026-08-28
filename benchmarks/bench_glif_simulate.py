#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Allen GLIF5 multi-language benchmark

"""Multi-language benchmark for ``GLIFNeuron.simulate``.

Times the N-step five-state exact-flow recurrence across the polyglot backend chain
(python / rust / julia / go / mojo), records the parity gap against the NumPy
reference, and writes a JSON artefact.

Usage::

    python benchmarks/bench_glif_simulate.py
    python benchmarks/bench_glif_simulate.py --json benchmarks/results/bench_glif_simulate.json

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

from sc_neurocore.neurons.models import glif
from sc_neurocore.neurons.models.glif import GLIFNeuron

N_STEPS = 2_000_000
CURRENT = 30.0
N_REPEATS = 5
REPOSITORY = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _probe_rust() -> tuple[bool, str]:
    ok = glif._HAS_RUST
    return (ok, "" if ok else "engine wheel lacks the symbol")


def _probe_julia() -> tuple[bool, str]:
    ok = glif._ensure_julia_loaded()
    return (ok, "" if ok else "juliacall/.jl unavailable")


def _probe_go() -> tuple[bool, str]:
    ok = glif._ensure_go_loaded()
    return (ok, "" if ok else "accel/go/neurons/glif/libglif.so not built")


def _probe_mojo() -> tuple[bool, str]:
    if not _os.path.isfile(_os.path.expanduser("~/.pixi/bin/mojo")):
        return False, "mojo binary not at ~/.pixi/bin/mojo"
    ok = glif._ensure_mojo_loaded()
    return (ok, "" if ok else "accel/mojo/neurons/libglif.so not built")


def _run(
    backend: str,
) -> tuple[float, float, npt.NDArray[np.float64], int, dict[str, float]]:
    GLIFNeuron().simulate(N_STEPS, CURRENT, backend=backend)  # warm-up (Julia JIT)
    times_ms: list[float] = []
    trace: npt.NDArray[np.float64] = np.empty(0, dtype=np.float64)
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        neuron = GLIFNeuron()
        trace, events = neuron.simulate(N_STEPS, CURRENT, backend=backend)
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    times_ms.sort()
    state = {
        "v": neuron.v,
        "theta_spike": neuron.theta_spike,
        "i_asc1": neuron.i_asc1,
        "i_asc2": neuron.i_asc2,
        "theta_voltage": neuron.theta_voltage,
        "refractory_remaining": neuron.refractory_remaining,
    }
    return times_ms[len(times_ms) // 2], times_ms[0], trace, events, state


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Allen GLIF5 multi-language benchmark.")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    print("# Teeter GLIF5 N-step exact-flow benchmark")
    print(f"# Workload: {N_STEPS:,} steps, default tonic regime, current={CURRENT}")
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
    reference_state: dict[str, float] | None = None
    reference_events: int | None = None
    python_median: float | None = None
    rows: list[dict[str, object]] = []

    print(f"{'backend':<8}  {'median ms':>12}  {'min ms':>12}  {'parity Δ':>12}  {'speedup':>9}")
    print(f"{'-' * 8}  {'-' * 12}  {'-' * 12}  {'-' * 12}  {'-' * 9}")
    for name, (avail, reason) in backends.items():
        if not avail:
            print(f"{name:<8}  {'(skip)':>12}  {'(skip)':>12}  {'-':>12}  {'-':>9}")
            rows.append({"backend": name, "skipped": True, "unavailable_reason": reason})
            continue
        median_ms, min_ms, trace, events, state = _run(name)
        if name == "python":
            reference = trace
            reference_state = state
            reference_events = events
            python_median = median_ms
            parity = 0.0
        else:
            assert reference is not None
            assert reference_state is not None
            assert reference_events is not None
            parity = max(
                float(np.max(np.abs(trace - reference))),
                max(abs(state[key] - reference_state[key]) for key in state),
            )
        speedup = (python_median / median_ms) if python_median and median_ms > 0 else float("nan")
        print(f"{name:<8}  {median_ms:>12.2f}  {min_ms:>12.2f}  {parity:>12.2e}  {speedup:>8.2f}x")
        rows.append(
            {
                "backend": name,
                "median_ms": median_ms,
                "min_ms": min_ms,
                "parity_max_abs_diff": parity,
                "speedup_vs_python": speedup,
                "events": events,
                "final_state": state,
                "event_delta_vs_python": 0
                if reference_events is None
                else events - reference_events,
            }
        )

    print()
    print("# Note: the GLIF5 exact-flow specialization includes exponential coefficients.")
    print("# Complete voltage traces, final states, and event counts are compared to Python;")
    print("# timing remains loaded-host local regression evidence, not a hardware claim.")

    sources = {
        "python": "src/sc_neurocore/neurons/models/glif.py",
        "rust": "engine/src/neurons/biophysical/glif.rs",
        "julia": "src/sc_neurocore/accel/julia/neurons/glif.jl",
        "go": "src/sc_neurocore/accel/go/neurons/glif/glif.go",
        "mojo": "src/sc_neurocore/accel/mojo/neurons/glif.mojo",
        "receipt": "src/sc_neurocore/neurons/reference_receipts/glif5_teeter_2018.json",
    }

    report = {
        "benchmark": "glif_simulate",
        "workload": {"n_steps": N_STEPS, "current": CURRENT, "repeats": N_REPEATS},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "isolation": "non-isolated (loaded workstation)",
        },
        "results": rows,
        "source_sha256": {name: _sha256(REPOSITORY / path) for name, path in sources.items()},
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
