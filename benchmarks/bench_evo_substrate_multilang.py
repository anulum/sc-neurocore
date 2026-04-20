# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — evo_substrate multi-language kernel benchmark

"""Cross-language benchmark of the evo_substrate compute hot paths.

Times four kernels — `genomic_distance`, `crossover_uniform`,
`point_mutation`, `population_diversity` — across five backends:

    Rust    crates/evo_substrate_core                      (PyO3)
    Julia   accel/julia/evo_substrate/evo_substrate_bench.jl (subprocess)
    Go      accel/go/evo_substrate/                        (pre-built binary)
    Mojo    accel/mojo/kernels/evo_substrate_bench.mojo    (pixi subprocess)
    Python  NumPy reference (via the Python evo_substrate module)

The Python `src/sc_neurocore/evo_substrate/evo_substrate.py` module
remains the orchestration authority (`ReplicationEngine`, lineage,
hall-of-fame, island model, safety guards, etc.). The four non-Python
references cover only the compute kernels, not the 40+ orchestration
classes.

Emits stdout markdown + `benchmarks/results/bench_evo_substrate_multilang.json`.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))


ITERS = 100_000
DIM = 19


# ─── Python reference (NumPy) ──────────────────────────────────────────

_EPSILON = 1e-10


def py_genomic_distance(a: np.ndarray, b: np.ndarray) -> float:
    diffs = np.abs(a - b)
    norms = np.abs(a) + np.abs(b) + _EPSILON
    return float(np.mean(diffs / norms))


def py_crossover_uniform(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask.astype(bool), a, b)


def py_point_mutation(gene: np.ndarray, mask: np.ndarray, noise: np.ndarray) -> np.ndarray:
    out = gene.copy()
    m = mask.astype(bool)
    out[m] += noise[m] * (np.abs(out[m]) + 1e-8)
    return out


def ns_per_call(fn, iters: int) -> float:
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) * 1e9 / iters


def _resolve(candidates: list[str]) -> str | None:
    for c in candidates:
        if os.path.exists(c):
            return c
        # fall back to PATH lookup
        which = subprocess.run(["which", c], capture_output=True)
        if which.returncode == 0:
            return which.stdout.decode().strip()
    return None


# ─── Rust via PyO3 ─────────────────────────────────────────────────────


def bench_rust() -> dict[str, float] | None:
    try:
        from sc_neurocore.evo_substrate import evo_substrate_core as ec
    except ImportError:
        return None

    a = (np.arange(DIM, dtype=np.float64) + 1) * 0.1
    b = (np.arange(DIM, dtype=np.float64) + 1) * 0.2
    mask = (np.arange(DIM, dtype=np.uint8) % 2).astype(np.uint8)
    noise = np.full(DIM, 0.01, dtype=np.float64)

    # Warm-up
    ec.py_genomic_distance(a, b)
    ec.py_crossover_uniform(a, b, mask)
    ec.py_point_mutation(a, mask, noise)

    return {
        "genomic_distance_ns_per_call": ns_per_call(lambda: ec.py_genomic_distance(a, b), ITERS),
        "crossover_uniform_ns_per_call": ns_per_call(
            lambda: ec.py_crossover_uniform(a, b, mask), ITERS
        ),
        "point_mutation_ns_per_call": ns_per_call(
            lambda: ec.py_point_mutation(a, mask, noise), ITERS
        ),
    }


# ─── Julia via subprocess ──────────────────────────────────────────────


def bench_julia() -> dict[str, float] | None:
    julia = _resolve([os.path.expanduser("~/.juliaup/bin/julia"), "julia"])
    if julia is None:
        return None
    script = os.path.join(
        REPO_ROOT,
        "src",
        "sc_neurocore",
        "accel",
        "julia",
        "evo_substrate",
        "evo_substrate_bench.jl",
    )
    proc = subprocess.run([julia, script], capture_output=True, timeout=180)
    if proc.returncode != 0:
        return None

    results: dict[str, float] = {}
    for line in proc.stdout.decode().splitlines():
        m = re.match(r"(\S+_ns_per_call)\s+([\d.]+)", line.strip())
        if m:
            results[m.group(1)] = float(m.group(2))
    return results or None


# ─── Go via pre-built binary ───────────────────────────────────────────


def bench_go() -> dict[str, float] | None:
    go_dir = os.path.join(REPO_ROOT, "src", "sc_neurocore", "accel", "go", "evo_substrate")
    binary = os.path.join(go_dir, "evo_substrate_bench")
    if not os.path.exists(binary):
        # Try to build it on the fly
        go = _resolve(["go"])
        if go is None:
            return None
        subprocess.run(
            [go, "build", "-o", binary, "."],
            cwd=go_dir,
            capture_output=True,
            timeout=120,
        )
    if not os.path.exists(binary):
        return None
    proc = subprocess.run([binary], capture_output=True, timeout=60)
    if proc.returncode != 0:
        return None

    # The Go binary emits JSON on stderr.
    try:
        return json.loads(proc.stderr.decode().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return None


# ─── Mojo via pixi subprocess ──────────────────────────────────────────


def bench_mojo() -> dict[str, float] | None:
    mojo_dir = os.path.join(REPO_ROOT, "src", "sc_neurocore", "accel", "mojo")
    pixi = _resolve([os.path.expanduser("~/.pixi/bin/pixi"), "pixi"])
    if pixi is None or not os.path.exists(os.path.join(mojo_dir, "pixi.toml")):
        return None
    proc = subprocess.run(
        [pixi, "run", "mojo", "run", "kernels/evo_substrate_bench.mojo"],
        cwd=mojo_dir,
        capture_output=True,
        timeout=300,
    )
    if proc.returncode != 0:
        return None

    results: dict[str, float] = {}
    for line in proc.stdout.decode().splitlines():
        m = re.match(r"(\S+_ns_per_call)\s+([\d.]+)", line.strip())
        if m:
            results[m.group(1)] = float(m.group(2))
    return results or None


# ─── Python baseline ──────────────────────────────────────────────────


def bench_python() -> dict[str, float]:
    a = (np.arange(DIM, dtype=np.float64) + 1) * 0.1
    b = (np.arange(DIM, dtype=np.float64) + 1) * 0.2
    mask = (np.arange(DIM, dtype=np.uint8) % 2).astype(np.uint8)
    noise = np.full(DIM, 0.01, dtype=np.float64)
    return {
        "genomic_distance_ns_per_call": ns_per_call(lambda: py_genomic_distance(a, b), ITERS),
        "crossover_uniform_ns_per_call": ns_per_call(
            lambda: py_crossover_uniform(a, b, mask), ITERS
        ),
        "point_mutation_ns_per_call": ns_per_call(lambda: py_point_mutation(a, mask, noise), ITERS),
    }


# ─── Main ─────────────────────────────────────────────────────────────


def main() -> int:
    backends: dict[str, dict[str, float] | None] = {
        "rust": bench_rust(),
        "julia": bench_julia(),
        "go": bench_go(),
        "mojo": bench_mojo(),
        "python": bench_python(),
    }

    kernels = (
        "genomic_distance_ns_per_call",
        "crossover_uniform_ns_per_call",
        "point_mutation_ns_per_call",
    )

    print(f"\n{'Kernel':<32} {'Rust':>10} {'Julia':>10} {'Go':>10} {'Mojo':>10} {'Python':>10}")
    print("-" * 86)
    for k in kernels:
        row = [k.replace("_ns_per_call", "")]
        for backend in ("rust", "julia", "go", "mojo", "python"):
            data = backends[backend]
            if data is None or k not in data:
                row.append("—")
            else:
                row.append(f"{data[k]:.1f}")
        print(f"{row[0]:<32} {row[1]:>10} {row[2]:>10} {row[3]:>10} {row[4]:>10} {row[5]:>10}")

    print(f"\n(units: ns/call, dim={DIM}, iters={ITERS})")

    summary = {
        "dim": DIM,
        "iters": ITERS,
        "backends": {k: v for k, v in backends.items() if v is not None},
        "unavailable": [k for k, v in backends.items() if v is None],
    }
    out_dir = os.path.join(SCRIPT_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bench_evo_substrate_multilang.json")
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Results written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
