#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Multi-language KL refinement benchmark (chiplet partitioner)

"""Reproducible multi-language benchmark for HierarchicalPartitioner._refine.

Measures wall-clock for the KL refine kernel — the post-#65 hot path
of `chiplet.HierarchicalPartitioner.partition`. Five backends:

1. **Python** (`_refine` in pure Python) — always available baseline.
2. **Rust**   (`engine/src/partition.rs` via PyO3 + ctypes-flat ABI).
3. **Julia**  (`accel/julia/chiplet/kl_refine.jl` via juliacall).
4. **Go**     (`accel/go/partition/libpartition.so` via cgo + ctypes).
5. **Mojo**   (`accel/mojo/partition/libpartition.so` via `mojo build
   --emit shared-lib` + ctypes).

Each backend probe returns `(available, reason)` so missing backends
are reported MISSING (not silently skipped — see
`feedback_multilang_workflow_canonical`).

Parity contract: every backend must produce IDENTICAL vertex →
partition mapping (membership), since the KL refine is fully
deterministic given the input. Wall-clock is reported per backend;
the `parity` column shows `ok` or `FAIL` per workload.

Workloads sweep V ∈ {100, 200, 500, 1000} at degree=8, P=4,
kl_iterations=3.

Usage:
    python benchmarks/bench_kl_refine.py
    python benchmarks/bench_kl_refine.py --json benchmarks/results/bench_kl_refine.json
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from sc_neurocore.chiplet import (
    CorrelationAwareGraph,
    CorrelationEdge,
    HierarchicalPartitioner,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
N_REPEATS = 5
N_PARTS = 4
KL_ITERATIONS = 3
WORKLOAD_VS = (100, 200, 500, 1000)


# ─────────────────────────── Workload builder ─────────────────────────

def _build_graph(n: int, deg: int = 8, seed: int = 42) -> CorrelationAwareGraph:
    rng = np.random.default_rng(seed)
    edges: list[CorrelationEdge] = []
    seen: set[tuple[int, int]] = set()
    for v in range(n):
        for u in rng.choice(n, size=min(deg, n - 1), replace=False):
            u = int(u)
            if u == v or (min(u, v), max(u, v)) in seen:
                continue
            seen.add((min(u, v), max(u, v)))
            edges.append(CorrelationEdge(u=u, v=v, conn_weight=1.0, scc_weight=0.1))
    return CorrelationAwareGraph(num_vertices=n, edges=edges)


def _initial_partitions(n_v: int, n_parts: int) -> list[list[int]]:
    return [[v for v in range(n_v) if v % n_parts == i] for i in range(n_parts)]


# ─────────────────────────── Backend probes ───────────────────────────

def probe_rust() -> dict:
    if importlib.util.find_spec("sc_neurocore_engine") is None:
        return {"available": False, "reason": "sc_neurocore_engine not installed"}
    mod = importlib.import_module("sc_neurocore_engine")
    fn = getattr(mod, "py_kl_refine", None)
    if fn is None:
        return {"available": False, "reason": "py_kl_refine missing from engine wheel"}
    return {"available": True, "kernel": fn}


def probe_julia() -> dict:
    if importlib.util.find_spec("juliacall") is None:
        return {"available": False, "reason": "juliacall not installed"}
    jl_path = REPO_ROOT / "src/sc_neurocore/accel/julia/chiplet/kl_refine.jl"
    if not jl_path.is_file():
        return {"available": False, "reason": f"{jl_path.name} not yet implemented"}
    try:
        from juliacall import Main as jl
        jl.include(str(jl_path))
        return {"available": True, "kernel": jl.KLRefineAccel.kl_refine}
    except Exception as exc:
        return {"available": False, "reason": f"julia init failed: {exc}"}


def probe_go() -> dict:
    import ctypes
    so_path = REPO_ROOT / "src/sc_neurocore/accel/go/partition/libpartition.so"
    if not so_path.is_file():
        return {"available": False, "reason": f"{so_path.name} not yet built"}
    try:
        lib = ctypes.CDLL(str(so_path))
    except OSError as exc:
        return {"available": False, "reason": f"ctypes CDLL failed: {exc}"}
    if not hasattr(lib, "kl_refine_c"):
        return {"available": False, "reason": "kl_refine_c missing from libpartition.so"}
    fn = lib.kl_refine_c
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_int64),    # adj_offsets
        ctypes.POINTER(ctypes.c_int32),    # adj_neighbours
        ctypes.POINTER(ctypes.c_double),   # adj_scc_abs
        ctypes.POINTER(ctypes.c_double),   # vertex_weights
        ctypes.POINTER(ctypes.c_int32),    # part_map (mut)
        ctypes.c_int64,                    # v_total
        ctypes.c_int64,                    # e_total
        ctypes.c_int32,                    # n_parts
        ctypes.c_int32,                    # kl_iterations
        ctypes.c_double,                   # correlation_penalty
    ]
    fn.restype = ctypes.c_uint64
    return {"available": True, "lib": lib}


def probe_mojo() -> dict:
    import ctypes
    mojo_bin = Path.home() / ".pixi/bin/mojo"
    if not mojo_bin.is_file():
        return {"available": False, "reason": "mojo not at ~/.pixi/bin/mojo"}
    so_path = REPO_ROOT / "src/sc_neurocore/accel/mojo/partition/libpartition.so"
    if not so_path.is_file():
        return {"available": False, "reason": f"{so_path.name} not yet built"}
    try:
        lib = ctypes.CDLL(str(so_path))
    except OSError as exc:
        return {"available": False, "reason": f"ctypes CDLL failed: {exc}"}
    if not hasattr(lib, "kl_refine_c"):
        return {"available": False, "reason": "kl_refine_c missing from Mojo .so"}
    fn = lib.kl_refine_c
    fn.argtypes = [
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
        ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_double,
    ]
    fn.restype = ctypes.c_uint64
    return {"available": True, "lib": lib}


# ─────────────────────────── Per-backend runners ─────────────────────

def _encode(graph: CorrelationAwareGraph, partitions: list[list[int]]):
    hp = HierarchicalPartitioner(num_partitions=N_PARTS, kl_iterations=KL_ITERATIONS)
    return hp._encode_csr(partitions, graph.adjacency(), graph)


def _run_python(graph, init):
    hp = HierarchicalPartitioner(num_partitions=N_PARTS, kl_iterations=KL_ITERATIONS,
                                  refine_backend="python")
    adj = graph.adjacency()
    hp._refine(copy.deepcopy(init), adj, graph)  # warm
    times: list[float] = []
    pm = None
    for _ in range(N_REPEATS):
        parts = copy.deepcopy(init)
        t0 = time.perf_counter()
        hp._refine(parts, adj, graph)
        times.append((time.perf_counter() - t0) * 1000.0)
        pm = np.full(graph.num_vertices, -1, dtype=np.int32)
        for i, p in enumerate(parts):
            for v in p:
                pm[v] = i
    times.sort()
    return times[len(times) // 2], pm


def _run_rust(graph, init, kernel):
    offsets, neighbours, scc_abs, vw, pm0 = _encode(graph, init)
    kernel(offsets, neighbours, scc_abs, vw, pm0, N_PARTS, KL_ITERATIONS, 2.0)  # warm
    times: list[float] = []
    pm = None
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        pm, _moves = kernel(offsets, neighbours, scc_abs, vw, pm0, N_PARTS,
                             KL_ITERATIONS, 2.0)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2], pm


def _run_julia(graph, init, kernel):
    offsets, neighbours, scc_abs, vw, pm0 = _encode(graph, init)
    pm0_jl = pm0.copy()
    kernel(offsets, neighbours, scc_abs, vw, pm0_jl, N_PARTS,
           KL_ITERATIONS, 2.0)  # warm
    times: list[float] = []
    pm = None
    for _ in range(N_REPEATS):
        pm0_jl = pm0.copy()
        t0 = time.perf_counter()
        pm = kernel(offsets, neighbours, scc_abs, vw, pm0_jl, N_PARTS,
                     KL_ITERATIONS, 2.0)
        times.append((time.perf_counter() - t0) * 1000.0)
        # juliacall gives back a Julia Vector{Int32} — convert
        pm = np.asarray(pm, dtype=np.int32)
    times.sort()
    return times[len(times) // 2], pm


def _run_go(graph, init, lib):
    import ctypes
    offsets, neighbours, scc_abs, vw, pm0 = _encode(graph, init)
    V, E = vw.size, scc_abs.size
    fn = lib.kl_refine_c
    pm = pm0.copy()
    fn(offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
       neighbours.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
       scc_abs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
       vw.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
       pm.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
       V, E, N_PARTS, KL_ITERATIONS, 2.0)  # warm
    times: list[float] = []
    for _ in range(N_REPEATS):
        pm = pm0.copy()
        t0 = time.perf_counter()
        fn(offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
           neighbours.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
           scc_abs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
           vw.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
           pm.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
           V, E, N_PARTS, KL_ITERATIONS, 2.0)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2], pm


def _run_mojo(graph, init, lib):
    import ctypes
    offsets, neighbours, scc_abs, vw, pm0 = _encode(graph, init)
    V, E = vw.size, scc_abs.size
    fn = lib.kl_refine_c
    pm = pm0.copy()
    fn(offsets.ctypes.data, neighbours.ctypes.data, scc_abs.ctypes.data,
       vw.ctypes.data, pm.ctypes.data, V, E, N_PARTS, KL_ITERATIONS, 2.0)  # warm
    times: list[float] = []
    for _ in range(N_REPEATS):
        pm = pm0.copy()
        t0 = time.perf_counter()
        fn(offsets.ctypes.data, neighbours.ctypes.data, scc_abs.ctypes.data,
           vw.ctypes.data, pm.ctypes.data, V, E, N_PARTS, KL_ITERATIONS, 2.0)
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2], pm


# ─────────────────────────── Driver ────────────────────────────────

def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    backends = {
        "python": {"available": True},
        "rust": probe_rust(),
        "julia": probe_julia(),
        "go": probe_go(),
        "mojo": probe_mojo(),
    }

    print("# HierarchicalPartitioner._refine multi-backend benchmark")
    print(f"# n_parts={N_PARTS}, kl_iterations={KL_ITERATIONS}, repeats={N_REPEATS}")
    print(f"# Python: {platform.python_version()}, NumPy: {np.__version__}")
    print(f"# platform: {platform.platform()}\n")
    print("# Backend availability")
    for name, info in backends.items():
        tag = "OK" if info.get("available") else "MISSING"
        reason = info.get("reason", "") if not info.get("available") else ""
        print(f"  {name:<8} {tag:<8} {reason}")
    print()
    cols = ["python", "rust", "julia", "go", "mojo"]
    header_cells = "  ".join(f"{c+' ms':>10}" for c in cols)
    print(f"{'V':>5}  {header_cells}  {'parity':>8}")
    print(f"{'-'*5}  {'  '.join(['-'*10]*5)}  {'-'*8}")

    rows: list[dict[str, object]] = []
    for v_n in WORKLOAD_VS:
        g = _build_graph(v_n)
        init = _initial_partitions(g.num_vertices, N_PARTS)
        row: dict[str, object] = {"V": v_n}
        pms: dict[str, np.ndarray] = {}
        # python
        py_ms, py_pm = _run_python(g, init)
        row["python_ms"] = py_ms
        pms["python"] = py_pm
        # rust
        if backends["rust"]["available"]:
            ms, pm = _run_rust(g, init, backends["rust"]["kernel"])
            row["rust_ms"] = ms
            pms["rust"] = pm
        else:
            row["rust_ms"] = None
        # julia
        if backends["julia"]["available"]:
            ms, pm = _run_julia(g, init, backends["julia"]["kernel"])
            row["julia_ms"] = ms
            pms["julia"] = pm
        else:
            row["julia_ms"] = None
        # go
        if backends["go"]["available"]:
            ms, pm = _run_go(g, init, backends["go"]["lib"])
            row["go_ms"] = ms
            pms["go"] = pm
        else:
            row["go_ms"] = None
        # mojo
        if backends["mojo"]["available"]:
            ms, pm = _run_mojo(g, init, backends["mojo"]["lib"])
            row["mojo_ms"] = ms
            pms["mojo"] = pm
        else:
            row["mojo_ms"] = None

        # Parity: every available backend must produce same membership.
        ref = pms["python"]
        ok = all(np.array_equal(ref, p) for p in pms.values())
        row["parity_ok"] = ok

        cells = []
        for c in cols:
            v = row.get(f"{c}_ms")
            cells.append(f"{v:>10.2f}" if v is not None else f"{'-':>10}")
        print(f"{v_n:>5}  {'  '.join(cells)}  {'ok' if ok else 'FAIL':>8}")
        rows.append(row)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        backends_json = {
            name: {k: v for k, v in info.items()
                   if k in ("available", "reason")}
            for name, info in backends.items()
        }
        payload = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "n_parts": N_PARTS,
            "kl_iterations": KL_ITERATIONS,
            "n_repeats": N_REPEATS,
            "backends": backends_json,
            "rows": rows,
        }
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
