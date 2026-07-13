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
import ctypes
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt
import sc_neurocore

from sc_neurocore.chiplet import (
    CorrelationAwareGraph,
    CorrelationEdge,
    HierarchicalPartitioner,
)


SC_NEUROCORE_FILE = sc_neurocore.__file__
if SC_NEUROCORE_FILE is None:  # pragma: no cover - import packages always expose it.
    raise RuntimeError("sc_neurocore package has no filesystem origin")
SOURCE_REPO_ROOT = Path(SC_NEUROCORE_FILE).resolve().parents[2]
BENCHMARK_SCHEMA_VERSION = "sc-neurocore.kl-refine-benchmark.v1"
N_REPEATS = 5
N_PARTS = 4
KL_ITERATIONS = 3
WORKLOAD_VS = (100, 200, 500, 1000)
MAINTAINED_KERNEL_PATHS = (
    "engine/src/partition.rs",
    "src/sc_neurocore/accel/julia/chiplet/kl_refine.jl",
    "src/sc_neurocore/accel/go/partition/partition.go",
    "src/sc_neurocore/accel/mojo/partition/partition.mojo",
)

Array: TypeAlias = npt.NDArray[Any]
PartitionMap: TypeAlias = npt.NDArray[np.int32]
EncodedBuffers: TypeAlias = tuple[Array, Array, Array, Array, Array, Array, Array]


class SourceManifest(TypedDict):
    """Source and binary hashes bound to one benchmark result."""

    combined_source_sha256: str
    files: dict[str, str]
    backend_binary_sha256: dict[str, str | None]
    runner_sha256: str


@dataclass(frozen=True)
class BackendProbe:
    """Availability and callable handle for one benchmark backend."""

    available: bool
    reason: str | None = None
    kernel: Any | None = None
    library: ctypes.CDLL | None = None


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_manifest() -> SourceManifest:
    """Bind the measurement to Python and maintained backend sources."""
    relative_paths = sorted(
        path.relative_to(SOURCE_REPO_ROOT)
        for path in (SOURCE_REPO_ROOT / "src" / "sc_neurocore" / "chiplet").glob("hierarchical*.py")
    )
    relative_paths.extend(Path(path) for path in MAINTAINED_KERNEL_PATHS)
    source_hashes = {str(path): _sha256(SOURCE_REPO_ROOT / path) for path in relative_paths}
    digest = hashlib.sha256()
    for path, file_hash in sorted(source_hashes.items()):
        digest.update(path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")

    rust_spec = importlib.util.find_spec("sc_neurocore_engine")
    rust_path = (
        Path(rust_spec.origin) if rust_spec is not None and rust_spec.origin is not None else None
    )
    binary_paths = {
        "rust": rust_path,
        "go": SOURCE_REPO_ROOT / "src/sc_neurocore/accel/go/partition/libpartition.so",
        "mojo": SOURCE_REPO_ROOT / "src/sc_neurocore/accel/mojo/partition/libpartition.so",
    }
    binary_hashes = {
        name: _sha256(path) if path is not None and path.is_file() else None
        for name, path in binary_paths.items()
    }
    return {
        "combined_source_sha256": digest.hexdigest(),
        "files": source_hashes,
        "backend_binary_sha256": binary_hashes,
        "runner_sha256": _sha256(Path(__file__).resolve()),
    }


def _gate_source_hashes(manifest: SourceManifest) -> dict[str, str]:
    """Flatten source hashes into dot-path-safe keys for the evidence gate."""
    hashes = {"runner": manifest["runner_sha256"]}
    for path, digest in manifest["files"].items():
        key = path.replace("/", "__").replace(".", "_").replace("-", "_")
        if key in hashes:
            raise RuntimeError(f"duplicate benchmark source key: {key}")
        hashes[key] = digest
    return hashes


def _environment() -> dict[str, object]:
    """Capture non-isolation diagnostics without promoting timing claims."""
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except AttributeError:
        affinity = []
    try:
        load_average = list(os.getloadavg())
    except OSError:
        load_average = []
    governor_path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    governor = (
        governor_path.read_text(encoding="utf-8").strip()
        if governor_path.is_file()
        else "unavailable"
    )
    return {
        "affinity": affinity,
        "load_average": load_average,
        "cpu_governor": governor,
        "isolated_cpu_claim": False,
        "timing_class": "loaded-host diagnostic",
    }


# ─────────────────────────── Workload builder ─────────────────────────


def _build_graph(n: int, deg: int = 8, seed: int = 42) -> CorrelationAwareGraph:
    """Build a deterministic sparse undirected correlation graph."""
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
    """Assign vertices round-robin to deterministic initial partitions."""
    return [[v for v in range(n_v) if v % n_parts == i] for i in range(n_parts)]


# ─────────────────────────── Backend probes ───────────────────────────


def probe_rust() -> BackendProbe:
    """Discover the installed Rust KL-refinement entry point."""
    if importlib.util.find_spec("sc_neurocore_engine") is None:
        return BackendProbe(False, "sc_neurocore_engine not installed")
    mod = importlib.import_module("sc_neurocore_engine")
    fn = getattr(mod, "py_kl_refine", None)
    if fn is None:
        return BackendProbe(False, "py_kl_refine missing from engine wheel")
    return BackendProbe(True, kernel=fn)


def probe_julia() -> BackendProbe:
    """Load the maintained Julia KL-refinement module when available."""
    if importlib.util.find_spec("juliacall") is None:
        return BackendProbe(False, "juliacall not installed")
    jl_path = SOURCE_REPO_ROOT / "src/sc_neurocore/accel/julia/chiplet/kl_refine.jl"
    if not jl_path.is_file():
        return BackendProbe(False, f"{jl_path.name} not yet implemented")
    try:
        juliacall: Any = importlib.import_module("juliacall")
        jl: Any = juliacall.Main
        jl.include(str(jl_path))
        return BackendProbe(True, kernel=jl.KLRefineAccel.kl_refine)
    except Exception as exc:
        return BackendProbe(False, f"julia init failed: {exc}")


def probe_go() -> BackendProbe:
    """Load and type the maintained Go shared-library ABI."""
    so_path = SOURCE_REPO_ROOT / "src/sc_neurocore/accel/go/partition/libpartition.so"
    if not so_path.is_file():
        return BackendProbe(False, f"{so_path.name} not yet built")
    try:
        lib = ctypes.CDLL(str(so_path))
    except OSError as exc:
        return BackendProbe(False, f"ctypes CDLL failed: {exc}")
    if not hasattr(lib, "kl_refine_c"):
        return BackendProbe(False, "kl_refine_c missing from libpartition.so")
    fn = lib.kl_refine_c
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_int64),  # adj_offsets
        ctypes.POINTER(ctypes.c_int32),  # adj_neighbours
        ctypes.POINTER(ctypes.c_double),  # adj_scc_abs
        ctypes.POINTER(ctypes.c_double),  # vertex_weights
        ctypes.POINTER(ctypes.c_int32),  # part_map (mut)
        ctypes.POINTER(ctypes.c_int32),  # parts_concat
        ctypes.POINTER(ctypes.c_int64),  # parts_offsets
        ctypes.c_int64,  # v_total
        ctypes.c_int64,  # e_total
        ctypes.c_int32,  # n_parts
        ctypes.c_int32,  # kl_iterations
        ctypes.c_double,  # correlation_penalty
    ]
    fn.restype = ctypes.c_uint64
    return BackendProbe(True, library=lib)


def probe_mojo() -> BackendProbe:
    """Load and type the maintained Mojo shared-library ABI."""
    mojo_bin = Path.home() / ".pixi/bin/mojo"
    if not mojo_bin.is_file():
        return BackendProbe(False, "mojo not at ~/.pixi/bin/mojo")
    so_path = SOURCE_REPO_ROOT / "src/sc_neurocore/accel/mojo/partition/libpartition.so"
    if not so_path.is_file():
        return BackendProbe(False, f"{so_path.name} not yet built")
    try:
        lib = ctypes.CDLL(str(so_path))
    except OSError as exc:
        return BackendProbe(False, f"ctypes CDLL failed: {exc}")
    if not hasattr(lib, "kl_refine_c"):
        return BackendProbe(False, "kl_refine_c missing from Mojo .so")
    fn = lib.kl_refine_c
    fn.argtypes = [
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_double,
    ]
    fn.restype = ctypes.c_uint64
    return BackendProbe(True, library=lib)


# ─────────────────────────── Per-backend runners ─────────────────────


def _encode(
    graph: CorrelationAwareGraph,
    partitions: list[list[int]],
) -> EncodedBuffers:
    """Encode graph and partition state into the shared flat-buffer ABI."""
    hp = HierarchicalPartitioner(num_partitions=N_PARTS, kl_iterations=KL_ITERATIONS)
    return hp._encode_csr(partitions, graph.adjacency(), graph)


def _run_python(
    graph: CorrelationAwareGraph,
    initial: list[list[int]],
) -> tuple[float, PartitionMap]:
    """Measure the Python reference path and return its final part map."""
    hp = HierarchicalPartitioner(
        num_partitions=N_PARTS, kl_iterations=KL_ITERATIONS, refine_backend="python"
    )
    adj = graph.adjacency()
    hp._refine(copy.deepcopy(initial), adj, graph)  # warm
    times: list[float] = []
    partition_map = np.full(graph.num_vertices, -1, dtype=np.int32)
    for _ in range(N_REPEATS):
        parts = copy.deepcopy(initial)
        t0 = time.perf_counter()
        hp._refine(parts, adj, graph)
        times.append((time.perf_counter() - t0) * 1000.0)
        partition_map.fill(-1)
        for i, p in enumerate(parts):
            for v in p:
                partition_map[v] = i
    times.sort()
    return times[len(times) // 2], partition_map


def _run_rust(
    graph: CorrelationAwareGraph,
    initial: list[list[int]],
    kernel: Any,
) -> tuple[float, PartitionMap]:
    """Measure the installed Rust entry point and return its part map."""
    offsets, neighbours, scc_abs, vw, pm0, pc, po = _encode(graph, initial)
    kernel(offsets, neighbours, scc_abs, vw, pm0, pc, po, N_PARTS, KL_ITERATIONS, 2.0)  # warm
    times: list[float] = []
    partition_map = np.asarray(pm0, dtype=np.int32)
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        result, _moves = kernel(
            offsets, neighbours, scc_abs, vw, pm0, pc, po, N_PARTS, KL_ITERATIONS, 2.0
        )
        times.append((time.perf_counter() - t0) * 1000.0)
        partition_map = np.asarray(result, dtype=np.int32)
    times.sort()
    return times[len(times) // 2], partition_map


def _run_julia(
    graph: CorrelationAwareGraph,
    initial: list[list[int]],
    kernel: Any,
) -> tuple[float, PartitionMap]:
    """Measure the maintained Julia module and return its part map."""
    offsets, neighbours, scc_abs, vw, pm0, pc, po = _encode(graph, initial)
    kernel(
        offsets, neighbours, scc_abs, vw, pm0.copy(), pc, po, N_PARTS, KL_ITERATIONS, 2.0
    )  # warm
    times: list[float] = []
    partition_map = np.asarray(pm0, dtype=np.int32)
    for _ in range(N_REPEATS):
        pm0_jl = pm0.copy()
        t0 = time.perf_counter()
        result = kernel(
            offsets,
            neighbours,
            scc_abs,
            vw,
            pm0_jl,
            pc,
            po,
            N_PARTS,
            KL_ITERATIONS,
            2.0,
        )
        times.append((time.perf_counter() - t0) * 1000.0)
        partition_map = np.asarray(result, dtype=np.int32)
    times.sort()
    return times[len(times) // 2], partition_map


def _run_go(
    graph: CorrelationAwareGraph,
    initial: list[list[int]],
    library: ctypes.CDLL,
) -> tuple[float, PartitionMap]:
    """Measure the maintained Go C ABI and return its part map."""
    offsets, neighbours, scc_abs, vw, pm0, pc, po = _encode(graph, initial)
    vertex_count, edge_count = vw.size, scc_abs.size
    fn = library.kl_refine_c
    pm = pm0.copy()
    fn(
        offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        neighbours.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        scc_abs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        vw.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        pm.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        pc.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        po.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        vertex_count,
        edge_count,
        N_PARTS,
        KL_ITERATIONS,
        2.0,
    )  # warm
    times: list[float] = []
    for _ in range(N_REPEATS):
        pm = pm0.copy()
        t0 = time.perf_counter()
        fn(
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            neighbours.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            scc_abs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            vw.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            pm.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            pc.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            po.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            vertex_count,
            edge_count,
            N_PARTS,
            KL_ITERATIONS,
            2.0,
        )
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2], pm


def _run_mojo(
    graph: CorrelationAwareGraph,
    initial: list[list[int]],
    library: ctypes.CDLL,
) -> tuple[float, PartitionMap]:
    """Measure the maintained Mojo C ABI and return its part map."""
    offsets, neighbours, scc_abs, vw, pm0, pc, po = _encode(graph, initial)
    vertex_count, edge_count = vw.size, scc_abs.size
    fn = library.kl_refine_c
    pm = pm0.copy()
    fn(
        offsets.ctypes.data,
        neighbours.ctypes.data,
        scc_abs.ctypes.data,
        vw.ctypes.data,
        pm.ctypes.data,
        pc.ctypes.data,
        po.ctypes.data,
        vertex_count,
        edge_count,
        N_PARTS,
        KL_ITERATIONS,
        2.0,
    )  # warm
    times: list[float] = []
    for _ in range(N_REPEATS):
        pm = pm0.copy()
        t0 = time.perf_counter()
        fn(
            offsets.ctypes.data,
            neighbours.ctypes.data,
            scc_abs.ctypes.data,
            vw.ctypes.data,
            pm.ctypes.data,
            pc.ctypes.data,
            po.ctypes.data,
            vertex_count,
            edge_count,
            N_PARTS,
            KL_ITERATIONS,
            2.0,
        )
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2], pm


# ─────────────────────────── Driver ────────────────────────────────


def main(argv: list[str]) -> int:
    """Run every backend workload, emit evidence, and fail on parity drift."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--label", default="candidate")
    args = parser.parse_args(argv)

    backends = {
        "python": BackendProbe(True),
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
        tag = "OK" if info.available else "MISSING"
        reason = "" if info.available else info.reason or ""
        print(f"  {name:<8} {tag:<8} {reason}")
    print()
    cols = ["python", "rust", "julia", "go", "mojo"]
    header_cells = "  ".join(f"{c + ' ms':>10}" for c in cols)
    print(f"{'V':>5}  {header_cells}  {'parity':>8}")
    print(f"{'-' * 5}  {'  '.join(['-' * 10] * 5)}  {'-' * 8}")

    rows: list[dict[str, object]] = []
    timings_ms: dict[str, dict[str, float | None]] = {}
    canonical_partitions: list[dict[str, object]] = []
    all_parity = True
    for v_n in WORKLOAD_VS:
        g = _build_graph(v_n)
        init = _initial_partitions(g.num_vertices, N_PARTS)
        row: dict[str, object] = {"V": v_n}
        pms: dict[str, PartitionMap] = {}
        # python
        py_ms, py_pm = _run_python(g, init)
        row["python_ms"] = py_ms
        pms["python"] = py_pm
        # rust
        rust_probe = backends["rust"]
        if rust_probe.available:
            assert rust_probe.kernel is not None
            ms, pm = _run_rust(g, init, rust_probe.kernel)
            row["rust_ms"] = ms
            pms["rust"] = pm
        else:
            row["rust_ms"] = None
        # julia
        julia_probe = backends["julia"]
        if julia_probe.available:
            assert julia_probe.kernel is not None
            ms, pm = _run_julia(g, init, julia_probe.kernel)
            row["julia_ms"] = ms
            pms["julia"] = pm
        else:
            row["julia_ms"] = None
        # go
        go_probe = backends["go"]
        if go_probe.available:
            assert go_probe.library is not None
            ms, pm = _run_go(g, init, go_probe.library)
            row["go_ms"] = ms
            pms["go"] = pm
        else:
            row["go_ms"] = None
        # mojo
        mojo_probe = backends["mojo"]
        if mojo_probe.available:
            assert mojo_probe.library is not None
            ms, pm = _run_mojo(g, init, mojo_probe.library)
            row["mojo_ms"] = ms
            pms["mojo"] = pm
        else:
            row["mojo_ms"] = None

        # Parity: every available backend must produce same membership.
        ref = pms["python"]
        ok = all(np.array_equal(ref, p) for p in pms.values())
        row["parity_ok"] = ok
        all_parity = all_parity and ok
        canonical_partitions.append({"V": v_n, "python_part_map": ref.astype(int).tolist()})

        cells: list[str] = []
        for c in cols:
            v = row.get(f"{c}_ms")
            cells.append(f"{v:>10.2f}" if isinstance(v, int | float) else f"{'-':>10}")
        timings_ms[f"v{v_n}"] = {
            backend: float(value) if isinstance(value, int | float) else None
            for backend in cols
            if (value := row.get(f"{backend}_ms")) is not None
        }
        print(f"{v_n:>5}  {'  '.join(cells)}  {'ok' if ok else 'FAIL':>8}")
        rows.append(row)

    canonical_bytes = json.dumps(
        canonical_partitions,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    source_manifest = _source_manifest()
    payload = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "label": args.label,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "n_parts": N_PARTS,
        "kl_iterations": KL_ITERATIONS,
        "n_repeats": N_REPEATS,
        "source_manifest": source_manifest,
        "gate_source_hashes": _gate_source_hashes(source_manifest),
        "environment": _environment(),
        "canonical_partition_sha256": hashlib.sha256(canonical_bytes).hexdigest(),
        "canonical_equivalence": all_parity,
        "backends": {
            name: {
                "available": info.available,
                "reason": info.reason,
            }
            for name, info in backends.items()
        },
        "rows": rows,
        "timings_ms": timings_ms,
    }

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")

    return 0 if all_parity else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
