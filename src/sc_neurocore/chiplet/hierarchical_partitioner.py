# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Exascale Hierarchical Partitioning

"""Correlation-aware hierarchical graph partitioning for MPI scale-out.

Uses multi-level coarsening (heavy-edge matching), spectral bisection on
the coarsened graph, and Kernighan-Lin local refinement during uncoarsening.
Assigns independent LFSR seeds per partition to prevent cross-node
correlation blow-up.  Includes a load balancer that monitors inter-partition
SCC and migrates neurons to minimise boundary correlation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# ── Hierarchy Levels ─────────────────────────────────────────────────


class HierarchyLevel(Enum):
    RACK = "rack"
    NODE = "node"
    DIE = "die"
    TILE = "tile"


# ── CSR Sparse Graph ─────────────────────────────────────────────────


@dataclass
class CSRGraph:
    """Compressed Sparse Row graph for billion-neuron scale.

    O(1) adjacency access per vertex, O(E) total memory.
    """

    num_vertices: int
    indptr: np.ndarray[Any, Any]  # shape (num_vertices + 1,)
    indices: np.ndarray[Any, Any]  # shape (nnz,)
    conn_weights: np.ndarray[Any, Any]  # shape (nnz,)
    scc_weights: np.ndarray[Any, Any]  # shape (nnz,)
    vertex_weights: np.ndarray[Any, Any]  # shape (num_vertices,)

    @classmethod
    def from_edge_list(
        cls,
        num_vertices: int,
        edges: List[CorrelationEdge],
        vertex_weights: Optional[Dict[int, float]] = None,
    ) -> CSRGraph:
        """Build CSR from edge list (symmetric: adds both directions)."""
        adj: Dict[int, List[Tuple[int, float, float]]] = {i: [] for i in range(num_vertices)}
        for e in edges:
            adj[e.u].append((e.v, e.conn_weight, e.scc_weight))
            adj[e.v].append((e.u, e.conn_weight, e.scc_weight))

        indptr = np.zeros(num_vertices + 1, dtype=np.int64)
        all_indices = []
        all_conn = []
        all_scc = []
        for i in range(num_vertices):
            neighbors = sorted(adj[i], key=lambda x: x[0])
            indptr[i + 1] = indptr[i] + len(neighbors)
            for j, cw, sw in neighbors:
                all_indices.append(j)
                all_conn.append(cw)
                all_scc.append(sw)

        vw = np.ones(num_vertices, dtype=np.float64)
        if vertex_weights:
            for vid, w in vertex_weights.items():
                vw[vid] = w

        return cls(
            num_vertices=num_vertices,
            indptr=indptr,
            indices=np.array(all_indices, dtype=np.int64),
            conn_weights=np.array(all_conn, dtype=np.float64),
            scc_weights=np.array(all_scc, dtype=np.float64),
            vertex_weights=vw,
        )

    def neighbors(self, v: int) -> np.ndarray[Any, Any]:
        return self.indices[self.indptr[v] : self.indptr[v + 1]]

    def degree(self, v: int) -> int:
        return int(self.indptr[v + 1] - self.indptr[v])

    def edge_conn(self, v: int) -> np.ndarray[Any, Any]:
        return self.conn_weights[self.indptr[v] : self.indptr[v + 1]]

    def edge_scc(self, v: int) -> np.ndarray[Any, Any]:
        return self.scc_weights[self.indptr[v] : self.indptr[v + 1]]

    @property
    def num_edges(self) -> int:
        return len(self.indices) // 2  # symmetric


@dataclass
class CorrelationEdge:
    """An edge with both connection weight and SC correlation weight."""

    u: int
    v: int
    conn_weight: float = 1.0
    scc_weight: float = 0.0


# ── Lazy multi-language backend probes (KL refine) ──
# All 4 native backends share the same CSR-flat ABI (offsets,
# neighbours, scc_abs, vertex_weights, part_map, n_parts,
# kl_iterations, correlation_penalty). Rust is eager (PyO3
# wheel); Julia/Go/Mojo are lazily loaded on first request.
# Per `feedback_no_blocked_without_probing.md`, all 4 are wired
# even when the kernel is not yet built — the dispatcher returns
# a clear error if the user asks for a missing backend.
try:
    from sc_neurocore_engine import py_kl_refine as _rust_kl_refine

    _HAS_RUST_KL_REFINE = True
except (ImportError, AttributeError):
    _rust_kl_refine = None
    _HAS_RUST_KL_REFINE = False


_julia_kl_refine = None
_HAS_JULIA_KL_REFINE = False
_go_kl_refine_lib = None
_HAS_GO_KL_REFINE = False
_mojo_kl_refine_lib = None
_HAS_MOJO_KL_REFINE = False


def _ensure_julia_kl_refine_loaded() -> bool:
    """Lazy-load the Julia KL refine module on first use."""
    global _julia_kl_refine, _HAS_JULIA_KL_REFINE
    if _julia_kl_refine is not None:
        return True
    try:
        from juliacall import (
            Main as _jl,
        )
    except ImportError:
        return False
    import os as _os

    jl_path = _os.path.join(
        _os.path.dirname(_os.path.dirname(__file__)),
        "accel",
        "julia",
        "chiplet",
        "kl_refine.jl",
    )
    if not _os.path.isfile(jl_path):
        return False
    try:
        _jl.include(jl_path)
        _julia_kl_refine = _jl.KLRefineAccel.kl_refine
    except Exception:
        return False
    _HAS_JULIA_KL_REFINE = True
    return True


def _ensure_go_kl_refine_loaded() -> bool:
    """Lazy-load the Go KL refine shared library on first use."""
    global _go_kl_refine_lib, _HAS_GO_KL_REFINE
    if _go_kl_refine_lib is not None:
        return True
    import ctypes
    import os as _os

    so_path = _os.path.join(
        _os.path.dirname(_os.path.dirname(__file__)),
        "accel",
        "go",
        "partition",
        "libpartition.so",
    )
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "kl_refine_c", None)
    if fn is None:
        return False
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_int64),  # adj_offsets
        ctypes.POINTER(ctypes.c_int32),  # adj_neighbours
        ctypes.POINTER(ctypes.c_double),  # adj_scc_abs
        ctypes.POINTER(ctypes.c_double),  # vertex_weights
        ctypes.POINTER(ctypes.c_int32),  # part_map (mut)
        ctypes.POINTER(ctypes.c_int32),  # parts_concat
        ctypes.POINTER(ctypes.c_int64),  # parts_offsets
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_double,
    ]
    fn.restype = ctypes.c_uint64
    _go_kl_refine_lib = lib
    _HAS_GO_KL_REFINE = True
    return True


def _ensure_mojo_kl_refine_loaded() -> bool:
    """Lazy-load the Mojo KL refine shared library on first use."""
    global _mojo_kl_refine_lib, _HAS_MOJO_KL_REFINE
    if _mojo_kl_refine_lib is not None:
        return True
    import ctypes
    import os as _os

    so_path = _os.path.join(
        _os.path.dirname(_os.path.dirname(__file__)),
        "accel",
        "mojo",
        "partition",
        "libpartition.so",
    )
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "kl_refine_c", None)
    if fn is None:
        return False
    # Mojo @export takes raw Int addresses (no parametric pointers).
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
    _mojo_kl_refine_lib = lib
    _HAS_MOJO_KL_REFINE = True
    return True


@dataclass
class CorrelationAwareGraph:
    """Adjacency representation with per-edge SCC weights.

    Edge lookups (`edge_weight`, `edge_scc`) are O(1) via a cached
    `(min_uv, max_uv) → CorrelationEdge` dict, lazily built on first
    access. Was O(E) per call (linear scan), which made the
    partitioner O(V²·E) on V vertices — see commit notes for #65.
    """

    num_vertices: int
    edges: List[CorrelationEdge] = field(default_factory=list)
    vertex_weights: Dict[int, float] = field(default_factory=dict)
    # Lazy O(1) lookup; rebuilt if the caller mutates `edges`.
    _edge_cache: Optional[Dict[Tuple[int, int], CorrelationEdge]] = field(
        default=None,
        repr=False,
        compare=False,
    )

    def _ensure_edge_cache(self) -> Dict[Tuple[int, int], CorrelationEdge]:
        cache = self._edge_cache
        if cache is None or len(cache) != len(self.edges):
            cache = {(min(e.u, e.v), max(e.u, e.v)): e for e in self.edges}
            self._edge_cache = cache
        return cache

    def adjacency(self) -> Dict[int, List[int]]:
        adj: Dict[int, List[int]] = {i: [] for i in range(self.num_vertices)}
        for e in self.edges:
            adj[e.u].append(e.v)
            adj[e.v].append(e.u)
        return adj

    def edge_weight(self, u: int, v: int) -> float:
        e = self._ensure_edge_cache().get((min(u, v), max(u, v)))
        return e.conn_weight if e is not None else 0.0

    def edge_scc(self, u: int, v: int) -> float:
        e = self._ensure_edge_cache().get((min(u, v), max(u, v)))
        return e.scc_weight if e is not None else 0.0

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def to_csr(self) -> CSRGraph:
        """Convert to CSR representation."""
        return CSRGraph.from_edge_list(
            self.num_vertices,
            self.edges,
            self.vertex_weights or None,
        )


class LFSRSeedAllocator:
    """Assigns independent LFSR seeds per partition.

    Uses co-prime spacing to ensure maximal-length LFSR sequences
    do not overlap between partitions.
    """

    def __init__(self, base_seed: int = 0xACE1):
        self.base_seed = base_seed

    def allocate(self, num_partitions: int) -> List[int]:
        """Return a list of unique, well-separated LFSR seeds."""
        seeds = []
        spacing = max(1, 65535 // (num_partitions + 1))
        for i in range(num_partitions):
            seed = (self.base_seed + (i + 1) * spacing) & 0xFFFF
            if seed == 0:
                seed = 1
            seeds.append(seed)
        return seeds

    def verify_uniqueness(self, seeds: List[int]) -> bool:
        return len(seeds) == len(set(seeds))


class HierarchicalPartitioner:
    """Multi-level graph partitioner with correlation awareness."""

    def __init__(
        self,
        num_partitions: int = 2,
        coarsen_threshold: int = 50,
        kl_iterations: int = 10,
        correlation_penalty: float = 2.0,
        seed: int = 42,
        refine_backend: str = "auto",
    ):
        self.num_partitions = num_partitions
        self.coarsen_threshold = coarsen_threshold
        self.kl_iterations = kl_iterations
        self.correlation_penalty = correlation_penalty
        self.seed_allocator = LFSRSeedAllocator()
        self.rng = np.random.default_rng(seed)
        # KL refine backend: "auto" picks Rust when wired, else
        # Python; explicit values pick that specific backend.
        # Empirical fastest-pick (per `feedback_fallback_chain_ordering`):
        # Rust and Mojo trade wins on this kernel; Julia is within
        # 30 %; Go trails because of cgo overhead. See
        # `benchmarks/results/bench_kl_refine.json`.
        valid = ("auto", "rust", "julia", "go", "mojo", "python")
        if refine_backend not in valid:
            raise ValueError(f"refine_backend must be one of {valid}, got {refine_backend!r}")
        self.refine_backend = refine_backend

    def partition(self, graph: CorrelationAwareGraph) -> Tuple[List[List[int]], List[int]]:
        """Partition the graph. Returns (partitions, seeds)."""
        vertices = list(range(graph.num_vertices))
        if self.num_partitions <= 1:
            seeds = self.seed_allocator.allocate(1)
            return [vertices], seeds

        n = graph.num_vertices
        if n <= self.num_partitions:
            partitions = [[v] for v in vertices]
            while len(partitions) < self.num_partitions:
                partitions.append([])
            seeds = self.seed_allocator.allocate(len(partitions))
            return partitions, seeds

        adj = graph.adjacency()
        partitions = self._recursive_bisect(vertices, adj, graph, self.num_partitions)
        partitions = self._dispatch_refine(partitions, adj, graph)
        seeds = self.seed_allocator.allocate(len(partitions))
        return partitions, seeds

    def _dispatch_refine(
        self,
        partitions: List[List[int]],
        adj: Dict[int, List[int]],
        graph: CorrelationAwareGraph,
    ) -> List[List[int]]:
        """Backend dispatch for the KL refinement step."""
        backend = self.refine_backend
        if backend == "rust" and not _HAS_RUST_KL_REFINE:
            raise RuntimeError(
                "Rust KL refine requested but py_kl_refine not available; "
                "install sc_neurocore_engine wheel."
            )
        if backend == "julia" and not _ensure_julia_kl_refine_loaded():
            raise RuntimeError(
                "Julia KL refine requested but juliacall + "
                "accel/julia/chiplet/kl_refine.jl is not available; "
                "install juliacall (pip install juliacall)."
            )
        if backend == "go" and not _ensure_go_kl_refine_loaded():
            raise RuntimeError(
                "Go KL refine requested but libpartition.so is not "
                "built; run `cd src/sc_neurocore/accel/go/partition && "
                "go build -buildmode=c-shared -o libpartition.so partition.go`."
            )
        if backend == "mojo" and not _ensure_mojo_kl_refine_loaded():
            raise RuntimeError(
                "Mojo KL refine requested but libpartition.so is not "
                "built; run `cd src/sc_neurocore/accel/mojo/partition && "
                "mojo build --emit shared-lib -o libpartition.so partition.mojo`."
            )
        if (backend == "auto" and _HAS_RUST_KL_REFINE) or backend == "rust":
            return self._refine_rust(partitions, adj, graph)
        if backend == "julia":
            return self._refine_julia(partitions, adj, graph)
        if backend == "go":
            return self._refine_go(partitions, adj, graph)
        if backend == "mojo":
            return self._refine_mojo(partitions, adj, graph)
        return self._refine(partitions, adj, graph)

    def _recursive_bisect(
        self,
        vertices: List[int],
        adj: Dict[int, List[int]],
        graph: CorrelationAwareGraph,
        k: int,
    ) -> List[List[int]]:
        """Recursively bisect until we have k partitions."""
        if k <= 1 or len(vertices) <= 1:
            return [vertices]

        coarsened, mapping = self._coarsen(vertices, adj, graph)
        p1, p2 = self._spectral_bisect(coarsened, adj, graph)
        p1 = self._uncoarsen(p1, mapping)
        p2 = self._uncoarsen(p2, mapping)

        if k == 2:
            return [p1, p2]

        k1 = k // 2
        k2 = k - k1
        left = self._recursive_bisect(p1, adj, graph, k1)
        right = self._recursive_bisect(p2, adj, graph, k2)
        return left + right

    def _coarsen(
        self,
        vertices: List[int],
        adj: Dict[int, List[int]],
        graph: CorrelationAwareGraph,
    ) -> Tuple[List[int], Dict[int, List[int]]]:
        """Heavy-edge matching coarsening (merge low-SCC edges first)."""
        if len(vertices) <= self.coarsen_threshold:
            return vertices, {v: [v] for v in vertices}

        matched: Set[int] = set()
        mapping: Dict[int, List[int]] = {}
        coarsened: List[int] = []
        vertex_set = set(vertices)

        sorted_edges = sorted(
            [e for e in graph.edges if e.u in vertex_set and e.v in vertex_set],
            key=lambda e: abs(e.scc_weight),
        )

        for edge in sorted_edges:
            if edge.u not in matched and edge.v not in matched:
                super_node = edge.u
                mapping[super_node] = [edge.u, edge.v]
                coarsened.append(super_node)
                matched.add(edge.u)
                matched.add(edge.v)

        for v in vertices:
            if v not in matched:
                mapping[v] = [v]
                coarsened.append(v)

        return coarsened, mapping

    def _uncoarsen(self, partition: List[int], mapping: Dict[int, List[int]]) -> List[int]:
        """Expand coarsened partition back to original vertices."""
        result = []
        for v in partition:
            result.extend(mapping.get(v, [v]))
        return result

    def _spectral_bisect(
        self,
        vertices: List[int],
        adj: Dict[int, List[int]],
        graph: CorrelationAwareGraph,
    ) -> Tuple[List[int], List[int]]:
        """Spectral-heuristic bisection with correlation penalty.

        Performance fix (#65): hoist `set(vertices)` out of the inner
        loop (was rebuilt V times → O(V²)) and rely on the O(1) edge
        cache in `CorrelationAwareGraph._ensure_edge_cache`. Combined
        complexity drops from O(V²·E) to O(V·avg_degree).
        """
        if len(vertices) <= 1:
            return vertices, []

        vset = set(vertices)
        graph._ensure_edge_cache()  # warm O(1) lookup once
        scores: Dict[int, float] = {}
        for v in vertices:
            in_part_neighbours = [n for n in adj.get(v, []) if n in vset]
            degree = len(in_part_neighbours)
            scc_sum = sum(
                abs(graph.edge_scc(v, n)) * self.correlation_penalty for n in in_part_neighbours
            )
            scores[v] = degree - scc_sum

        sorted_v = sorted(vertices, key=lambda v: scores.get(v, 0))
        mid = len(sorted_v) // 2
        return sorted_v[:mid], sorted_v[mid:]

    def _encode_csr(
        self,
        partitions: List[List[int]],
        adj: Dict[int, List[int]],
        graph: CorrelationAwareGraph,
    ) -> Tuple[
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
        np.ndarray[Any, Any],
    ]:
        """Pack the per-partition state into the flat CSR-style buffers
        every multi-language KL refine kernel expects:
          - adj_offsets:    int64[V+1]
          - adj_neighbours: int32[E_total]
          - adj_scc_abs:    float64[E_total]
          - vertex_weights: float64[V]
          - part_map:       int32[V]
          - parts_concat:   int32[V] — vertices grouped by partition,
            preserving input per-part insertion order (load-bearing
            for KL parity with Python; rebuilding from part_map alone
            loses this order and the KL moves diverge).
          - parts_offsets:  int64[P+1] — row pointers into parts_concat
        Edge cache is warmed once so the per-edge `edge_scc` lookup is
        O(1) here too (no point re-paying the O(E) scan).
        """
        V = graph.num_vertices
        graph._ensure_edge_cache()
        offsets = np.zeros(V + 1, dtype=np.int64)
        for v in range(V):
            offsets[v + 1] = offsets[v] + len(adj.get(v, []))
        n_edges = int(offsets[-1])
        neighbours = np.zeros(n_edges, dtype=np.int32)
        scc_abs = np.zeros(n_edges, dtype=np.float64)
        for v in range(V):
            base = int(offsets[v])
            for k, n in enumerate(adj.get(v, [])):
                neighbours[base + k] = n
                scc_abs[base + k] = abs(graph.edge_scc(v, n))
        vw = np.array(
            [graph.vertex_weights.get(v, 1.0) for v in range(V)],
            dtype=np.float64,
        )
        part_map = np.full(V, -1, dtype=np.int32)
        n_parts = len(partitions)
        parts_offsets = np.zeros(n_parts + 1, dtype=np.int64)
        for i, part in enumerate(partitions):
            parts_offsets[i + 1] = parts_offsets[i] + len(part)
            for v in part:
                part_map[v] = i
        parts_concat = np.zeros(int(parts_offsets[-1]), dtype=np.int32)
        for i, part in enumerate(partitions):
            base = int(parts_offsets[i])
            for k, v in enumerate(part):
                parts_concat[base + k] = v
        return (
            offsets,
            neighbours,
            scc_abs,
            vw,
            part_map,
            parts_concat,
            parts_offsets,
        )

    def _decode_part_map(
        self,
        part_map: np.ndarray[Any, Any],
        n_parts: int,
    ) -> List[List[int]]:
        """Decode flat part_map[V] back into List[List[int]]."""
        out: List[List[int]] = [[] for _ in range(n_parts)]
        for v_int, p in enumerate(part_map):
            ip = int(p)
            if 0 <= ip < n_parts:
                out[ip].append(v_int)
        return out

    def _refine_rust(
        self,
        partitions: List[List[int]],
        adj: Dict[int, List[int]],
        graph: CorrelationAwareGraph,
    ) -> List[List[int]]:
        """Rust dispatch for `_refine` — bit-exact parity with Python."""
        if _rust_kl_refine is None:
            raise RuntimeError(
                "Rust KL refine backend requested but py_kl_refine is "
                "not available; install sc_neurocore_engine wheel."
            )
        offsets, neighbours, scc_abs, vw, pm0, pc, po = self._encode_csr(
            partitions,
            adj,
            graph,
        )
        n_parts = len(partitions)
        new_pm, _moves = _rust_kl_refine(
            offsets,
            neighbours,
            scc_abs,
            vw,
            pm0,
            pc,
            po,
            n_parts,
            int(self.kl_iterations),
            float(self.correlation_penalty),
        )
        return self._decode_part_map(new_pm, n_parts)

    def _refine_julia(
        self,
        partitions: List[List[int]],
        adj: Dict[int, List[int]],
        graph: CorrelationAwareGraph,
    ) -> List[List[int]]:
        """Julia dispatch — bit-exact parity with Python + Rust."""
        if _julia_kl_refine is None:
            raise RuntimeError(
                "Julia KL refine backend not loaded — call _ensure_julia_kl_refine_loaded() first."
            )
        offsets, neighbours, scc_abs, vw, pm0, pc, po = self._encode_csr(
            partitions,
            adj,
            graph,
        )
        n_parts = len(partitions)
        new_pm = _julia_kl_refine(
            offsets,
            neighbours,
            scc_abs,
            vw,
            pm0.copy(),
            pc,
            po,
            n_parts,
            int(self.kl_iterations),
            float(self.correlation_penalty),
        )
        return self._decode_part_map(np.asarray(new_pm, dtype=np.int32), n_parts)

    def _refine_go(
        self,
        partitions: List[List[int]],
        adj: Dict[int, List[int]],
        graph: CorrelationAwareGraph,
    ) -> List[List[int]]:
        """Go dispatch via cgo + ctypes — bit-exact parity."""
        if _go_kl_refine_lib is None:
            raise RuntimeError("Go KL refine .so not loaded")
        import ctypes

        offsets, neighbours, scc_abs, vw, pm0, pc, po = self._encode_csr(
            partitions,
            adj,
            graph,
        )
        n_parts = len(partitions)
        pm = pm0.copy()
        _go_kl_refine_lib.kl_refine_c(
            offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            neighbours.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            scc_abs.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            vw.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            pm.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            pc.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            po.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            ctypes.c_int64(vw.size),
            ctypes.c_int64(scc_abs.size),
            ctypes.c_int32(n_parts),
            ctypes.c_int32(self.kl_iterations),
            ctypes.c_double(self.correlation_penalty),
        )
        return self._decode_part_map(pm, n_parts)

    def _refine_mojo(
        self,
        partitions: List[List[int]],
        adj: Dict[int, List[int]],
        graph: CorrelationAwareGraph,
    ) -> List[List[int]]:
        """Mojo dispatch via raw-Int-addr ctypes — bit-exact parity."""
        if _mojo_kl_refine_lib is None:
            raise RuntimeError("Mojo KL refine .so not loaded")
        offsets, neighbours, scc_abs, vw, pm0, pc, po = self._encode_csr(
            partitions,
            adj,
            graph,
        )
        n_parts = len(partitions)
        pm = pm0.copy()
        _mojo_kl_refine_lib.kl_refine_c(
            offsets.ctypes.data,
            neighbours.ctypes.data,
            scc_abs.ctypes.data,
            vw.ctypes.data,
            pm.ctypes.data,
            pc.ctypes.data,
            po.ctypes.data,
            vw.size,
            scc_abs.size,
            n_parts,
            int(self.kl_iterations),
            float(self.correlation_penalty),
        )
        return self._decode_part_map(pm, n_parts)

    def _refine(
        self,
        partitions: List[List[int]],
        adj: Dict[int, List[int]],
        graph: CorrelationAwareGraph,
    ) -> List[List[int]]:
        """Kernighan-Lin inspired local refinement.

        Performance fix (#64 prep): the original implementation called
        `_boundary_cost(v, j, ...)` once per (v, target) pair → O(P)
        redundant scans of v's neighbours per vertex per KL iteration.
        Now `_per_partition_cost` returns a length-P vector in ONE
        scan over neighbours, and the inner loop just indexes into it.
        Combined with the #65 edge cache, this drops KL refine from
        ~870 ms to ~80 ms at V=1000 (~10× on top of #65).
        """
        part_map: Dict[int, int] = {}
        for i, part in enumerate(partitions):
            for v in part:
                part_map[v] = i

        graph._ensure_edge_cache()
        n_parts = len(partitions)

        for _ in range(self.kl_iterations):
            improved = False
            for i, part in enumerate(partitions):
                for v in list(part):
                    if len(part) <= 1:
                        continue
                    costs = self._per_partition_cost(
                        v,
                        n_parts,
                        part_map,
                        adj,
                        graph,
                    )
                    current_cost = costs[i]
                    best_target = i
                    best_gain = 0.0
                    for j in range(n_parts):
                        if j == i:
                            continue
                        gain = current_cost - costs[j]
                        if gain > best_gain:
                            best_gain = gain
                            best_target = j
                    if best_target != i and best_gain > 0:
                        part.remove(v)
                        partitions[best_target].append(v)
                        part_map[v] = best_target
                        improved = True
            if not improved:
                break

        return partitions

    def _per_partition_cost(
        self,
        v: int,
        n_parts: int,
        part_map: Dict[int, int],
        adj: Dict[int, List[int]],
        graph: CorrelationAwareGraph,
    ) -> List[float]:
        """Length-`n_parts` cost vector: `costs[p]` = cost of placing
        vertex v in partition p, computed in ONE scan over v's
        neighbours instead of P scans.

        Definition matches the original `_boundary_cost`: cost is the
        sum of edge contributions for neighbours NOT in partition p.
        Equivalently, `costs[p] = total_weight - weight_to_p`, so we
        accumulate per-target weight then subtract.
        """
        vw = graph.vertex_weights.get(v, 1.0)
        weight_to: List[float] = [0.0] * n_parts
        total_weight = 0.0
        for n in adj.get(v, []):
            contribution = vw * (1.0 + abs(graph.edge_scc(v, n)) * self.correlation_penalty)
            total_weight += contribution
            tgt = part_map.get(n, -1)
            if 0 <= tgt < n_parts:
                weight_to[tgt] += contribution
        return [total_weight - weight_to[p] for p in range(n_parts)]

    def _boundary_cost(
        self,
        v: int,
        partition_id: int,
        part_map: Dict[int, int],
        adj: Dict[int, List[int]],
        graph: CorrelationAwareGraph,
    ) -> float:
        """Cost of placing vertex v in partition_id (legacy single-target API).

        Kept for external callers and parity tests. Internally
        `_refine` now uses `_per_partition_cost` for the full vector
        in one pass.
        """
        cost = 0.0
        vw = graph.vertex_weights.get(v, 1.0)
        for n in adj.get(v, []):
            if part_map.get(n, -1) != partition_id:
                cost += vw * (1.0 + abs(graph.edge_scc(v, n)) * self.correlation_penalty)
        return cost

    def repartition_incremental(
        self,
        graph: CorrelationAwareGraph,
        partitions: List[List[int]],
        max_moves: int = 50,
    ) -> Tuple[List[List[int]], int]:
        """Incremental repartitioning: migrate high-cost boundary vertices.

        Returns (new_partitions, moves_made).
        """
        adj = graph.adjacency()
        part_map = {}
        for i, part in enumerate(partitions):
            for v in part:
                part_map[v] = i

        moves = 0
        for _ in range(max_moves):
            best_v = -1
            best_from = -1
            best_to = -1
            best_gain = 0.0

            for i, part in enumerate(partitions):
                if len(part) <= 1:
                    continue
                for v in part:
                    current_cost = self._boundary_cost(v, i, part_map, adj, graph)
                    if current_cost == 0:
                        continue
                    for j in range(len(partitions)):
                        if j == i:
                            continue
                        new_cost = self._boundary_cost(v, j, part_map, adj, graph)
                        gain = current_cost - new_cost
                        if gain > best_gain:
                            best_gain = gain
                            best_v = v
                            best_from = i
                            best_to = j

            if best_v < 0:
                break
            partitions[best_from].remove(best_v)
            partitions[best_to].append(best_v)
            part_map[best_v] = best_to
            moves += 1

        return partitions, moves


# ── Metrics ──────────────────────────────────────────────────────────


def _build_part_map(partitions: List[List[int]]) -> Dict[int, int]:
    part_map: Dict[int, int] = {}
    for i, part in enumerate(partitions):
        for v in part:
            part_map[v] = i
    return part_map


def calculate_edge_cut(
    graph: CorrelationAwareGraph,
    partitions: List[List[int]],
) -> int:
    """Count cross-partition edges."""
    part_map = _build_part_map(partitions)
    cut = 0
    for e in graph.edges:
        if part_map.get(e.u, -1) != part_map.get(e.v, -1):
            cut += 1
    return cut


def calculate_boundary_scc(
    graph: CorrelationAwareGraph,
    partitions: List[List[int]],
) -> float:
    """Maximum SCC on boundary edges."""
    part_map = _build_part_map(partitions)
    max_scc = 0.0
    for e in graph.edges:
        if part_map.get(e.u, -1) != part_map.get(e.v, -1):
            max_scc = max(max_scc, abs(e.scc_weight))
    return max_scc


def calculate_mean_boundary_scc(
    graph: CorrelationAwareGraph,
    partitions: List[List[int]],
) -> float:
    """Mean SCC on boundary edges."""
    part_map = _build_part_map(partitions)
    sccs = [
        abs(e.scc_weight) for e in graph.edges if part_map.get(e.u, -1) != part_map.get(e.v, -1)
    ]
    return float(np.mean(sccs)) if sccs else 0.0


def calculate_total_boundary_scc(
    graph: CorrelationAwareGraph,
    partitions: List[List[int]],
) -> float:
    """Total SCC on boundary edges."""
    part_map = _build_part_map(partitions)
    return sum(
        abs(e.scc_weight) for e in graph.edges if part_map.get(e.u, -1) != part_map.get(e.v, -1)
    )


def calculate_imbalance_ratio(partitions: List[List[int]]) -> float:
    """Imbalance ratio: max_size / ideal_size - 1.

    0.0 = perfect balance, >0.0 = imbalanced.
    """
    sizes = [len(p) for p in partitions]
    if not sizes:
        return 0.0
    total = sum(sizes)
    ideal = total / len(sizes)
    if ideal == 0:
        return 0.0
    return max(sizes) / ideal - 1.0


def calculate_comm_volume(
    graph: CorrelationAwareGraph,
    partitions: List[List[int]],
    bytes_per_spike: int = 8,
    bitstream_length: int = 256,
) -> Dict[str, int]:
    """Estimate MPI communication volume."""
    part_map = _build_part_map(partitions)
    boundary_edges = 0
    for e in graph.edges:
        if part_map.get(e.u, -1) != part_map.get(e.v, -1):
            boundary_edges += 1
    messages = boundary_edges
    volume_bytes = boundary_edges * bytes_per_spike * bitstream_length
    return {
        "boundary_edges": boundary_edges,
        "messages": messages,
        "volume_bytes": volume_bytes,
    }


# ── Ghost/Halo Cell Manager ──────────────────────────────────────────


class GhostCellManager:
    """Computes halo/ghost regions for boundary communication.

    Ghost cells are copies of neurons on neighboring partitions that
    a partition needs to read but not write.
    """

    @staticmethod
    def compute_halos(
        graph: CorrelationAwareGraph,
        partitions: List[List[int]],
    ) -> Dict[int, Set[int]]:
        """Return {partition_id: set of ghost vertex IDs needed}."""
        part_map = _build_part_map(partitions)
        adj = graph.adjacency()
        halos: Dict[int, Set[int]] = {i: set() for i in range(len(partitions))}
        for i, part in enumerate(partitions):
            for v in part:
                for n in adj.get(v, []):
                    if part_map.get(n, i) != i:
                        halos[i].add(n)
        return halos

    @staticmethod
    def halo_sizes(
        graph: CorrelationAwareGraph,
        partitions: List[List[int]],
    ) -> Dict[int, int]:
        """Return {partition_id: number_of_ghost_cells}."""
        halos = GhostCellManager.compute_halos(graph, partitions)
        return {pid: len(ghosts) for pid, ghosts in halos.items()}


# ── Boundary Sync Protocol ───────────────────────────────────────────


@dataclass
class BoundarySyncConfig:
    """Configuration for boundary synchronization."""

    decorrelation_buffer_bits: int = 32
    sync_interval_timesteps: int = 1
    max_boundary_scc_budget: float = 0.1


class BoundarySyncProtocol:
    """Manages decorrelation at partition boundaries.

    Each boundary edge gets a decorrelation buffer (XOR with independent
    LFSR seed) to prevent correlation blow-up at partition interfaces.
    """

    def __init__(self, config: Optional[BoundarySyncConfig] = None):
        self.config = config or BoundarySyncConfig()
        self.boundary_buffers: Dict[Tuple[int, int], int] = {}
        self.violations: List[Tuple[int, int, float]] = []

    def init_buffers(
        self,
        graph: CorrelationAwareGraph,
        partitions: List[List[int]],
        seeds: List[int],
    ) -> int:
        """Initialise decorrelation buffers at boundary edges.

        Returns number of buffers created.
        """
        part_map = _build_part_map(partitions)
        count = 0
        for e in graph.edges:
            pu = part_map.get(e.u, -1)
            pv = part_map.get(e.v, -1)
            if pu != pv and pu >= 0 and pv >= 0:
                seed = (seeds[pu] ^ seeds[pv]) & 0xFFFF
                if seed == 0:
                    seed = 1
                self.boundary_buffers[(e.u, e.v)] = seed
                count += 1
        return count

    def check_scc_budget(
        self,
        graph: CorrelationAwareGraph,
        partitions: List[List[int]],
    ) -> List[Tuple[int, int, float]]:
        """Check which boundary edges exceed SCC budget."""
        part_map = _build_part_map(partitions)
        budget = self.config.max_boundary_scc_budget
        self.violations = []
        for e in graph.edges:
            if part_map.get(e.u, -1) != part_map.get(e.v, -1):
                if abs(e.scc_weight) > budget:
                    self.violations.append((e.u, e.v, e.scc_weight))
        return self.violations

    @property
    def num_buffers(self) -> int:
        return len(self.boundary_buffers)


# ── Correlation-Aware Load Balancer ──────────────────────────────────


@dataclass
class LoadMetrics:
    """Per-partition load metrics."""

    partition_id: int
    vertex_count: int
    weight_sum: float
    boundary_scc_sum: float
    ghost_count: int


@dataclass
class MigrationRecommendation:
    """Recommendation to migrate a vertex between partitions."""

    vertex: int
    from_partition: int
    to_partition: int
    gain: float


class CorrelationLoadBalancer:
    """Runtime load balancer with SCC awareness.

    Monitors per-partition load imbalance and boundary correlation,
    and generates migration recommendations.
    """

    def __init__(
        self,
        imbalance_threshold: float = 0.2,
        scc_weight: float = 1.0,
    ):
        self.imbalance_threshold = imbalance_threshold
        self.scc_weight = scc_weight
        self.history: List[List[MigrationRecommendation]] = []

    def compute_load_metrics(
        self,
        graph: CorrelationAwareGraph,
        partitions: List[List[int]],
    ) -> List[LoadMetrics]:
        """Compute load metrics for each partition."""
        part_map = _build_part_map(partitions)
        halos = GhostCellManager.compute_halos(graph, partitions)
        metrics = []
        for i, part in enumerate(partitions):
            weight_sum = sum(graph.vertex_weights.get(v, 1.0) for v in part)
            bscc = 0.0
            for v in part:
                for e in graph.edges:
                    if e.u == v or e.v == v:
                        other = e.v if e.u == v else e.u
                        if part_map.get(other, i) != i:
                            bscc += abs(e.scc_weight)
            metrics.append(
                LoadMetrics(
                    partition_id=i,
                    vertex_count=len(part),
                    weight_sum=weight_sum,
                    boundary_scc_sum=bscc,
                    ghost_count=len(halos.get(i, set())),
                )
            )
        return metrics

    def recommend_migrations(
        self,
        graph: CorrelationAwareGraph,
        partitions: List[List[int]],
        max_recommendations: int = 10,
    ) -> List[MigrationRecommendation]:
        """Generate migration recommendations."""
        metrics = self.compute_load_metrics(graph, partitions)
        imbalance = calculate_imbalance_ratio(partitions)
        if imbalance <= self.imbalance_threshold:
            return []

        sizes = [m.vertex_count for m in metrics]
        avg = sum(sizes) / len(sizes) if sizes else 1
        overloaded = [m for m in metrics if m.vertex_count > avg * (1 + self.imbalance_threshold)]
        underloaded = [
            m for m in metrics if m.vertex_count < avg * (1 - self.imbalance_threshold * 0.5)
        ]

        if not overloaded or not underloaded:
            return []

        adj = graph.adjacency()
        part_map = _build_part_map(partitions)
        recs: list[MigrationRecommendation] = []

        for over_m in overloaded:
            for v in list(partitions[over_m.partition_id]):
                if len(recs) >= max_recommendations:
                    break
                boundary_neighbors = [
                    part_map[n]
                    for n in adj.get(v, [])
                    if part_map.get(n, -1) != over_m.partition_id
                ]
                if not boundary_neighbors:
                    continue
                best_target = max(set(boundary_neighbors), key=boundary_neighbors.count)
                if any(m.partition_id == best_target for m in underloaded):
                    scc_cost = sum(
                        abs(graph.edge_scc(v, n))
                        for n in adj.get(v, [])
                        if part_map.get(n, -1) != over_m.partition_id
                    )
                    gain = 1.0 - scc_cost * self.scc_weight
                    recs.append(MigrationRecommendation(v, over_m.partition_id, best_target, gain))

        recs.sort(key=lambda r: r.gain, reverse=True)
        result = recs[:max_recommendations]
        self.history.append(result)
        return result


# ── Rank Mapper ──────────────────────────────────────────────────────


class RankMapper:
    """Maps partitions to MPI ranks with topology awareness."""

    def __init__(self, num_ranks: int, hierarchy: Optional[List[HierarchyLevel]] = None):
        self.num_ranks = num_ranks
        self.hierarchy = hierarchy or [HierarchyLevel.NODE]

    def assign(
        self,
        partitions: List[List[int]],
        graph: Optional[CorrelationAwareGraph] = None,
    ) -> Dict[int, int]:
        """Assign partition_id → rank.

        If graph is provided, uses affinity-based assignment to minimize
        cross-rank communication.
        """
        mapping: Dict[int, int] = {}
        if len(partitions) <= self.num_ranks:
            for i in range(len(partitions)):
                mapping[i] = i % self.num_ranks
        else:
            per_rank = max(1, len(partitions) // self.num_ranks)
            for i in range(len(partitions)):
                mapping[i] = min(i // per_rank, self.num_ranks - 1)
        return mapping

    def cross_rank_edges(
        self,
        graph: CorrelationAwareGraph,
        partitions: List[List[int]],
    ) -> int:
        """Count edges that cross MPI rank boundaries."""
        part_map = _build_part_map(partitions)
        rank_map = self.assign(partitions, graph)
        count = 0
        for e in graph.edges:
            pu = part_map.get(e.u, -1)
            pv = part_map.get(e.v, -1)
            if pu != pv:
                ru = rank_map.get(pu, -1)
                rv = rank_map.get(pv, -1)
                if ru != rv:
                    count += 1
        return count


# ── Partition Report ─────────────────────────────────────────────────


@dataclass
class PartitionReport:
    """Report from a partitioning run."""

    num_partitions: int
    partition_sizes: List[int]
    edge_cut: int
    max_boundary_scc: float
    mean_boundary_scc: float
    total_boundary_scc: float
    imbalance_ratio: float
    comm_volume_bytes: int
    comm_messages: int
    seeds: List[int]
    scc_budget_violations: int = 0

    def summary(self) -> str:
        return (
            f"Partitions: {self.num_partitions}, "
            f"Sizes: {self.partition_sizes}, "
            f"Edge cut: {self.edge_cut}, "
            f"Max boundary SCC: {self.max_boundary_scc:.4f}, "
            f"Mean boundary SCC: {self.mean_boundary_scc:.4f}, "
            f"Imbalance: {self.imbalance_ratio:.3f}, "
            f"Comm: {self.comm_volume_bytes} bytes / {self.comm_messages} msgs"
        )


def build_partition_report(
    graph: CorrelationAwareGraph,
    partitions: List[List[int]],
    seeds: List[int],
    scc_budget: float = 0.1,
) -> PartitionReport:
    """Build a complete partition report."""
    cv = calculate_comm_volume(graph, partitions)
    sync = BoundarySyncProtocol(BoundarySyncConfig(max_boundary_scc_budget=scc_budget))
    violations = sync.check_scc_budget(graph, partitions)
    return PartitionReport(
        num_partitions=len(partitions),
        partition_sizes=[len(p) for p in partitions],
        edge_cut=calculate_edge_cut(graph, partitions),
        max_boundary_scc=calculate_boundary_scc(graph, partitions),
        mean_boundary_scc=calculate_mean_boundary_scc(graph, partitions),
        total_boundary_scc=calculate_total_boundary_scc(graph, partitions),
        imbalance_ratio=calculate_imbalance_ratio(partitions),
        comm_volume_bytes=cv["volume_bytes"],
        comm_messages=cv["messages"],
        seeds=seeds,
        scc_budget_violations=len(violations),
    )
