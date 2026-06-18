# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Topological observables for SCPN phase dynamics

"""Topological and geometric observables for coupled oscillator networks.

These are the quick-win implementations from the Holonomic Atlas
mathematical foundations audit (Round 2).

    from sc_neurocore.math.topology import (
        winding_number,
        ollivier_ricci_curvature,
        sheaf_consistency_defect,
    )
"""

from __future__ import annotations

import importlib as _importlib
import os as _os
from typing import Any, Callable, Optional

import numpy as np

# ───────────────────────── backend detection ─────────────────────────
#
# `ollivier_ricci_curvature` is the one compute-bound observable in this
# module (an exact optimal-transport solve per node pair). It carries a
# polyglot accelerator chain — Rust (PyO3), Julia (juliacall), Go (cgo),
# and Mojo (FFI) — that all reproduce the NumPy reference to float64
# round-off. The lighter observables (winding number, sheaf defect,
# connection curvature) are single vectorised NumPy expressions for which
# NumPy is already the fastest path, so they are not accelerated.

_RustOllivier = Callable[..., float]


def _load_rust_ollivier() -> _RustOllivier:
    engine = _importlib.import_module("sc_neurocore_engine")
    return engine.py_ollivier_ricci_curvature  # type: ignore[no-any-return]


try:
    _rust_ollivier: Optional[_RustOllivier] = _load_rust_ollivier()
    _HAS_RUST_TOPOLOGY = True
except (ImportError, AttributeError):
    _rust_ollivier = None
    _HAS_RUST_TOPOLOGY = False

# Lazy accelerator handles (loaded on first explicit request).
_julia_module = None
_HAS_JULIA_TOPOLOGY = False
_go_lib = None
_HAS_GO_TOPOLOGY = False
_mojo_lib = None
_HAS_MOJO_TOPOLOGY = False


def _ensure_julia_loaded() -> bool:
    """Lazy-load the Julia `TopologyAccel` module on first request.

    Julia startup latency is ~5 s — never paid unless ``backend='julia'``
    is requested or selected by ``auto``.
    """
    global _julia_module, _HAS_JULIA_TOPOLOGY
    if _julia_module is not None:
        return True
    import importlib.util as importlib_util

    if importlib_util.find_spec("juliacall") is None:
        return False
    juliacall = _importlib.import_module("juliacall")
    jl = juliacall.Main
    jl_path = _os.path.join(
        _os.path.dirname(__file__), "..", "accel", "julia", "math", "topology.jl"
    )
    jl_path = _os.path.abspath(jl_path)
    if not _os.path.isfile(jl_path):
        return False
    jl.include(jl_path)
    _julia_module = jl.TopologyAccel
    _HAS_JULIA_TOPOLOGY = True
    return True


def _ensure_go_loaded() -> bool:
    """Lazy-load the Go topology shared library on first request.

    Built once via::

        cd src/sc_neurocore/accel/go/topology
        go build -buildmode=c-shared -o libtopology.so topology.go
    """
    global _go_lib, _HAS_GO_TOPOLOGY
    if _go_lib is not None:
        return True
    import ctypes

    so_path = _os.path.join(
        _os.path.dirname(__file__), "..", "accel", "go", "topology", "libtopology.so"
    )
    so_path = _os.path.abspath(so_path)
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "ollivier_ricci_curvature_c", None)
    if fn is None:
        return False
    fn.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int]
    fn.restype = ctypes.c_double
    _go_lib = lib
    _HAS_GO_TOPOLOGY = True
    return True


def _ensure_mojo_loaded() -> bool:
    """Lazy-load the Mojo topology shared library on first request.

    Built once via::

        cd src/sc_neurocore/accel/mojo/math
        mojo build --emit shared-lib -o libtopology.so topology.mojo

    Per ``feedback_mojo_026_ffi_pattern``, the coupling matrix is passed
    as a raw int64 address (numpy ``arr.ctypes.data``) and the Mojo side
    reconstructs ``UnsafePointer[Float64, MutAnyOrigin]`` internally.
    """
    global _mojo_lib, _HAS_MOJO_TOPOLOGY
    if _mojo_lib is not None:
        return True
    import ctypes

    so_path = _os.path.join(
        _os.path.dirname(__file__), "..", "accel", "mojo", "math", "libtopology.so"
    )
    so_path = _os.path.abspath(so_path)
    if not _os.path.isfile(so_path):
        return False
    try:
        lib = ctypes.CDLL(so_path)
    except OSError:
        return False
    fn = getattr(lib, "ollivier_ricci_curvature_c", None)
    if fn is None:
        return False
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_int64]
    fn.restype = ctypes.c_double
    _mojo_lib = lib
    _HAS_MOJO_TOPOLOGY = True
    return True


def _validate_coupling_graph(knm: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    graph = np.asarray(knm, dtype=np.float64)
    if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
        raise ValueError("knm must be a square coupling matrix")
    if graph.shape[0] == 0:
        raise ValueError("knm must contain at least one node")
    if not np.all(np.isfinite(graph)):
        raise ValueError("knm must contain only finite values")
    if np.any(graph < 0.0):
        raise ValueError("knm must be non-negative for Ollivier-Ricci curvature")
    return graph.copy()


def _validate_node_index(name: str, index: int, n_nodes: int) -> int:
    if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
        raise ValueError(f"{name} must be an integer node index")
    index = int(index)
    if index < 0 or index >= n_nodes:
        raise ValueError(f"{name} out of range for coupling graph")
    return index


def _shortest_path_distances(graph: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    adjacency = graph > 0.0
    np.fill_diagonal(adjacency, False)
    n_nodes = graph.shape[0]
    distances = np.full((n_nodes, n_nodes), np.inf, dtype=np.float64)
    for source in range(n_nodes):
        distances[source, source] = 0.0
        frontier = [source]
        while frontier:
            current = frontier.pop(0)
            next_distance = distances[source, current] + 1.0
            for target in np.flatnonzero(adjacency[current]):
                if next_distance < distances[source, target]:
                    distances[source, target] = next_distance
                    frontier.append(int(target))
    return distances


def _lazy_random_walk(
    graph: np.ndarray[Any, Any], node: int, *, idleness: float = 0.5
) -> np.ndarray[Any, Any]:
    distribution = np.zeros(graph.shape[0], dtype=np.float64)
    distribution[node] = idleness
    row = graph[node].copy()
    row[node] = 0.0
    row_sum = float(row.sum())
    if row_sum == 0.0:
        distribution[node] = 1.0
        return distribution
    distribution += (1.0 - idleness) * row / row_sum
    return distribution


def _minimum_transport_cost(
    source: np.ndarray[Any, Any], target: np.ndarray[Any, Any], distances: np.ndarray[Any, Any]
) -> float:
    source_nodes = np.flatnonzero(source > 0.0)
    target_nodes = np.flatnonzero(target > 0.0)
    if source_nodes.size == 0 or target_nodes.size == 0:
        return 0.0

    supply = source[source_nodes].astype(np.float64)
    demand = target[target_nodes].astype(np.float64)
    costs = distances[np.ix_(source_nodes, target_nodes)]
    if not np.all(np.isfinite(costs)):
        return float("inf")

    total_supply = supply.size
    total_demand = demand.size
    source_id = total_supply + total_demand
    sink_id = source_id + 1
    node_count = sink_id + 1
    residual = [[0.0 for _ in range(node_count)] for _ in range(node_count)]
    edge_cost = [[0.0 for _ in range(node_count)] for _ in range(node_count)]

    for idx, amount in enumerate(supply):
        residual[source_id][idx] = float(amount)
    for idx, amount in enumerate(demand):
        residual[total_supply + idx][sink_id] = float(amount)
    for s_idx in range(total_supply):
        for d_idx in range(total_demand):
            u = s_idx
            v = total_supply + d_idx
            residual[u][v] = float("inf")
            edge_cost[u][v] = float(costs[s_idx, d_idx])
            edge_cost[v][u] = -float(costs[s_idx, d_idx])

    required = float(source.sum())
    transported = 0.0
    total_cost = 0.0
    tolerance = 1e-12
    while transported + tolerance < required:
        dist = [float("inf")] * node_count
        parent = [-1] * node_count
        dist[source_id] = 0.0
        for _ in range(node_count - 1):
            updated = False
            for u in range(node_count):
                if not np.isfinite(dist[u]):
                    continue
                for v in range(node_count):
                    if residual[u][v] <= tolerance:
                        continue
                    candidate = dist[u] + edge_cost[u][v]
                    if candidate < dist[v] - tolerance:
                        dist[v] = candidate
                        parent[v] = u
                        updated = True
            if not updated:
                break
        if parent[sink_id] == -1:
            raise ValueError("transport problem is infeasible")

        increment = required - transported
        v = sink_id
        while v != source_id:
            u = parent[v]
            increment = min(increment, residual[u][v])
            v = u
        v = sink_id
        while v != source_id:
            u = parent[v]
            residual[u][v] -= increment
            residual[v][u] += increment
            total_cost += increment * edge_cost[u][v]
            v = u
        transported += increment
    return float(total_cost)


def winding_number(phases: np.ndarray[Any, Any]) -> int:
    """Compute the winding number of a phase trajectory around S^1.

    The winding number counts how many times the phase wraps around
    the circle [0, 2*pi). It is a topological invariant — continuous
    deformations of the trajectory cannot change it.

    Parameters
    ----------
    phases : np.ndarray, shape (T,)
        Time series of phase values (radians).

    Returns
    -------
    int
        Number of complete windings (positive = counterclockwise).
    """
    diffs = np.diff(phases)
    # Unwrap: large jumps indicate wrapping
    diffs = np.where(diffs > np.pi, diffs - 2 * np.pi, diffs)
    diffs = np.where(diffs < -np.pi, diffs + 2 * np.pi, diffs)
    return int(np.round(np.sum(diffs) / (2 * np.pi)))


def _ollivier_ricci_python(graph: np.ndarray[Any, Any], i: int, j: int) -> float:
    """Pure-NumPy Ollivier-Ricci curvature on a validated coupling graph."""
    distances = _shortest_path_distances(graph)
    graph_distance = distances[i, j]
    if not np.isfinite(graph_distance) or graph_distance <= 0.0:
        return 0.0
    mu_i = _lazy_random_walk(graph, i)
    mu_j = _lazy_random_walk(graph, j)
    w1 = _minimum_transport_cost(mu_i, mu_j, distances)
    return float(1.0 - w1 / graph_distance)


def _ollivier_ricci_rust(graph: np.ndarray[Any, Any], i: int, j: int) -> float:
    if _rust_ollivier is None:
        raise RuntimeError("Rust topology backend probed False; cannot dispatch")
    flat = np.ascontiguousarray(graph, dtype=np.float64).ravel(order="C").tolist()
    return float(_rust_ollivier(flat, graph.shape[0], i, j))


def _ollivier_ricci_julia(graph: np.ndarray[Any, Any], i: int, j: int) -> float:
    if _julia_module is None:
        raise RuntimeError("Julia topology module not loaded; cannot dispatch")
    # Julia uses 1-based node indices.
    return float(
        _julia_module.ollivier_ricci_curvature(
            np.ascontiguousarray(graph, dtype=np.float64), i + 1, j + 1
        )
    )


def _ollivier_ricci_go(graph: np.ndarray[Any, Any], i: int, j: int) -> float:
    if _go_lib is None:
        raise RuntimeError("Go topology library not loaded; cannot dispatch")
    import ctypes

    flat = np.ascontiguousarray(graph, dtype=np.float64).ravel(order="C")
    return float(
        _go_lib.ollivier_ricci_curvature_c(
            flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.c_int(graph.shape[0]),
            ctypes.c_int(i),
            ctypes.c_int(j),
        )
    )


def _ollivier_ricci_mojo(graph: np.ndarray[Any, Any], i: int, j: int) -> float:
    if _mojo_lib is None:
        raise RuntimeError("Mojo topology library not loaded; cannot dispatch")
    flat = np.ascontiguousarray(graph, dtype=np.float64).ravel(order="C")
    return float(
        _mojo_lib.ollivier_ricci_curvature_c(
            int(flat.ctypes.data), int(graph.shape[0]), int(i), int(j)
        )
    )


def ollivier_ricci_curvature(
    knm: np.ndarray[Any, Any], i: int, j: int, backend: str = "auto"
) -> float:
    """Compute Ollivier-Ricci curvature between nodes i and j on the coupling graph.

    Ollivier (2009), "Ricci curvature of Markov chains on metric spaces."
    The curvature kappa(i,j) measures how much the neighborhoods of i and j
    overlap. Positive curvature = neighborhoods converge (community structure).
    Negative curvature = neighborhoods diverge (bottleneck).

    kappa(i,j) = 1 - W1(mu_i, mu_j) / d(i,j)
    where mu_i is the lazy random walk distribution from node i,
    and W1 is the Wasserstein-1 distance on the unweighted support graph
    (an exact successive-shortest-path min-cost flow).

    Parameters
    ----------
    knm : np.ndarray, shape (N, N)
        Coupling matrix (non-negative, not necessarily symmetric).
    i, j : int
        Node indices.
    backend : {"auto", "rust", "julia", "go", "mojo", "python"}
        Acceleration backend selector. ``auto`` prefers Rust when the
        ``sc_neurocore_engine`` wheel is built, else the pure-NumPy path.
        The named backends force a specific path and raise ``RuntimeError``
        when that backend is unavailable. Every backend reproduces the
        NumPy reference to float64 round-off.

    Returns
    -------
    float
        Ollivier-Ricci curvature. Returns 0.0 for self or disconnected pairs.
    """
    if backend not in ("auto", "rust", "julia", "go", "mojo", "python"):
        raise ValueError(f"backend must be auto/rust/julia/go/mojo/python, got {backend!r}")

    graph = _validate_coupling_graph(knm)
    n_nodes = graph.shape[0]
    i = _validate_node_index("i", i, n_nodes)
    j = _validate_node_index("j", j, n_nodes)
    if i == j:
        return 0.0

    if backend == "rust" and not _HAS_RUST_TOPOLOGY:
        raise RuntimeError(
            "Rust topology backend requested but py_ollivier_ricci_curvature "
            "is not available; install the sc_neurocore_engine wheel."
        )
    if backend == "julia" and not _ensure_julia_loaded():
        raise RuntimeError(
            "Julia topology backend requested but juliacall + the "
            "accel/julia/math/topology.jl module is not available."
        )
    if backend == "go" and not _ensure_go_loaded():
        raise RuntimeError(
            "Go topology backend requested but libtopology.so is not built; "
            "run `cd src/sc_neurocore/accel/go/topology && "
            "go build -buildmode=c-shared -o libtopology.so topology.go`."
        )
    if backend == "mojo" and not _ensure_mojo_loaded():
        raise RuntimeError(
            "Mojo topology backend requested but libtopology.so is not built; "
            "run `cd src/sc_neurocore/accel/mojo/math && "
            "mojo build --emit shared-lib -o libtopology.so topology.mojo`."
        )

    if backend == "rust" or (backend == "auto" and _HAS_RUST_TOPOLOGY):
        return _ollivier_ricci_rust(graph, i, j)
    if backend == "julia":
        return _ollivier_ricci_julia(graph, i, j)
    if backend == "go":
        return _ollivier_ricci_go(graph, i, j)
    if backend == "mojo":
        return _ollivier_ricci_mojo(graph, i, j)
    return _ollivier_ricci_python(graph, i, j)


def sheaf_consistency_defect(phases: np.ndarray[Any, Any], knm: np.ndarray[Any, Any]) -> float:
    """Compute the sheaf consistency defect for the SCPN phase state.

    In sheaf theory, a global section exists iff the gluing conditions
    are satisfied on all overlaps. For the SCPN, the coupling matrix
    defines the overlaps, and the phase differences weighted by coupling
    measure the failure to glue.

    defect = (1/N^2) * sum_{i,j} |K_ij| * |1 - cos(theta_i - theta_j)|

    When phases are synchronized (all equal), defect = 0.
    When phases are maximally incoherent, defect approaches max(|K|).

    This is equivalent to (1 - Kuramoto_R) weighted by coupling.

    Parameters
    ----------
    phases : np.ndarray, shape (N,)
        Phase values (radians) for each layer/oscillator.
    knm : np.ndarray, shape (N, N)
        Coupling matrix.

    Returns
    -------
    float
        Sheaf consistency defect >= 0. Zero means globally coherent.
    """
    N = len(phases)
    diffs = phases[np.newaxis, :] - phases[:, np.newaxis]
    cost = np.abs(knm) * (1.0 - np.cos(diffs))
    return float(cost.sum() / (N * N))


def connection_curvature(
    phases: np.ndarray[Any, Any], knm: np.ndarray[Any, Any]
) -> np.ndarray[Any, Any]:
    """Compute the connection curvature from PGBO phase dynamics.

    The PGBO covariant derivative u_mu = dphi_mu - alpha * A_mu
    defines a U(1) connection. The curvature F_{ij} = K_{ij} * cos(theta_i - theta_j)
    measures the obstruction to parallel transport between layers i and j.

    Parameters
    ----------
    phases : np.ndarray, shape (N,)
        Phase values.
    knm : np.ndarray, shape (N, N)
        Coupling matrix.

    Returns
    -------
    np.ndarray, shape (N, N)
        Connection curvature matrix. Diagonal is zero.
    """
    diffs = phases[np.newaxis, :] - phases[:, np.newaxis]
    curvature: np.ndarray[Any, Any] = knm * np.cos(diffs)
    return curvature
