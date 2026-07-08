# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Polyglot parity tests for Ollivier-Ricci curvature

"""Cross-backend parity for ``ollivier_ricci_curvature``.

The Rust, Julia, Go, and Mojo accelerators must reproduce the pure-NumPy
reference value for every node pair on a representative spread of graph
topologies (complete, ring, star, path, weighted, and random sparse) to
within float64 round-off. Backends that are not built on the host are
skipped, never silently passed.
"""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.math import topology
from sc_neurocore.math.topology import ollivier_ricci_curvature

ATOL = 1e-9


def _complete(n: int) -> np.ndarray:
    g = np.ones((n, n), dtype=np.float64)
    np.fill_diagonal(g, 0.0)
    return g


def _ring(n: int) -> np.ndarray:
    g = np.zeros((n, n), dtype=np.float64)
    for k in range(n):
        g[k, (k + 1) % n] = 1.0
        g[(k + 1) % n, k] = 1.0
    return g


def _star(n: int) -> np.ndarray:
    g = np.zeros((n, n), dtype=np.float64)
    for k in range(1, n):
        g[0, k] = 1.0
        g[k, 0] = 1.0
    return g


def _path(n: int) -> np.ndarray:
    g = np.zeros((n, n), dtype=np.float64)
    for k in range(n - 1):
        g[k, k + 1] = 1.0
        g[k + 1, k] = 1.0
    return g


def _weighted_random(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    g = rng.random((n, n))
    g[g < 0.6] = 0.0
    g = 0.5 * (g + g.T)  # symmetric weighted graph
    np.fill_diagonal(g, 0.0)
    return g


_GRAPHS = {
    "complete7": _complete(7),
    "ring8": _ring(8),
    "star6": _star(6),
    "path6": _path(6),
    "weighted9a": _weighted_random(9, 1),
    "weighted9b": _weighted_random(9, 2),
}


def _node_pairs(n: int) -> list[tuple[int, int]]:
    return [(0, 1), (0, n - 1), (1, n - 2), (2, 3)]


def _reference(graph: np.ndarray, i: int, j: int) -> float:
    return ollivier_ricci_curvature(graph, i, j, backend="python")


# ───────────────────────── availability probes ─────────────────────────


def _rust_available() -> bool:
    return topology._HAS_RUST_TOPOLOGY


def _julia_available() -> bool:
    return topology._ensure_julia_loaded()


def _go_available() -> bool:
    return topology._ensure_go_loaded()


def _mojo_available() -> bool:
    return topology._ensure_mojo_loaded()


def _parametrised_cases() -> list[tuple[str, np.ndarray, int, int]]:
    cases: list[tuple[str, np.ndarray, int, int]] = []
    for name, graph in _GRAPHS.items():
        for i, j in _node_pairs(graph.shape[0]):
            cases.append((name, graph, i, j))
    return cases


_CASES = _parametrised_cases()
_CASE_IDS = [f"{name}-{i}-{j}" for name, _g, i, j in _CASES]


# ───────────────────────── per-backend parity ─────────────────────────


@pytest.mark.skipif(not _rust_available(), reason="Rust topology backend unavailable")
@pytest.mark.parametrize("name,graph,i,j", _CASES, ids=_CASE_IDS)
def test_rust_parity(name: str, graph: np.ndarray, i: int, j: int) -> None:
    expected = _reference(graph, i, j)
    got = ollivier_ricci_curvature(graph, i, j, backend="rust")
    np.testing.assert_allclose(got, expected, atol=ATOL)


@pytest.mark.skipif(not _julia_available(), reason="Julia topology backend unavailable")
@pytest.mark.parametrize("name,graph,i,j", _CASES, ids=_CASE_IDS)
def test_julia_parity(name: str, graph: np.ndarray, i: int, j: int) -> None:
    expected = _reference(graph, i, j)
    got = ollivier_ricci_curvature(graph, i, j, backend="julia")
    np.testing.assert_allclose(got, expected, atol=ATOL)


@pytest.mark.skipif(not _go_available(), reason="Go topology backend unavailable")
@pytest.mark.parametrize("name,graph,i,j", _CASES, ids=_CASE_IDS)
def test_go_parity(name: str, graph: np.ndarray, i: int, j: int) -> None:
    expected = _reference(graph, i, j)
    got = ollivier_ricci_curvature(graph, i, j, backend="go")
    np.testing.assert_allclose(got, expected, atol=ATOL)


@pytest.mark.skipif(not _mojo_available(), reason="Mojo topology backend unavailable")
@pytest.mark.parametrize("name,graph,i,j", _CASES, ids=_CASE_IDS)
def test_mojo_parity(name: str, graph: np.ndarray, i: int, j: int) -> None:
    expected = _reference(graph, i, j)
    got = ollivier_ricci_curvature(graph, i, j, backend="mojo")
    np.testing.assert_allclose(got, expected, atol=ATOL)


# ───────────────────────── auto-dispatch + guards ─────────────────────────


def test_auto_matches_python_reference() -> None:
    graph = _GRAPHS["weighted9a"]
    for i, j in _node_pairs(graph.shape[0]):
        auto = ollivier_ricci_curvature(graph, i, j, backend="auto")
        ref = ollivier_ricci_curvature(graph, i, j, backend="python")
        np.testing.assert_allclose(auto, ref, atol=ATOL)


def test_invalid_backend_name_raises() -> None:
    with pytest.raises(ValueError, match="backend must be"):
        ollivier_ricci_curvature(_complete(3), 0, 1, backend="cuda")


def test_backend_param_preserves_index_validation() -> None:
    with np.testing.assert_raises_regex(ValueError, "integer"):
        ollivier_ricci_curvature(_complete(3), True, 1, backend="python")


@pytest.mark.skipif(_go_available(), reason="Go backend is built; cannot test the unavailable path")
def test_go_unavailable_raises_when_requested() -> None:
    with pytest.raises(RuntimeError, match="Go topology backend"):
        ollivier_ricci_curvature(_complete(3), 0, 1, backend="go")


@pytest.mark.skipif(
    _mojo_available(), reason="Mojo backend is built; cannot test the unavailable path"
)
def test_mojo_unavailable_raises_when_requested() -> None:
    with pytest.raises(RuntimeError, match="Mojo topology backend"):
        ollivier_ricci_curvature(_complete(3), 0, 1, backend="mojo")


def test_self_pair_zero_across_requested_backend() -> None:
    # i == j short-circuits before dispatch, so every backend returns 0.0.
    for backend in ("auto", "python"):
        assert ollivier_ricci_curvature(_complete(4), 2, 2, backend=backend) == 0.0


# ─────────────── backend-availability guards (forced, host-independent) ───────────────
#
# The skip-guarded parity tests above only exercise the *loaded* path of each
# accelerator (and the unavailable path only on a host where that backend is
# absent). These tests force each guard regardless of which accelerators are
# built locally, by overriding the module's backend handles / loaders.


def test_python_curvature_zero_for_disconnected_pair() -> None:
    # Nodes in different components have infinite graph distance, so the
    # curvature is defined as 0 rather than dividing by infinity.
    graph = np.zeros((3, 3), dtype=np.float64)
    graph[0, 1] = graph[1, 0] = 1.0  # node 2 is isolated
    assert ollivier_ricci_curvature(graph, 0, 2, backend="python") == 0.0


def test_rust_dispatch_without_handle_raises(monkeypatch) -> None:
    monkeypatch.setattr(topology, "_rust_ollivier", None)
    with pytest.raises(RuntimeError, match="Rust topology backend probed False"):
        topology._ollivier_ricci_rust(_complete(3), 0, 1)


def test_julia_dispatch_without_module_raises(monkeypatch) -> None:
    monkeypatch.setattr(topology, "_julia_module", None)
    with pytest.raises(RuntimeError, match="Julia topology module not loaded"):
        topology._ollivier_ricci_julia(_complete(3), 0, 1)


def test_go_dispatch_without_lib_raises(monkeypatch) -> None:
    monkeypatch.setattr(topology, "_go_lib", None)
    with pytest.raises(RuntimeError, match="Go topology library not loaded"):
        topology._ollivier_ricci_go(_complete(3), 0, 1)


def test_mojo_dispatch_without_lib_raises(monkeypatch) -> None:
    monkeypatch.setattr(topology, "_mojo_lib", None)
    with pytest.raises(RuntimeError, match="Mojo topology library not loaded"):
        topology._ollivier_ricci_mojo(_complete(3), 0, 1)


def test_rust_backend_requested_but_unavailable_raises(monkeypatch) -> None:
    monkeypatch.setattr(topology, "_HAS_RUST_TOPOLOGY", False)
    with pytest.raises(RuntimeError, match="Rust topology backend requested"):
        ollivier_ricci_curvature(_complete(3), 0, 1, backend="rust")


def test_julia_backend_requested_but_unavailable_raises(monkeypatch) -> None:
    monkeypatch.setattr(topology, "_ensure_julia_loaded", lambda: False)
    with pytest.raises(RuntimeError, match="Julia topology backend requested"):
        ollivier_ricci_curvature(_complete(3), 0, 1, backend="julia")


def test_go_backend_requested_but_unavailable_raises(monkeypatch) -> None:
    monkeypatch.setattr(topology, "_ensure_go_loaded", lambda: False)
    with pytest.raises(RuntimeError, match="Go topology backend requested"):
        ollivier_ricci_curvature(_complete(3), 0, 1, backend="go")


def test_mojo_backend_requested_but_unavailable_raises(monkeypatch) -> None:
    monkeypatch.setattr(topology, "_ensure_mojo_loaded", lambda: False)
    with pytest.raises(RuntimeError, match="Mojo topology backend requested"):
        ollivier_ricci_curvature(_complete(3), 0, 1, backend="mojo")


def test_julia_loader_returns_false_without_juliacall(monkeypatch) -> None:
    import importlib.util as importlib_util

    monkeypatch.setattr(topology, "_julia_module", None)
    monkeypatch.setattr(
        importlib_util, "find_spec", lambda name: None if name == "juliacall" else None
    )
    assert topology._ensure_julia_loaded() is False


def test_julia_loader_returns_false_without_module_file(monkeypatch) -> None:
    monkeypatch.setattr(topology, "_julia_module", None)
    real_isfile = topology._os.path.isfile
    monkeypatch.setattr(
        topology._os.path,
        "isfile",
        lambda p: False if str(p).endswith("topology.jl") else real_isfile(p),
    )
    assert topology._ensure_julia_loaded() is False


@pytest.mark.parametrize(
    "lib_global,ensure_fn",
    [("_go_lib", "_ensure_go_loaded"), ("_mojo_lib", "_ensure_mojo_loaded")],
)
def test_native_loader_returns_false_when_so_absent(monkeypatch, lib_global, ensure_fn) -> None:
    monkeypatch.setattr(topology, lib_global, None)
    real_isfile = topology._os.path.isfile
    monkeypatch.setattr(
        topology._os.path,
        "isfile",
        lambda p: False if str(p).endswith("libtopology.so") else real_isfile(p),
    )
    assert getattr(topology, ensure_fn)() is False


@pytest.mark.parametrize(
    "lib_global,ensure_fn",
    [("_go_lib", "_ensure_go_loaded"), ("_mojo_lib", "_ensure_mojo_loaded")],
)
def test_native_loader_returns_false_on_cdll_oserror(monkeypatch, lib_global, ensure_fn) -> None:
    import ctypes

    monkeypatch.setattr(topology, lib_global, None)
    monkeypatch.setattr(topology._os.path, "isfile", lambda p: True)

    def _raise(_path):
        raise OSError("simulated broken shared object")

    monkeypatch.setattr(ctypes, "CDLL", _raise)
    assert getattr(topology, ensure_fn)() is False


@pytest.mark.parametrize(
    "lib_global,ensure_fn",
    [("_go_lib", "_ensure_go_loaded"), ("_mojo_lib", "_ensure_mojo_loaded")],
)
def test_native_loader_returns_false_when_symbol_missing(
    monkeypatch, lib_global, ensure_fn
) -> None:
    import ctypes

    monkeypatch.setattr(topology, lib_global, None)
    monkeypatch.setattr(topology._os.path, "isfile", lambda p: True)

    class _EmptyLib:
        # No ollivier_ricci_curvature_c attribute, and getattr default returns None.
        def __getattr__(self, name):
            raise AttributeError(name)

    monkeypatch.setattr(ctypes, "CDLL", lambda _path: _EmptyLib())
    assert getattr(topology, ensure_fn)() is False


def test_rust_import_failure_sets_flag_false() -> None:
    # Reload the module with a stand-in sc_neurocore_engine that lacks the
    # curvature symbol, driving the import-time AttributeError fallback, then
    # restore the real module state.
    import importlib
    import sys
    import types

    from tests.module_reload import restore_module_namespace, snapshot_module_namespace

    fake = types.ModuleType("sc_neurocore_engine")  # no py_ollivier_ricci_curvature
    had = sys.modules.get("sc_neurocore_engine")
    sys.modules["sc_neurocore_engine"] = fake
    saved_namespace = snapshot_module_namespace(topology)
    try:
        reloaded = importlib.reload(topology)
        assert reloaded._HAS_RUST_TOPOLOGY is False
        assert reloaded._rust_ollivier is None
    finally:
        if had is not None:
            sys.modules["sc_neurocore_engine"] = had
        else:
            sys.modules.pop("sc_neurocore_engine", None)
        restore_module_namespace(topology, saved_namespace)
