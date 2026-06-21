# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Dedicated tests for analysis/spike_stats/dimensionality.py

"""PCA, demixed PCA and factor-analysis tests: degenerate inputs, the
deterministic sign-canonicalised reference, and the NumPy / Rust / Julia / Go /
Mojo backend dispatch contract."""

from __future__ import annotations

import importlib

import numpy as np
import numpy.testing as npt
import pytest

from sc_neurocore.analysis.spike_stats.dimensionality import (
    spike_train_pca,
    demixed_pca,
    factor_analysis,
)

_DIM = importlib.import_module("sc_neurocore.analysis.spike_stats.dimensionality")

_RUST_AVAILABLE = _DIM._rust_pca is not None
_JULIA_AVAILABLE = importlib.util.find_spec("juliacall") is not None
_GO_AVAILABLE = _DIM._ensure_go_dim()
_MOJO_AVAILABLE = _DIM._ensure_mojo_dim()


def _raise_oserror(_path: str) -> object:
    raise OSError("library load failed")


def _trains(n: int = 6, length: int = 400, seed: int = 3) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [(rng.random(length) < (0.1 + 0.04 * i)).astype(np.int8) for i in range(n)]


def _conditions(seed: int = 3) -> dict[int, list[np.ndarray]]:
    t = _trains(6, 400, seed)
    rng = np.random.default_rng(seed + 1)
    extra = [(rng.random(400) < 0.2).astype(np.int8) for _ in range(3)]
    return {0: t[:3], 1: t[3:], 2: extra}


class TestCanonicalSign:
    def test_empty(self) -> None:
        empty = np.empty((3, 0))
        npt.assert_array_equal(_DIM._canonical_sign(empty), empty)

    def test_flips_negative_dominant_column(self) -> None:
        comps = np.array([[-0.9, 0.1], [0.2, 0.8]])
        fixed = _DIM._canonical_sign(comps)
        # column 0's dominant entry (-0.9) becomes positive; column 1 unchanged
        assert fixed[0, 0] > 0
        npt.assert_allclose(fixed[:, 1], comps[:, 1])

    def test_zero_column_keeps_sign(self) -> None:
        comps = np.array([[0.0, 0.5], [0.0, -0.5]])
        fixed = _DIM._canonical_sign(comps)
        npt.assert_array_equal(fixed[:, 0], comps[:, 0])


class TestSpikeTrainPCA:
    def test_typical(self) -> None:
        proj, expl = spike_train_pca(_trains(), n_components=3)
        assert proj.shape == (3, 40)
        assert expl.shape == (3,)
        assert expl[0] >= expl[1] >= expl[2]
        assert expl.sum() <= 1.0 + 1e-9

    def test_python_backend(self) -> None:
        proj, expl = spike_train_pca(_trains(), n_components=2, backend="python")
        assert proj.shape == (2, 40)

    def test_empty_trains(self) -> None:
        proj, expl = spike_train_pca([])
        assert proj.size == 0 and expl.size == 0

    def test_single_neuron(self) -> None:
        proj, expl = spike_train_pca([np.tile([1, 0], 10).astype(np.int8)], bin_size=2)
        assert proj.shape[0] == 1
        npt.assert_array_equal(expl, [1.0])

    def test_unknown_backend(self) -> None:
        with pytest.raises(ValueError, match="not available"):
            spike_train_pca(_trains(), backend="cuda")


class TestDemixedPCA:
    def test_typical(self) -> None:
        proj, expl = demixed_pca(_conditions(), n_components=2)
        assert proj.ndim == 2
        assert expl.size == 2

    def test_python_backend(self) -> None:
        proj, expl = demixed_pca(_conditions(), n_components=2, backend="python")
        assert proj.shape[1] == 2

    def test_insufficient_conditions(self) -> None:
        proj, expl = demixed_pca({0: [np.array([1, 0], dtype=np.int8)]})
        assert proj.size == 0 and expl.size == 0

    def test_empty_condition_skipped(self) -> None:
        # a condition with no neurons is skipped; two usable conditions remain
        conds = {0: _trains(3, 400), 1: [], 2: _trains(3, 400, seed=9)}
        proj, expl = demixed_pca(conds, n_components=2, bin_size=10)
        assert proj.size > 0 and expl.size == 2

    def test_only_empty_conditions(self) -> None:
        # fewer than two usable conditions -> empty result
        proj, expl = demixed_pca({0: _trains(3, 400), 1: []}, n_components=2)
        assert proj.size == 0 and expl.size == 0

    def test_unknown_backend(self) -> None:
        with pytest.raises(ValueError, match="not available"):
            demixed_pca(_conditions(), backend="cuda")


class TestFactorAnalysis:
    def test_typical(self) -> None:
        loadings, psi = factor_analysis(_trains(5), n_factors=2)
        assert loadings.shape == (5, 2)
        assert psi.shape == (5,)
        assert np.all(psi > 0)

    def test_python_backend(self) -> None:
        loadings, psi = factor_analysis(_trains(5), n_factors=2, backend="python")
        assert loadings.shape == (5, 2)

    def test_empty_trains(self) -> None:
        loadings, psi = factor_analysis([])
        assert loadings.size == 0 and psi.size == 0

    def test_unknown_backend(self) -> None:
        with pytest.raises(ValueError, match="not available"):
            factor_analysis(_trains(5), backend="cuda")


def _parity(backend: str, atol: float = 1e-6) -> None:
    trains = _trains()
    conds = _conditions()
    cases = [
        (spike_train_pca, (trains, 3, 10)),
        (demixed_pca, (conds, 2, 10)),
        (factor_analysis, (trains, 2, 10, 30)),
    ]
    for fn, args in cases:
        p0, p1 = fn(*args, backend="python")
        b0, b1 = fn(*args, backend=backend)
        npt.assert_allclose(b0, p0, atol=atol)
        npt.assert_allclose(b1, p1, atol=atol)


@pytest.mark.skipif(not _RUST_AVAILABLE, reason="Rust engine not built")
class TestRustParity:
    def test_parity(self) -> None:
        _parity("rust")


def test_auto_uses_numpy_reference() -> None:
    # auto resolves to the NumPy/LAPACK path (fastest for dense eigendecomposition)
    for fn, args in (
        (spike_train_pca, (_trains(), 3, 10)),
        (factor_analysis, (_trains(5), 2, 10, 30)),
    ):
        a = fn(*args, backend="auto")
        p = fn(*args, backend="python")
        npt.assert_array_equal(a[0], p[0])
        npt.assert_array_equal(a[1], p[1])


@pytest.mark.skipif(not _JULIA_AVAILABLE, reason="juliacall not installed")
class TestJuliaParity:
    def test_parity(self) -> None:
        _parity("julia")

    def test_ensure_cached(self) -> None:
        assert _DIM._ensure_julia_dim() is True
        assert _DIM._ensure_julia_dim() is True


@pytest.mark.skipif(not _GO_AVAILABLE, reason="Go dimensionality library not built")
class TestGoParity:
    def test_parity(self) -> None:
        _parity("go")

    def test_ensure_cached(self) -> None:
        assert _DIM._ensure_go_dim() is True
        assert _DIM._ensure_go_dim() is True


@pytest.mark.skipif(not _MOJO_AVAILABLE, reason="Mojo dimensionality library not built")
class TestMojoParity:
    def test_parity(self) -> None:
        _parity("mojo")

    def test_ensure_cached(self) -> None:
        assert _DIM._ensure_mojo_dim() is True
        assert _DIM._ensure_mojo_dim() is True


def test_rust_probe_returns_none_for_missing_symbol() -> None:
    assert _DIM._load_rust_dim("py_no_such_symbol") is None


def test_rust_probe_returns_none_when_engine_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_name: str) -> object:
        raise ImportError("engine absent")

    monkeypatch.setattr(_DIM._importlib, "import_module", _raise)
    assert _DIM._load_rust_dim("py_pca_components") is None


def test_rust_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_DIM, "_rust_pca", None)
    monkeypatch.setattr(_DIM, "_rust_demixed", None)
    monkeypatch.setattr(_DIM, "_rust_fa", None)
    with pytest.raises(RuntimeError, match="not available"):
        spike_train_pca(_trains(), backend="rust")
    with pytest.raises(RuntimeError, match="not available"):
        demixed_pca(_conditions(), backend="rust")
    with pytest.raises(RuntimeError, match="not available"):
        factor_analysis(_trains(5), backend="rust")
    # auto falls back to the NumPy reference when the engine is absent
    proj, _ = spike_train_pca(_trains(), backend="auto")
    assert proj.shape[0] == 3


def test_ensure_julia_false_without_juliacall(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_DIM, "_julia_dim", None)
    monkeypatch.setattr(_DIM._importlib_util, "find_spec", lambda _name: None)
    assert _DIM._ensure_julia_dim() is False


def test_ensure_julia_false_when_module_file_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_DIM, "_julia_dim", None)
    monkeypatch.setattr(_DIM._importlib_util, "find_spec", lambda _name: object())
    monkeypatch.setattr(_DIM._os.path, "isfile", lambda _path: False)
    assert _DIM._ensure_julia_dim() is False


def test_julia_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_DIM, "_ensure_julia_dim", lambda: False)
    with pytest.raises(RuntimeError, match="Julia dimensionality backend is not available"):
        spike_train_pca(_trains(), backend="julia")
    with pytest.raises(RuntimeError, match="Julia dimensionality backend is not available"):
        demixed_pca(_conditions(), backend="julia")
    with pytest.raises(RuntimeError, match="Julia dimensionality backend is not available"):
        factor_analysis(_trains(5), backend="julia")


def test_ensure_go_false_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_DIM, "_go_dim_lib", None)
    monkeypatch.setattr(_DIM._os.path, "isfile", lambda _path: False)
    assert _DIM._ensure_go_dim() is False


def test_ensure_go_false_on_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_DIM, "_go_dim_lib", None)
    monkeypatch.setattr(_DIM._os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(_DIM._ctypes, "CDLL", _raise_oserror)
    assert _DIM._ensure_go_dim() is False


def test_ensure_go_false_when_symbol_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_DIM, "_go_dim_lib", None)
    monkeypatch.setattr(_DIM._os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(_DIM._ctypes, "CDLL", lambda _path: object())
    assert _DIM._ensure_go_dim() is False


def test_go_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_DIM, "_ensure_go_dim", lambda: False)
    with pytest.raises(RuntimeError, match="Go dimensionality backend is not available"):
        spike_train_pca(_trains(), backend="go")
    with pytest.raises(RuntimeError, match="Go dimensionality backend is not available"):
        demixed_pca(_conditions(), backend="go")
    with pytest.raises(RuntimeError, match="Go dimensionality backend is not available"):
        factor_analysis(_trains(5), backend="go")


def test_ensure_mojo_false_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_DIM, "_mojo_dim_lib", None)
    monkeypatch.setattr(_DIM._os.path, "isfile", lambda _path: False)
    assert _DIM._ensure_mojo_dim() is False


def test_ensure_mojo_false_on_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_DIM, "_mojo_dim_lib", None)
    monkeypatch.setattr(_DIM._os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(_DIM._ctypes, "CDLL", _raise_oserror)
    assert _DIM._ensure_mojo_dim() is False


def test_ensure_mojo_false_when_symbol_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_DIM, "_mojo_dim_lib", None)
    monkeypatch.setattr(_DIM._os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(_DIM._ctypes, "CDLL", lambda _path: object())
    assert _DIM._ensure_mojo_dim() is False


def test_mojo_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_DIM, "_ensure_mojo_dim", lambda: False)
    with pytest.raises(RuntimeError, match="Mojo dimensionality backend is not available"):
        spike_train_pca(_trains(), backend="mojo")
    with pytest.raises(RuntimeError, match="Mojo dimensionality backend is not available"):
        demixed_pca(_conditions(), backend="mojo")
    with pytest.raises(RuntimeError, match="Mojo dimensionality backend is not available"):
        factor_analysis(_trains(5), backend="mojo")
