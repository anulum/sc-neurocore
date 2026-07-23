# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Module-level tests from former test_spike_stats_dimensionality.py

"""Module-level tests from former test_spike_stats_dimensionality.py."""

from __future__ import annotations

from tests.spike_stats_dimensionality_support import *  # noqa: F403

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
