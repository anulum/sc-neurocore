# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Module-level tests from former test_gpfa.py

"""Module-level tests from former test_gpfa.py."""

from __future__ import annotations

from tests.gpfa_support import *  # noqa: F403


def test_dispatch_python_matches_reference() -> None:
    Y = np.asarray(_synthetic_trains(6, 300), dtype=np.float64)[:, :25]
    c0, d0, r0, tau = gpfa_pca_init(Y, 2, 20.0)
    direct = gpfa_em(Y, c0, d0, r0, tau, 20, 1e-4)
    routed = _gpfa_em_dispatch(Y, c0, d0, r0, tau, 20, 1e-4, "python")
    npt.assert_array_equal(direct[0], routed[0])


def test_ensure_julia_false_without_juliacall(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_GPFA_MODULE, "_julia_gpfa", None)
    monkeypatch.setattr(_GPFA_MODULE._importlib_util, "find_spec", lambda _name: None)
    assert _GPFA_MODULE._ensure_julia_gpfa() is False


def test_ensure_julia_false_when_module_file_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_GPFA_MODULE, "_julia_gpfa", None)
    monkeypatch.setattr(_GPFA_MODULE._importlib_util, "find_spec", lambda _name: object())
    monkeypatch.setattr(_GPFA_MODULE._os.path, "isfile", lambda _path: False)
    assert _GPFA_MODULE._ensure_julia_gpfa() is False


def test_julia_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_GPFA_MODULE, "_ensure_julia_gpfa", lambda: False)
    trains = _synthetic_trains(n_neurons=3, n_samples=120)
    with pytest.raises(RuntimeError, match="Julia GPFA backend is not available"):
        gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=3, backend="julia")


def test_ensure_go_false_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_GPFA_MODULE, "_go_gpfa_lib", None)
    monkeypatch.setattr(_GPFA_MODULE._os.path, "isfile", lambda _path: False)
    assert _GPFA_MODULE._ensure_go_gpfa() is False


def test_ensure_go_false_on_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_GPFA_MODULE, "_go_gpfa_lib", None)
    monkeypatch.setattr(_GPFA_MODULE._os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(_GPFA_MODULE._ctypes, "CDLL", _raise_oserror)
    assert _GPFA_MODULE._ensure_go_gpfa() is False


def test_ensure_go_false_when_symbol_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_GPFA_MODULE, "_go_gpfa_lib", None)
    monkeypatch.setattr(_GPFA_MODULE._os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(_GPFA_MODULE._ctypes, "CDLL", lambda _path: object())
    assert _GPFA_MODULE._ensure_go_gpfa() is False


def test_go_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_GPFA_MODULE, "_ensure_go_gpfa", lambda: False)
    trains = _synthetic_trains(n_neurons=3, n_samples=120)
    with pytest.raises(RuntimeError, match="Go GPFA backend is not available"):
        gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=3, backend="go")


def test_ensure_mojo_false_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_GPFA_MODULE, "_mojo_gpfa_lib", None)
    monkeypatch.setattr(_GPFA_MODULE._os.path, "isfile", lambda _path: False)
    assert _GPFA_MODULE._ensure_mojo_gpfa() is False


def test_ensure_mojo_false_on_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_GPFA_MODULE, "_mojo_gpfa_lib", None)
    monkeypatch.setattr(_GPFA_MODULE._os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(_GPFA_MODULE._ctypes, "CDLL", _raise_oserror)
    assert _GPFA_MODULE._ensure_mojo_gpfa() is False


def test_ensure_mojo_false_when_symbol_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_GPFA_MODULE, "_mojo_gpfa_lib", None)
    monkeypatch.setattr(_GPFA_MODULE._os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(_GPFA_MODULE._ctypes, "CDLL", lambda _path: object())
    assert _GPFA_MODULE._ensure_mojo_gpfa() is False


def test_mojo_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_GPFA_MODULE, "_ensure_mojo_gpfa", lambda: False)
    trains = _synthetic_trains(n_neurons=3, n_samples=120)
    with pytest.raises(RuntimeError, match="Mojo GPFA backend is not available"):
        gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=3, backend="mojo")


def test_rust_probe_returns_none_when_engine_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(_name: str) -> object:
        raise ImportError("engine absent")

    monkeypatch.setattr(_GPFA_MODULE._importlib, "import_module", _raise)
    assert _load_rust_gpfa_em() is None


def test_rust_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_GPFA_MODULE, "_rust_gpfa_em", None)
    trains = _synthetic_trains(n_neurons=3, n_samples=120)
    with pytest.raises(RuntimeError, match="not available"):
        gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=3, backend="rust")
    # auto falls back to the NumPy reference when the Rust engine is absent
    result = gpfa(trains, n_latents=2, bin_ms=20.0, max_iter=3, backend="auto")
    assert result["trajectories"].size > 0
