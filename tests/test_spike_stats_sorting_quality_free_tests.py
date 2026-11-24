# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Module-level tests from former test_spike_stats_sorting_quality.py

"""Module-level tests from former test_spike_stats_sorting_quality.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.analysis.spike_stats.sorting_quality import (
    isolation_distance,
    l_ratio,
)
from tests.spike_stats_sorting_quality_support import (
    _cluster_noise,
    _raise_oserror,
    _SQ,
)


def test_rust_probe_returns_none_for_missing_symbol() -> None:
    """The Rust engine probe returns None for a symbol the engine lacks."""
    assert _SQ._load_rust_metric("py_no_such_symbol") is None


def test_rust_probe_returns_none_when_engine_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Rust engine probe returns None when the engine cannot be imported."""

    def _raise(_name: str) -> object:
        raise ImportError("engine absent")

    monkeypatch.setattr(_SQ._importlib, "import_module", _raise)
    assert _SQ._load_rust_metric("py_isolation_distance") is None


def test_rust_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the Rust engine the rust backend raises RuntimeError and auto falls back to NumPy."""
    monkeypatch.setattr(_SQ, "_rust_isolation", None)
    monkeypatch.setattr(_SQ, "_rust_l_ratio", None)
    cluster, noise = _cluster_noise(10, 20, 2)
    with pytest.raises(RuntimeError, match="not available"):
        isolation_distance(cluster, noise, backend="rust")
    with pytest.raises(RuntimeError, match="not available"):
        l_ratio(cluster, noise, backend="rust")
    # auto falls back to the NumPy reference when the engine is absent
    assert np.isfinite(isolation_distance(cluster, noise, backend="auto"))


def test_ensure_julia_false_without_juliacall(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Julia backend probe returns False when juliacall is not installed."""
    monkeypatch.setattr(_SQ, "_julia_sq", None)
    monkeypatch.setattr(_SQ._importlib_util, "find_spec", lambda _name: None)
    assert _SQ._ensure_julia_sq() is False


def test_ensure_julia_false_when_module_file_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Julia backend probe returns False when its Julia module file is missing."""
    monkeypatch.setattr(_SQ, "_julia_sq", None)
    monkeypatch.setattr(_SQ._importlib_util, "find_spec", lambda _name: object())
    monkeypatch.setattr(_SQ._os.path, "isfile", lambda _path: False)
    assert _SQ._ensure_julia_sq() is False


def test_julia_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the Julia backend unavailable, isolation distance raises RuntimeError."""
    monkeypatch.setattr(_SQ, "_ensure_julia_sq", lambda: False)
    cluster, noise = _cluster_noise(10, 20, 2)
    with pytest.raises(RuntimeError, match="Julia sorting-quality backend is not available"):
        isolation_distance(cluster, noise, backend="julia")


def test_ensure_go_false_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Go backend probe returns False when its shared library is missing."""
    monkeypatch.setattr(_SQ, "_go_sq_lib", None)
    monkeypatch.setattr(_SQ._os.path, "isfile", lambda _path: False)
    assert _SQ._ensure_go_sq() is False


def test_ensure_go_false_on_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Go backend probe returns False when loading its library raises OSError."""
    monkeypatch.setattr(_SQ, "_go_sq_lib", None)
    monkeypatch.setattr(_SQ._os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(_SQ._ctypes, "CDLL", _raise_oserror)
    assert _SQ._ensure_go_sq() is False


def test_ensure_go_false_when_symbol_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Go backend probe returns False when the library lacks its symbols."""
    monkeypatch.setattr(_SQ, "_go_sq_lib", None)
    monkeypatch.setattr(_SQ._os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(_SQ._ctypes, "CDLL", lambda _path: object())
    assert _SQ._ensure_go_sq() is False


def test_go_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the Go backend unavailable, the L-ratio raises RuntimeError."""
    monkeypatch.setattr(_SQ, "_ensure_go_sq", lambda: False)
    cluster, noise = _cluster_noise(10, 20, 2)
    with pytest.raises(RuntimeError, match="Go sorting-quality backend is not available"):
        l_ratio(cluster, noise, backend="go")


def test_ensure_mojo_false_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Mojo backend probe returns False when its shared library is missing."""
    monkeypatch.setattr(_SQ, "_mojo_sq_lib", None)
    monkeypatch.setattr(_SQ._os.path, "isfile", lambda _path: False)
    assert _SQ._ensure_mojo_sq() is False


def test_ensure_mojo_false_on_load_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Mojo backend probe returns False when loading its library raises OSError."""
    monkeypatch.setattr(_SQ, "_mojo_sq_lib", None)
    monkeypatch.setattr(_SQ._os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(_SQ._ctypes, "CDLL", _raise_oserror)
    assert _SQ._ensure_mojo_sq() is False


def test_ensure_mojo_false_when_symbol_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Mojo backend probe returns False when the library lacks its symbols."""
    monkeypatch.setattr(_SQ, "_mojo_sq_lib", None)
    monkeypatch.setattr(_SQ._os.path, "isfile", lambda _path: True)
    monkeypatch.setattr(_SQ._ctypes, "CDLL", lambda _path: object())
    assert _SQ._ensure_mojo_sq() is False


def test_mojo_backend_raises_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the Mojo backend unavailable, isolation distance raises RuntimeError."""
    monkeypatch.setattr(_SQ, "_ensure_mojo_sq", lambda: False)
    cluster, noise = _cluster_noise(10, 20, 2)
    with pytest.raises(RuntimeError, match="Mojo sorting-quality backend is not available"):
        isolation_distance(cluster, noise, backend="mojo")
