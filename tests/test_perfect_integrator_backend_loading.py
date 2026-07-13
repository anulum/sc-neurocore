# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Perfect Integrator optional native-library loading contracts

"""Fail-closed loading tests for the optional Perfect Integrator runtimes."""

from __future__ import annotations

import ctypes
import importlib
import os

import pytest

from sc_neurocore.accel import perfect_integrator as backends


def test_missing_rust_engine_is_detected_at_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the optional-engine import boundary without leaving module drift."""
    real_import = importlib.import_module

    def without_engine(name: str, package: str | None = None) -> object:
        if name == "sc_neurocore_engine":
            raise ImportError("engine intentionally hidden")
        return real_import(name, package)

    with monkeypatch.context() as patch:
        patch.setattr(importlib, "import_module", without_engine)
        reloaded = importlib.reload(backends)
        assert reloaded._HAS_RUST is False
        assert reloaded._EnginePerfectIntegratorCls is None
    importlib.reload(backends)
    assert backends._HAS_RUST is True


@pytest.mark.parametrize("backend", ("go", "mojo"))
@pytest.mark.parametrize("failure", ("missing", "load", "symbol"))
def test_c_backend_loader_rejects_invalid_library_boundaries(
    backend: str,
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep absent, unloadable, or symbol-incomplete libraries unavailable."""
    monkeypatch.setattr(backends, f"_{backend}_lib", None)
    monkeypatch.setattr(backends, f"_HAS_{backend.upper()}", False)
    monkeypatch.setattr(os.path, "isfile", lambda _path: failure != "missing")
    if failure == "load":

        def reject_load(_path: str) -> object:
            raise OSError("invalid shared library")

        monkeypatch.setattr(ctypes, "CDLL", reject_load)
    elif failure == "symbol":
        monkeypatch.setattr(ctypes, "CDLL", lambda _path: object())
    assert getattr(backends, f"ensure_{backend}_loaded")() is False
    assert getattr(backends, f"_{backend}_lib") is None
    assert getattr(backends, f"_HAS_{backend.upper()}") is False


@pytest.mark.parametrize("failure", ("missing", "source", "module"))
def test_julia_loader_rejects_invalid_runtime_boundaries(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep missing runtimes, source files, and broken modules unavailable."""
    monkeypatch.setattr(backends, "_julia_module", None)
    monkeypatch.setattr(backends, "_HAS_JULIA", False)
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda _name: None if failure == "missing" else 1
    )
    monkeypatch.setattr(os.path, "isfile", lambda _path: failure != "source")
    if failure == "module":

        def reject_import(_name: str) -> object:
            raise RuntimeError("broken Julia runtime")

        monkeypatch.setattr(importlib, "import_module", reject_import)
    assert backends.ensure_julia_loaded() is False
    assert backends._julia_module is None
    assert backends._HAS_JULIA is False
