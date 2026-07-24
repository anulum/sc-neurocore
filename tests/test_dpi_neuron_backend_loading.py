# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DPI optional native-library loading contracts

"""Fail-closed loading tests for the optional DPI runtimes."""

from __future__ import annotations

import ctypes
import importlib
import os
from pathlib import Path
import shutil
import site
import subprocess
import sys
import sysconfig

import pytest

from sc_neurocore.accel import dpi_neuron as backends


def test_missing_rust_engine_is_detected_at_import(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the optional-engine import boundary without leaving module drift."""
    real_import = importlib.import_module

    def without_engine(name: str, package: str | None = None) -> object:
        if name == "sc_neurocore_engine":
            raise ImportError("engine intentionally hidden")
        return real_import(name, package)

    with monkeypatch.context() as context:
        context.setattr(importlib, "import_module", without_engine)
        reloaded = importlib.reload(backends)
        assert reloaded._HAS_RUST is False
        assert reloaded._EngineDPICls is None
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


@pytest.mark.parametrize("failure", ("missing", "source", "import", "attribute"))
def test_julia_loader_rejects_invalid_runtime_boundaries(
    failure: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep missing runtimes, source files, and broken modules unavailable."""
    monkeypatch.setattr(backends, "_julia_module", None)
    monkeypatch.setattr(backends, "_HAS_JULIA", False)
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda _name: None if failure == "missing" else 1,
    )
    monkeypatch.setattr(os.path, "isfile", lambda _path: failure != "source")
    if failure == "import":

        def reject_import(_name: str) -> object:
            raise RuntimeError("broken Julia runtime")

        monkeypatch.setattr(importlib, "import_module", reject_import)
    elif failure == "attribute":
        monkeypatch.setattr(importlib, "import_module", lambda _name: object())
    assert backends.ensure_julia_loaded() is False
    assert backends._julia_module is None
    assert backends._HAS_JULIA is False


@pytest.mark.parametrize("backend", ("julia", "go", "mojo"))
def test_loaded_runtime_is_reused_without_reprobing(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid filesystem or runtime work once a handle has been established."""
    marker = object()
    monkeypatch.setattr(
        backends, f"_{backend}_module" if backend == "julia" else f"_{backend}_lib", marker
    )
    monkeypatch.setattr(os.path, "isfile", lambda _path: pytest.fail("unexpected probe"))
    assert getattr(backends, f"ensure_{backend}_loaded")() is True


def test_engine_loader_returns_the_extension_class() -> None:
    """Bind the import helper to the real DPINeuron engine symbol."""
    assert backends._load_engine_dpi().__name__ == "DPINeuron"


def test_checkout_bridge_discovers_maturin_compiled_extensions(tmp_path: Path) -> None:
    """Load each compiled extension when a source checkout shadows its package."""
    repository = Path(__file__).resolve().parents[1]
    source_package = repository / "bridge" / "sc_neurocore_engine"
    checkout_package = tmp_path / "sc_neurocore_engine"
    shutil.copytree(
        source_package,
        checkout_package,
        ignore=shutil.ignore_patterns("*.so", "*.pyd", "__pycache__"),
    )

    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    assert isinstance(extension_suffix, str) and extension_suffix
    purelib = Path(sysconfig.get_path("purelib"))
    compiled_extensions = sorted(
        {
            *source_package.glob(f"sc_neurocore_engine*{extension_suffix}"),
            *(purelib / "sc_neurocore_engine").glob(f"sc_neurocore_engine*{extension_suffix}"),
        }
    )
    assert compiled_extensions, "maturin-compiled engine extension is required"

    script = (
        "import importlib, pathlib, sys; "
        "package = importlib.import_module('sc_neurocore_engine'); "
        "extension = importlib.import_module('sc_neurocore_engine.sc_neurocore_engine'); "
        "expected = pathlib.Path(sys.argv[1]).resolve(); "
        "actual = pathlib.Path(extension.__file__).resolve(); "
        "assert actual == expected, (actual, expected); "
        "assert package.DPINeuron is extension.DPINeuron"
    )
    # ``python -S`` disables the site module (so purelib / usersite are not
    # auto-appended). Dependencies such as NumPy often live only in the user
    # site on developer workstations; purelib alone is insufficient.
    dependency_roots = [str(purelib)]
    user_site = site.getusersitepackages()
    if user_site and user_site not in dependency_roots:
        dependency_roots.append(user_site)

    for extension in compiled_extensions:
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(tmp_path), str(extension.parent.parent), *dependency_roots)
        )
        completed = subprocess.run(
            [sys.executable, "-S", "-c", script, str(extension)],
            cwd=tmp_path,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert completed.returncode == 0, (
            f"checkout bridge failed for {extension}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
