# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (fail_closed_loaders) from former test_chialvo_map_backends.py

from __future__ import annotations

from tests.chialvo_map_backends_support import *  # noqa: F403


def test_explicit_unavailable_backend_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit lane request must never fall through to Python silently."""
    monkeypatch.setattr(chialvo_map, "_backend_available", lambda _name: False)
    for backend in ("rust", "julia", "go", "mojo"):
        with pytest.raises(RuntimeError, match="unavailable"):
            ChialvoMapNeuron().simulate(1, backend=backend)


def test_import_without_rust_extension_keeps_python_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing optional Rust extension must not prevent Python use."""
    real_import = importlib.import_module

    def import_without_engine(name: str, package: str | None = None) -> object:
        if name == "sc_neurocore_engine":
            raise ImportError("extension absent")
        return real_import(name, package)

    with monkeypatch.context() as context:
        context.setattr(importlib, "import_module", import_without_engine)
        reloaded = importlib.reload(chialvo_map)
        trace, spikes = reloaded.ChialvoMapNeuron().simulate(2, backend="python")
        assert trace.shape == (2,)
        assert spikes in (0, 1, 2)
        with pytest.raises(RuntimeError, match="unavailable"):
            reloaded.ChialvoMapNeuron().simulate(1, backend="rust")
    importlib.reload(chialvo_map)


def test_julia_loader_reports_missing_runtime_and_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Julia selection must fail closed when either required surface is absent."""
    with monkeypatch.context() as context:
        context.setattr(chialvo_map, "_julia_module", None)
        context.setattr(importlib.util, "find_spec", lambda _name: None)
        with pytest.raises(RuntimeError, match="unavailable"):
            ChialvoMapNeuron().simulate(1, backend="julia")

    with monkeypatch.context() as context:
        context.setattr(chialvo_map, "_julia_module", None)
        context.setattr(importlib.util, "find_spec", lambda _name: object())
        context.setattr(os.path, "isfile", lambda _path: False)
        with pytest.raises(RuntimeError, match="unavailable"):
            ChialvoMapNeuron().simulate(1, backend="julia")


def test_c_loader_reports_missing_file_load_failure_and_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Go selection must reject each shared-library discovery failure."""
    with monkeypatch.context() as context:
        context.setattr(chialvo_map, "_go_lib", None)
        context.setattr(os.path, "isfile", lambda _path: False)
        with pytest.raises(RuntimeError, match="Go Chialvo backend unavailable"):
            ChialvoMapNeuron().simulate(1, backend="go")

    def reject_library(_path: str) -> ctypes.CDLL:
        raise OSError("invalid shared object")

    with monkeypatch.context() as context:
        context.setattr(chialvo_map, "_go_lib", None)
        context.setattr(os.path, "isfile", lambda _path: True)
        context.setattr(ctypes, "CDLL", reject_library)
        with pytest.raises(RuntimeError, match="Go Chialvo backend unavailable"):
            ChialvoMapNeuron().simulate(1, backend="go")

    with monkeypatch.context() as context:
        context.setattr(chialvo_map, "_go_lib", None)
        context.setattr(os.path, "isfile", lambda _path: True)
        context.setattr(ctypes, "CDLL", lambda _path: object())
        with pytest.raises(RuntimeError, match="Go Chialvo backend unavailable"):
            ChialvoMapNeuron().simulate(1, backend="go")


def test_compiled_error_sentinel_becomes_floating_point_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """C ABI numerical rejection must not expose an uninitialised trace."""

    class RejectingFunction:
        def __call__(self, *_args: object) -> int:
            return -1

    class RejectingLibrary:
        chialvo_map_simulate_c = RejectingFunction()

    monkeypatch.setattr(chialvo_map, "_backend_available", lambda _name: True)
    monkeypatch.setattr(chialvo_map, "_mojo_lib", RejectingLibrary())
    with pytest.raises(FloatingPointError, match="rejected"):
        ChialvoMapNeuron().simulate(1, backend="mojo")
