# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (loaders) from former test_topology_backends.py

from __future__ import annotations

from tests.topology_backends_support import *  # noqa: F403


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
