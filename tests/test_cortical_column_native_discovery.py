# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNativeDiscovery from former test_cortical_column.py

"""Focused suite: TestNativeDiscovery from former test_cortical_column.py."""

from __future__ import annotations

from tests.cortical_column_support import *  # noqa: F403

class TestNativeDiscovery:
    def test_rust_discovery_uses_root_package_fallback(self, monkeypatch):
        real_import_module = cortical_column_module._importlib.import_module

        def root_only_engine(name):
            if name == "sc_neurocore_engine.sc_neurocore_engine":
                raise ImportError(name)
            if name == "sc_neurocore_engine":
                return SimpleNamespace(
                    py_parallel_csr_spmv_add=lambda *args: None,
                    py_parallel_csr_multi_spmv_add=lambda *args: None,
                )
            return real_import_module(name)

        monkeypatch.setattr(cortical_column_module._importlib, "import_module", root_only_engine)
        _saved_ns = snapshot_module_namespace(cortical_column_module)
        reloaded = importlib.reload(cortical_column_module)
        try:
            assert reloaded._HAS_RUST_CSR_SPMV is True
            assert reloaded._HAS_RUST_CSR_MULTI_SPMV is True
        finally:
            monkeypatch.undo()
            restore_module_namespace(cortical_column_module, _saved_ns)

    def test_rust_discovery_fails_closed_without_symbols(self, monkeypatch):
        real_import_module = cortical_column_module._importlib.import_module

        def missing_engine(name):
            if name in {"sc_neurocore_engine.sc_neurocore_engine", "sc_neurocore_engine"}:
                raise ImportError(name)
            return real_import_module(name)

        monkeypatch.setattr(cortical_column_module._importlib, "import_module", missing_engine)
        _saved_ns = snapshot_module_namespace(cortical_column_module)
        reloaded = importlib.reload(cortical_column_module)
        try:
            assert reloaded._HAS_RUST_CSR_SPMV is False
            assert reloaded._rust_csr_spmv_add is None
            assert reloaded._HAS_RUST_CSR_MULTI_SPMV is False
            assert reloaded._rust_csr_multi_spmv_add is None
        finally:
            monkeypatch.undo()
            restore_module_namespace(cortical_column_module, _saved_ns)

    def test_julia_discovery_failure_remains_optional(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "juliacall", None)
        _saved_ns = snapshot_module_namespace(cortical_column_module)
        reloaded = importlib.reload(cortical_column_module)
        try:
            assert reloaded._HAS_JULIA_MULTI_SPMV is False
            assert reloaded._julia_multi_spmv is None
        finally:
            monkeypatch.undo()
            restore_module_namespace(cortical_column_module, _saved_ns)

    def test_optional_ctypes_backend_load_failures_remain_optional(self, monkeypatch):
        def fake_exists(path):
            return path.endswith("libcortical_column.so")

        def reject_cdll(path):
            raise OSError(path)

        monkeypatch.setattr(cortical_column_module.os.path, "exists", fake_exists)
        monkeypatch.setattr(cortical_column_module.ctypes, "CDLL", reject_cdll)
        _saved_ns = snapshot_module_namespace(cortical_column_module)
        reloaded = importlib.reload(cortical_column_module)
        try:
            assert reloaded._HAS_GO_MULTI_SPMV is False
            assert reloaded._go_multi_spmv is None
            assert reloaded._HAS_MOJO_MULTI_SPMV is False
            assert reloaded._mojo_multi_spmv is None
        finally:
            monkeypatch.undo()
            restore_module_namespace(cortical_column_module, _saved_ns)

    def test_mojo_ctypes_discovery_configures_symbol(self, monkeypatch):
        class FakeFunction:
            argtypes = None
            restype = object()

        fake_function = FakeFunction()
        fake_lib = SimpleNamespace(py_parallel_csr_multi_spmv_add_c=fake_function)

        def fake_exists(path):
            return path.endswith("libcortical_column.so")

        monkeypatch.setattr(cortical_column_module.os.path, "exists", fake_exists)
        monkeypatch.setattr(cortical_column_module.ctypes, "CDLL", lambda _path: fake_lib)
        _saved_ns = snapshot_module_namespace(cortical_column_module)
        reloaded = importlib.reload(cortical_column_module)
        try:
            assert reloaded._HAS_MOJO_MULTI_SPMV is True
            assert fake_function.argtypes is not None
            assert fake_function.restype is None
        finally:
            monkeypatch.undo()
            restore_module_namespace(cortical_column_module, _saved_ns)
