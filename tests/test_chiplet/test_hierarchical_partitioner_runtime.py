# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hierarchical partitioner runtime probe tests

"""Failure-path contracts for optional backend discovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


class TestProbeFailureBranches:
    """Force every lazy-load probe through its failure branches via
    monkeypatch — these are the user's signal that a backend is
    misconfigured at runtime, so they must be tested even though
    they need cooperative mocking to reach."""

    def _reset_probes(self, hp_mod: Any) -> None:
        hp_mod._julia_kl_refine = None
        hp_mod._HAS_JULIA_KL_REFINE = False
        hp_mod._go_kl_refine_lib = None
        hp_mod._HAS_GO_KL_REFINE = False
        hp_mod._mojo_kl_refine_lib = None
        hp_mod._HAS_MOJO_KL_REFINE = False

    def test_julia_probe_returns_false_when_juliacall_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from sc_neurocore.chiplet import hierarchical_backend_runtime as hp_mod

        saved_jl = hp_mod._julia_kl_refine
        saved_has = hp_mod._HAS_JULIA_KL_REFINE
        try:
            self._reset_probes(hp_mod)
            # Make `from juliacall import Main` fail.
            import builtins

            real_import = builtins.__import__

            def fail_import(name: str, *args: Any, **kwargs: Any) -> Any:
                if name == "juliacall":
                    raise ImportError("simulated missing juliacall")
                return real_import(name, *args, **kwargs)

            monkeypatch.setattr(builtins, "__import__", fail_import)
            assert hp_mod._ensure_julia_kl_refine_loaded() is False
        finally:
            hp_mod._julia_kl_refine = saved_jl
            hp_mod._HAS_JULIA_KL_REFINE = saved_has

    def test_julia_probe_returns_false_when_jl_file_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from sc_neurocore.chiplet import hierarchical_backend_runtime as hp_mod

        saved_jl = hp_mod._julia_kl_refine
        saved_has = hp_mod._HAS_JULIA_KL_REFINE
        try:
            self._reset_probes(hp_mod)
            monkeypatch.setattr(
                hp_mod,
                "_accel_path",
                lambda *parts: tmp_path / "missing-kl-refine.jl",
            )
            assert hp_mod._ensure_julia_kl_refine_loaded() is False
        finally:
            hp_mod._julia_kl_refine = saved_jl
            hp_mod._HAS_JULIA_KL_REFINE = saved_has

    def test_julia_probe_returns_false_when_include_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Cover the `except Exception: return False` branch in
        `_ensure_julia_kl_refine_loaded` by feeding a syntactically
        broken .jl file into the include path. The probe catches
        the parser error and returns False.
        """
        from sc_neurocore.chiplet import hierarchical_backend_runtime as hp_mod

        try:
            import juliacall  # noqa: F401
        except ImportError:
            pytest.skip("juliacall not installed")

        saved_jl = hp_mod._julia_kl_refine
        saved_has = hp_mod._HAS_JULIA_KL_REFINE
        try:
            self._reset_probes(hp_mod)
            broken_jl = tmp_path / "broken.jl"
            broken_jl.write_text("THIS IS NOT VALID JULIA\n")
            monkeypatch.setattr(hp_mod, "_accel_path", lambda *parts: broken_jl)
            assert hp_mod._ensure_julia_kl_refine_loaded() is False
        finally:
            hp_mod._julia_kl_refine = saved_jl
            hp_mod._HAS_JULIA_KL_REFINE = saved_has

    def test_go_probe_returns_false_when_so_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from sc_neurocore.chiplet import hierarchical_backend_runtime as hp_mod

        saved_lib = hp_mod._go_kl_refine_lib
        saved_has = hp_mod._HAS_GO_KL_REFINE
        try:
            self._reset_probes(hp_mod)
            monkeypatch.setattr(
                hp_mod,
                "_accel_path",
                lambda *parts: tmp_path / "missing-go-partition.so",
            )
            assert hp_mod._ensure_go_kl_refine_loaded() is False
        finally:
            hp_mod._go_kl_refine_lib = saved_lib
            hp_mod._HAS_GO_KL_REFINE = saved_has

    def test_go_probe_returns_false_when_cdll_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from sc_neurocore.chiplet import hierarchical_backend_runtime as hp_mod

        saved_lib = hp_mod._go_kl_refine_lib
        saved_has = hp_mod._HAS_GO_KL_REFINE
        try:
            self._reset_probes(hp_mod)
            import ctypes

            real_cdll = ctypes.CDLL

            class FailingCDLL:
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    raise OSError("simulated CDLL failure")

            monkeypatch.setattr(ctypes, "CDLL", FailingCDLL)
            assert hp_mod._ensure_go_kl_refine_loaded() is False
            monkeypatch.setattr(ctypes, "CDLL", real_cdll)
        finally:
            hp_mod._go_kl_refine_lib = saved_lib
            hp_mod._HAS_GO_KL_REFINE = saved_has

    def test_go_probe_returns_false_when_symbol_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from sc_neurocore.chiplet import hierarchical_backend_runtime as hp_mod

        saved_lib = hp_mod._go_kl_refine_lib
        saved_has = hp_mod._HAS_GO_KL_REFINE
        try:
            self._reset_probes(hp_mod)
            import ctypes

            class EmptyLib:
                pass

            real_cdll = ctypes.CDLL
            monkeypatch.setattr(ctypes, "CDLL", lambda p: EmptyLib())
            assert hp_mod._ensure_go_kl_refine_loaded() is False
            monkeypatch.setattr(ctypes, "CDLL", real_cdll)
        finally:
            hp_mod._go_kl_refine_lib = saved_lib
            hp_mod._HAS_GO_KL_REFINE = saved_has

    def test_mojo_probe_returns_false_when_so_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        from sc_neurocore.chiplet import hierarchical_backend_runtime as hp_mod

        saved_lib = hp_mod._mojo_kl_refine_lib
        saved_has = hp_mod._HAS_MOJO_KL_REFINE
        try:
            self._reset_probes(hp_mod)
            monkeypatch.setattr(
                hp_mod,
                "_accel_path",
                lambda *parts: tmp_path / "missing-mojo-partition.so",
            )
            assert hp_mod._ensure_mojo_kl_refine_loaded() is False
        finally:
            hp_mod._mojo_kl_refine_lib = saved_lib
            hp_mod._HAS_MOJO_KL_REFINE = saved_has

    def test_mojo_probe_returns_false_when_cdll_raises(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from sc_neurocore.chiplet import hierarchical_backend_runtime as hp_mod

        saved_lib = hp_mod._mojo_kl_refine_lib
        saved_has = hp_mod._HAS_MOJO_KL_REFINE
        try:
            self._reset_probes(hp_mod)
            import ctypes

            class FailingCDLL:
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    raise OSError("simulated mojo CDLL failure")

            real_cdll = ctypes.CDLL
            monkeypatch.setattr(ctypes, "CDLL", FailingCDLL)
            assert hp_mod._ensure_mojo_kl_refine_loaded() is False
            monkeypatch.setattr(ctypes, "CDLL", real_cdll)
        finally:
            hp_mod._mojo_kl_refine_lib = saved_lib
            hp_mod._HAS_MOJO_KL_REFINE = saved_has

    def test_mojo_probe_returns_false_when_symbol_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from sc_neurocore.chiplet import hierarchical_backend_runtime as hp_mod

        saved_lib = hp_mod._mojo_kl_refine_lib
        saved_has = hp_mod._HAS_MOJO_KL_REFINE
        try:
            self._reset_probes(hp_mod)
            import ctypes

            class EmptyLib:
                pass

            real_cdll = ctypes.CDLL
            monkeypatch.setattr(ctypes, "CDLL", lambda p: EmptyLib())
            assert hp_mod._ensure_mojo_kl_refine_loaded() is False
            monkeypatch.setattr(ctypes, "CDLL", real_cdll)
        finally:
            hp_mod._mojo_kl_refine_lib = saved_lib
            hp_mod._HAS_MOJO_KL_REFINE = saved_has
