# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo namespace availability contract tests

"""Regression tests for the public Mojo acceleration namespace."""

from __future__ import annotations

import builtins
import importlib
from collections.abc import Mapping, Sequence
from types import ModuleType

import pytest

import sc_neurocore.accel.mojo as mojo_namespace
from tests.module_reload import restore_module_namespace, snapshot_module_namespace


def test_mojo_namespace_keeps_runner_symbol_when_runner_import_fails() -> None:
    """Expose optional Mojo runner failures through the real package namespace."""
    real_import = builtins.__import__
    saved_namespace = snapshot_module_namespace(mojo_namespace)

    def blocked_runner_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name == "runner" and level == 1 and "MojoKernelRunner" in set(fromlist or ()):
            raise RuntimeError("controlled Mojo runner load failure")
        imported = real_import(name, globals, locals, fromlist, level)
        if not isinstance(imported, ModuleType):
            raise TypeError(f"expected module import result, got {type(imported)!r}")
        return imported

    try:
        builtins.__import__ = blocked_runner_import
        failed_namespace = importlib.reload(mojo_namespace)

        assert failed_namespace._HAS_MOJO is False
        assert failed_namespace._mojo_import_reason is not None
        assert "controlled Mojo runner load failure" in failed_namespace._mojo_import_reason
        assert failed_namespace.MOJO_HELPER_BACKEND == "unavailable"
        assert failed_namespace.MOJO_HELPER_IPC_AVAILABLE is False
        assert "MojoKernelRunner" in failed_namespace.__all__
        with pytest.raises(RuntimeError, match="Mojo runner unavailable"):
            failed_namespace.MojoKernelRunner()
    finally:
        builtins.__import__ = real_import
        restore_module_namespace(mojo_namespace, saved_namespace)
