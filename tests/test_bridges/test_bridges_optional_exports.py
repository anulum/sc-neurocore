# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Package-boundary contracts for optional bridge exports."""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
from pathlib import Path
import subprocess
import sys
from collections.abc import Sequence
from types import ModuleType

import pytest

import sc_neurocore.bridges as bridges
from tests.module_reload import restore_module_namespace, snapshot_module_namespace

_REPO_ROOT = Path(__file__).resolve().parents[2]


class _BlockingFinder(importlib.abc.MetaPathFinder):
    """Raise a named module-not-found error for one exact import."""

    def __init__(self, blocked_name: str) -> None:
        self._blocked_name = blocked_name

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        """Block the configured import and ignore every other module."""
        if fullname == self._blocked_name:
            raise ModuleNotFoundError(f"blocked {fullname}", name=fullname)
        return None


def _reload_without_modules(
    blocked_name: str,
    removed_names: tuple[str, ...],
) -> ModuleType:
    """Reload the bridge package with selected import-table entries absent."""
    finder = _BlockingFinder(blocked_name)
    sys.meta_path.insert(0, finder)
    for name in removed_names:
        sys.modules.pop(name, None)
    try:
        return importlib.reload(bridges)
    finally:
        sys.meta_path.remove(finder)


def test_dna_bridge_import_survives_source_only_optics_absence() -> None:
    """The base wheel must not require the intentionally excluded optics tree."""
    program = """
import importlib.abc
import sys

class BlockOptics(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "sc_neurocore.optics":
            raise ModuleNotFoundError("blocked source-only optics", name=fullname)
        return None

sys.meta_path.insert(0, BlockOptics())
import sc_neurocore.bridges as bridges
from sc_neurocore.bridges.dna_mapper import BitstreamToDNA

assert bridges.BitstreamToDNA is BitstreamToDNA
assert "BitstreamToDNA" in bridges.__all__
assert "PhotonicCoDesignConfig" not in bridges.__all__
assert BitstreamToDNA(seed=42).compile_network([], [], []).total_gates == 0
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr


def test_bridge_reload_omits_codesign_when_optics_package_is_absent() -> None:
    """The in-process export registry follows the clean-wheel boundary."""
    module_names = (
        "sc_neurocore.bridges.photonic_codesign",
        "sc_neurocore.optics",
        "sc_neurocore.optics.photonic_emitter",
    )
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    saved_namespace = snapshot_module_namespace(bridges)

    try:
        reloaded = _reload_without_modules("sc_neurocore.optics", module_names)

        assert "BitstreamToDNA" in reloaded.__all__
        assert "PhotonicCoDesignConfig" not in reloaded.__all__
    finally:
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module
        restore_module_namespace(bridges, saved_namespace)


def test_bridge_reload_does_not_hide_unrelated_missing_dependencies() -> None:
    """Only the declared source-only optics boundary may be suppressed."""
    module_names = (
        "sc_neurocore.bridges.photonic_codesign",
        "sc_neurocore.edge.bitstream",
    )
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    saved_namespace = snapshot_module_namespace(bridges)

    try:
        with pytest.raises(ModuleNotFoundError, match="blocked sc_neurocore.edge.bitstream"):
            _reload_without_modules("sc_neurocore.edge.bitstream", module_names)
    finally:
        for name, module in saved_modules.items():
            if module is not None:
                sys.modules[name] = module
        restore_module_namespace(bridges, saved_namespace)
