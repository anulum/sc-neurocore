# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Cortical-column modular architecture contracts

"""Architecture contracts for the cortical-column responsibility split."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from sc_neurocore import network
from sc_neurocore.network import _cortical_column_parameters as parameters
from sc_neurocore.network import cortical_column
from sc_neurocore.network.cortical_column import CorticalColumn

_REPO_ROOT = Path(__file__).resolve().parents[1]
_NETWORK_ROOT = _REPO_ROOT / "src" / "sc_neurocore" / "network"
_MODULE_PATHS = {
    "public": _NETWORK_ROOT / "cortical_column.py",
    "backends": _NETWORK_ROOT / "_cortical_column_backends.py",
    "connectivity": _NETWORK_ROOT / "_cortical_column_connectivity.py",
    "parameters": _NETWORK_ROOT / "_cortical_column_parameters.py",
}


def _relative_imports(path: Path) -> set[str]:
    """Return relative module imports declared by one Python source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level > 0 and node.module is not None
    }


def test_historical_public_identity_and_signature_remain_stable() -> None:
    """The refactor keeps the established constructor at its historical path."""
    assert network.CorticalColumn is CorticalColumn
    assert CorticalColumn.__module__ == "sc_neurocore.network.cortical_column"
    assert CorticalColumn.__qualname__ == "CorticalColumn"
    assert str(inspect.signature(CorticalColumn)) == (
        "(scale: 'float' = 0.1, bg_rate: 'float' = 8.0, g_inh: 'float' = 4.0, "
        "scale_correction: 'bool' = True, delay_distribution: 'bool' = True, "
        "n_delay_bins: 'int' = 5, use_block_csr: 'bool' = False, "
        "seed: 'int | None' = None, backend: 'str' = 'auto') -> 'None'"
    )


def test_historical_parameter_exports_are_direct_reexports() -> None:
    """Published constants retain identity at the historical public module."""
    assert cortical_column.POPULATIONS is parameters.POPULATIONS
    assert cortical_column.FULL_SIZES is parameters.FULL_SIZES
    assert cortical_column.K_BG is parameters.K_BG
    assert cortical_column.CONN_PROBS is parameters.CONN_PROBS


def test_internal_dependency_direction_is_acyclic() -> None:
    """Backend and parameter leaves stay independent of the public facade."""
    assert _relative_imports(_MODULE_PATHS["backends"]) == set()
    assert _relative_imports(_MODULE_PATHS["parameters"]) == set()
    assert _relative_imports(_MODULE_PATHS["connectivity"]) == {"_cortical_column_parameters"}
    assert _relative_imports(_MODULE_PATHS["public"]) == {
        "_cortical_column_backends",
        "_cortical_column_connectivity",
        "_cortical_column_parameters",
    }


def test_responsibility_modules_remain_bounded() -> None:
    """Prevent the cortical-column implementation from regrowing a GodFile."""
    limits = {
        "public": 800,
        "backends": 220,
        "connectivity": 450,
        "parameters": 180,
    }
    for name, path in _MODULE_PATHS.items():
        lines = len(path.read_text(encoding="utf-8").splitlines())
        assert lines <= limits[name], f"{name} module grew to {lines} lines"
