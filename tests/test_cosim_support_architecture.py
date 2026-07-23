# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Co-simulation support architecture contracts

"""Pin generic simulator ownership outside the legacy model-reference surface."""

from __future__ import annotations

import ast
from pathlib import Path

from tests import cosim_runtime, cosim_support

_RUNTIME_NAMES = (
    "HAS_IVERILOG",
    "_python_spike_count",
    "_verilog_spike_count",
    "simulate",
    "spike_count_method",
    "verilog_spike_count_method",
    "verilog_spike_count_method_pipelined",
)


def test_legacy_surface_reexports_runtime_objects_without_wrappers() -> None:
    for name in _RUNTIME_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_runtime, name)


def test_runtime_functions_have_one_definition_owner() -> None:
    facade_path = Path(cosim_support.__file__)
    facade_tree = ast.parse(facade_path.read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_RUNTIME_NAMES)

    runtime_path = Path(cosim_runtime.__file__)
    runtime_tree = ast.parse(runtime_path.read_text(encoding="utf-8"))
    runtime_functions = {
        node.name for node in runtime_tree.body if isinstance(node, ast.FunctionDef)
    }
    assert runtime_functions == set(_RUNTIME_NAMES) - {"HAS_IVERILOG"}


def test_runtime_dependency_is_one_way_and_surfaces_cannot_regrow() -> None:
    runtime_text = Path(cosim_runtime.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in runtime_text
    assert len(runtime_text.splitlines()) <= 180
    assert len(Path(cosim_support.__file__).read_text(encoding="utf-8").splitlines()) <= 2_230
