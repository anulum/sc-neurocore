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

from tests import (
    cosim_reference_perfect_integrator,
    cosim_reference_statistics,
    cosim_runtime,
    cosim_support,
)

_RUNTIME_NAMES = (
    "HAS_IVERILOG",
    "_python_spike_count",
    "_verilog_spike_count",
    "simulate",
    "spike_count_method",
    "verilog_spike_count_method",
    "verilog_spike_count_method_pipelined",
)

_PERFECT_INTEGRATOR_NAMES = (
    "_perfect_integrator_hand_spike_count",
    "_perfect_integrator_sawtooth_features",
)


def test_legacy_surface_reexports_runtime_objects_without_wrappers() -> None:
    for name in _RUNTIME_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_runtime, name)

    for name in _PERFECT_INTEGRATOR_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_reference_perfect_integrator, name)
    assert cosim_support._summarise is cosim_reference_statistics._summarise


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
    assert (
        len(Path(cosim_reference_statistics.__file__).read_text(encoding="utf-8").splitlines())
        <= 55
    )
    perfect_text = Path(cosim_reference_perfect_integrator.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in perfect_text
    assert len(perfect_text.splitlines()) <= 65
    assert len(Path(cosim_support.__file__).read_text(encoding="utf-8").splitlines()) <= 2_160


def test_perfect_integrator_and_statistics_have_exact_definition_ownership() -> None:
    perfect_tree = ast.parse(
        Path(cosim_reference_perfect_integrator.__file__).read_text(encoding="utf-8")
    )
    perfect_functions = {
        node.name for node in perfect_tree.body if isinstance(node, ast.FunctionDef)
    }
    assert perfect_functions == set(_PERFECT_INTEGRATOR_NAMES)

    statistics_tree = ast.parse(
        Path(cosim_reference_statistics.__file__).read_text(encoding="utf-8")
    )
    statistics_functions = {
        node.name for node in statistics_tree.body if isinstance(node, ast.FunctionDef)
    }
    assert statistics_functions == {"_summarise"}
