# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (ownership_map_special) from former test_cosim_support_architecture.py

from __future__ import annotations

from tests.cosim_support_architecture_support import *  # noqa: F403


def test_ibarz_tanaka_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_IBARZ_TANAKA_NAMES)

    owner_tree = ast.parse(Path(cosim_reference_ibarz_tanaka.__file__).read_text(encoding="utf-8"))
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_IBARZ_TANAKA_NAMES)


def test_dpi_neuron_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_DPI_NEURON_NAMES)

    owner_tree = ast.parse(Path(cosim_reference_dpi_neuron.__file__).read_text(encoding="utf-8"))
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_DPI_NEURON_NAMES)


def test_exponential_relaxation_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_EXPONENTIAL_RELAXATION_NAMES)

    owner_tree = ast.parse(
        Path(cosim_reference_exponential_relaxation.__file__).read_text(encoding="utf-8")
    )
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_EXPONENTIAL_RELAXATION_NAMES)


def test_rulkov_map_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_RULKOV_MAP_NAMES)

    owner_tree = ast.parse(Path(cosim_reference_rulkov_map.__file__).read_text(encoding="utf-8"))
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_RULKOV_MAP_NAMES)
