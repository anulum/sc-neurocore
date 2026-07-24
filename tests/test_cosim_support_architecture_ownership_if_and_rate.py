# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (ownership_if_and_rate) from former test_cosim_support_architecture.py

from __future__ import annotations

from tests.cosim_support_architecture_support import *  # noqa: F403


def test_adex_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_ADEX_NAMES)

    owner_tree = ast.parse(Path(cosim_reference_adex.__file__).read_text(encoding="utf-8"))
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_ADEX_NAMES)


def test_conductance_rates_have_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_CONDUCTANCE_RATE_NAMES)

    owner_tree = ast.parse(
        Path(cosim_reference_conductance_rates.__file__).read_text(encoding="utf-8")
    )
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_CONDUCTANCE_RATE_NAMES)


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


def test_glif_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_GLIF_NAMES)

    glif_tree = ast.parse(Path(cosim_reference_glif.__file__).read_text(encoding="utf-8"))
    glif_functions = {node.name for node in glif_tree.body if isinstance(node, ast.FunctionDef)}
    assert glif_functions == set(_GLIF_NAMES)


def test_exp_if_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_EXP_IF_NAMES)

    owner_tree = ast.parse(Path(cosim_reference_exp_if.__file__).read_text(encoding="utf-8"))
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_EXP_IF_NAMES)


def test_izhikevich2007_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_IZHIKEVICH2007_NAMES)

    owner_tree = ast.parse(
        Path(cosim_reference_izhikevich2007.__file__).read_text(encoding="utf-8")
    )
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_IZHIKEVICH2007_NAMES)


def test_izhikevich_rs_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_IZHIKEVICH_RS_NAMES)

    owner_tree = ast.parse(Path(cosim_reference_izhikevich_rs.__file__).read_text(encoding="utf-8"))
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_IZHIKEVICH_RS_NAMES)


def test_lif_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_LIF_NAMES)

    owner_tree = ast.parse(Path(cosim_reference_lif.__file__).read_text(encoding="utf-8"))
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_LIF_NAMES)


def test_quadratic_if_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_QUADRATIC_IF_NAMES)

    quadratic_if_tree = ast.parse(
        Path(cosim_reference_quadratic_if.__file__).read_text(encoding="utf-8")
    )
    quadratic_if_functions = {
        node.name for node in quadratic_if_tree.body if isinstance(node, ast.FunctionDef)
    }
    assert quadratic_if_functions == set(_QUADRATIC_IF_NAMES)


def test_theta_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_THETA_NAMES)

    theta_tree = ast.parse(Path(cosim_reference_theta.__file__).read_text(encoding="utf-8"))
    theta_functions = {node.name for node in theta_tree.body if isinstance(node, ast.FunctionDef)}
    assert theta_functions == set(_THETA_NAMES)
