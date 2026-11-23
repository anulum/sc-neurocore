# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (ownership_if_and_rate) from former test_cosim_support_architecture.py

"""Pin exact definition ownership for the integrate-and-fire and rate references."""

from __future__ import annotations

import ast
from pathlib import Path

from tests import (
    cosim_reference_adex,
    cosim_reference_conductance_rates,
    cosim_reference_exp_if,
    cosim_reference_glif,
    cosim_reference_izhikevich2007,
    cosim_reference_izhikevich_rs,
    cosim_reference_lif,
    cosim_reference_perfect_integrator,
    cosim_reference_quadratic_if,
    cosim_reference_statistics,
    cosim_reference_theta,
    cosim_support,
)
from tests.cosim_support_architecture_support import (
    _ADEX_NAMES,
    _CONDUCTANCE_RATE_NAMES,
    _EXP_IF_NAMES,
    _GLIF_NAMES,
    _IZHIKEVICH2007_NAMES,
    _IZHIKEVICH_RS_NAMES,
    _LIF_NAMES,
    _PERFECT_INTEGRATOR_NAMES,
    _QUADRATIC_IF_NAMES,
    _THETA_NAMES,
)


def test_adex_has_exact_definition_ownership() -> None:
    """The AdEx reference defines exactly its functions; the facade none."""
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_ADEX_NAMES)

    owner_tree = ast.parse(Path(cosim_reference_adex.__file__).read_text(encoding="utf-8"))
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_ADEX_NAMES)


def test_conductance_rates_have_exact_definition_ownership() -> None:
    """The conductance-rate reference defines exactly its functions; the facade none."""
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_CONDUCTANCE_RATE_NAMES)

    owner_tree = ast.parse(
        Path(cosim_reference_conductance_rates.__file__).read_text(encoding="utf-8")
    )
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_CONDUCTANCE_RATE_NAMES)


def test_perfect_integrator_and_statistics_have_exact_definition_ownership() -> None:
    """The perfect-integrator and statistics references define exactly their functions."""
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
    """The GLIF reference defines exactly its functions; the facade none."""
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_GLIF_NAMES)

    glif_tree = ast.parse(Path(cosim_reference_glif.__file__).read_text(encoding="utf-8"))
    glif_functions = {node.name for node in glif_tree.body if isinstance(node, ast.FunctionDef)}
    assert glif_functions == set(_GLIF_NAMES)


def test_exp_if_has_exact_definition_ownership() -> None:
    """The exponential IF reference defines exactly its functions; the facade none."""
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_EXP_IF_NAMES)

    owner_tree = ast.parse(Path(cosim_reference_exp_if.__file__).read_text(encoding="utf-8"))
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_EXP_IF_NAMES)


def test_izhikevich2007_has_exact_definition_ownership() -> None:
    """The Izhikevich 2007 reference defines exactly its functions; the facade none."""
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_IZHIKEVICH2007_NAMES)

    owner_tree = ast.parse(
        Path(cosim_reference_izhikevich2007.__file__).read_text(encoding="utf-8")
    )
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_IZHIKEVICH2007_NAMES)


def test_izhikevich_rs_has_exact_definition_ownership() -> None:
    """The Izhikevich RS reference defines exactly its functions; the facade none."""
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_IZHIKEVICH_RS_NAMES)

    owner_tree = ast.parse(Path(cosim_reference_izhikevich_rs.__file__).read_text(encoding="utf-8"))
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_IZHIKEVICH_RS_NAMES)


def test_lif_has_exact_definition_ownership() -> None:
    """The LIF reference defines exactly its functions; the facade none."""
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_LIF_NAMES)

    owner_tree = ast.parse(Path(cosim_reference_lif.__file__).read_text(encoding="utf-8"))
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_LIF_NAMES)


def test_quadratic_if_has_exact_definition_ownership() -> None:
    """The quadratic IF reference defines exactly its functions; the facade none."""
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
    """The theta-neuron reference defines exactly its functions; the facade none."""
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_THETA_NAMES)

    theta_tree = ast.parse(Path(cosim_reference_theta.__file__).read_text(encoding="utf-8"))
    theta_functions = {node.name for node in theta_tree.body if isinstance(node, ast.FunctionDef)}
    assert theta_functions == set(_THETA_NAMES)
