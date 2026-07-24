# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (ownership_biophysical) from former test_cosim_support_architecture.py

from __future__ import annotations

from tests.cosim_support_architecture_support import *  # noqa: F403


def test_connor_stevens_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_CONNOR_STEVENS_NAMES)

    owner_tree = ast.parse(
        Path(cosim_reference_connor_stevens.__file__).read_text(encoding="utf-8")
    )
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_CONNOR_STEVENS_NAMES)


def test_pernarowski_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_PERNAROWSKI_NAMES)

    owner_tree = ast.parse(Path(cosim_reference_pernarowski.__file__).read_text(encoding="utf-8"))
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_PERNAROWSKI_NAMES)


def test_hindmarsh_rose_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_HINDMARSH_ROSE_NAMES)

    owner_tree = ast.parse(
        Path(cosim_reference_hindmarsh_rose.__file__).read_text(encoding="utf-8")
    )
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_HINDMARSH_ROSE_NAMES)


def test_hodgkin_huxley_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_HODGKIN_HUXLEY_NAMES)

    owner_tree = ast.parse(
        Path(cosim_reference_hodgkin_huxley.__file__).read_text(encoding="utf-8")
    )
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_HODGKIN_HUXLEY_NAMES)


def test_fitzhugh_nagumo_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_FITZHUGH_NAGUMO_NAMES)

    owner_tree = ast.parse(
        Path(cosim_reference_fitzhugh_nagumo.__file__).read_text(encoding="utf-8")
    )
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_FITZHUGH_NAGUMO_NAMES)


def test_fitzhugh_rinzel_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_FITZHUGH_RINZEL_NAMES)

    owner_tree = ast.parse(
        Path(cosim_reference_fitzhugh_rinzel.__file__).read_text(encoding="utf-8")
    )
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_FITZHUGH_RINZEL_NAMES)


def test_mckean_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_MCKEAN_NAMES)
    facade_assignments = {
        target.id
        for node in facade_tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "_MCKEAN_PARAMS" not in facade_assignments

    owner_tree = ast.parse(Path(cosim_reference_mckean.__file__).read_text(encoding="utf-8"))
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_MCKEAN_NAMES)
    owner_assignments = {
        target.id
        for node in owner_tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert owner_assignments == {"_MCKEAN_PARAMS"}


def test_mihalas_niebur_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_MIHALAS_NIEBUR_NAMES)
    facade_assignments = {
        target.id
        for node in facade_tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert "_MIHALAS_NIEBUR_PARAMS" not in facade_assignments

    owner_tree = ast.parse(
        Path(cosim_reference_mihalas_niebur.__file__).read_text(encoding="utf-8")
    )
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_MIHALAS_NIEBUR_NAMES)
    owner_assignments = {
        target.id
        for node in owner_tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert owner_assignments == {"_MIHALAS_NIEBUR_PARAMS"}


def test_morris_lecar_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_MORRIS_LECAR_NAMES)

    owner_tree = ast.parse(Path(cosim_reference_morris_lecar.__file__).read_text(encoding="utf-8"))
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_MORRIS_LECAR_NAMES)


def test_terman_wang_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_TERMAN_WANG_NAMES)

    owner_tree = ast.parse(Path(cosim_reference_terman_wang.__file__).read_text(encoding="utf-8"))
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_TERMAN_WANG_NAMES)


def test_wang_buzsaki_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_WANG_BUZSAKI_NAMES)

    owner_tree = ast.parse(Path(cosim_reference_wang_buzsaki.__file__).read_text(encoding="utf-8"))
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_WANG_BUZSAKI_NAMES)


def test_wilson_hr_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_WILSON_HR_NAMES)

    owner_tree = ast.parse(Path(cosim_reference_wilson_hr.__file__).read_text(encoding="utf-8"))
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_WILSON_HR_NAMES)
