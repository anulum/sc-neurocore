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
    cosim_reference_adex,
    cosim_reference_conductance_rates,
    cosim_reference_connor_stevens,
    cosim_reference_dpi_neuron,
    cosim_reference_exp_if,
    cosim_reference_fitzhugh_nagumo,
    cosim_reference_fitzhugh_rinzel,
    cosim_reference_glif,
    cosim_reference_hindmarsh_rose,
    cosim_reference_hodgkin_huxley,
    cosim_reference_ibarz_tanaka,
    cosim_reference_izhikevich2007,
    cosim_reference_izhikevich_rs,
    cosim_reference_mckean,
    cosim_reference_mihalas_niebur,
    cosim_reference_morris_lecar,
    cosim_reference_perfect_integrator,
    cosim_reference_pernarowski,
    cosim_reference_quadratic_if,
    cosim_reference_rulkov_map,
    cosim_reference_statistics,
    cosim_reference_terman_wang,
    cosim_reference_theta,
    cosim_reference_wilson_hr,
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

_ADEX_NAMES = ("_adex_subthreshold_euler_features",)

_CONDUCTANCE_RATE_NAMES = (
    "_np_exp",
    "_reference_exprel",
)

_CONNOR_STEVENS_NAMES = (
    "_connor_stevens_hand_spike_count",
    "_connor_stevens_macrostep_rk4_features",
)

_DPI_NEURON_NAMES = (
    "_dpi_neuron_driven_euler_features",
    "_dpi_neuron_hand_spike_count",
    "_dpi_neuron_verilog_q1616_trace",
)

_EXP_IF_NAMES = ("_exp_if_rk4_features",)

_FITZHUGH_NAGUMO_NAMES = (
    "_fitzhugh_nagumo_hand_spike_count",
    "_fitzhugh_nagumo_rk4_features",
)

_FITZHUGH_RINZEL_NAMES = (
    "_fitzhugh_rinzel_hand_spike_count",
    "_fitzhugh_rinzel_rk4_features",
)

_GLIF_NAMES = (
    "_glif_driven_rk4_features",
    "_glif_hand_spike_count",
)

_HINDMARSH_ROSE_NAMES = (
    "_hindmarsh_rose_hand_spike_count",
    "_hindmarsh_rose_rk4_features",
)

_HODGKIN_HUXLEY_NAMES = (
    "_hodgkin_huxley_hand_spike_count",
    "_hodgkin_huxley_macrostep_rk4_features",
)

_IBARZ_TANAKA_NAMES = ("_ibarz_tanaka_verilog_q1616_trace",)

_IZHIKEVICH2007_NAMES = (
    "_izhikevich2007_euler_features",
    "_izhikevich2007_hand_euler_spike_count",
)

_IZHIKEVICH_RS_NAMES = ("_izhikevich_rs_euler_features",)

_MCKEAN_NAMES = (
    "_mckean_hand_spike_count",
    "_mckean_rk4_features",
)

_MIHALAS_NIEBUR_NAMES = (
    "_mihalas_niebur_driven_rk4_features",
    "_mihalas_niebur_hand_spike_count",
)

_MORRIS_LECAR_NAMES = (
    "_morris_lecar_hand_spike_count",
    "_morris_lecar_rk4_features",
)

_PERFECT_INTEGRATOR_NAMES = (
    "_perfect_integrator_hand_spike_count",
    "_perfect_integrator_sawtooth_features",
)

_PERNAROWSKI_NAMES = (
    "_pernarowski_hand_spike_count",
    "_pernarowski_rk4_features",
)

_QUADRATIC_IF_NAMES = ("_quadratic_if_zero_current_features",)

_RULKOV_MAP_NAMES = (
    "_rulkov_map_features",
    "_rulkov_map_verilog_q1616_trace",
)

_THETA_NAMES = ("_theta_constant_current_features",)

_TERMAN_WANG_NAMES = (
    "_terman_wang_hand_spike_count",
    "_terman_wang_rk4_features",
)

_WILSON_HR_NAMES = (
    "_wilson_hr_hand_spike_count",
    "_wilson_hr_rk4_features",
)


def test_legacy_surface_reexports_runtime_objects_without_wrappers() -> None:
    for name in _RUNTIME_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_runtime, name)

    for name in _ADEX_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_reference_adex, name)
    for name in _CONDUCTANCE_RATE_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_reference_conductance_rates, name)
    for name in _CONNOR_STEVENS_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_reference_connor_stevens, name)
    for name in _DPI_NEURON_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_reference_dpi_neuron, name)
    for name in _EXP_IF_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_reference_exp_if, name)
    for name in _FITZHUGH_NAGUMO_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_reference_fitzhugh_nagumo, name)
    for name in _FITZHUGH_RINZEL_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_reference_fitzhugh_rinzel, name)
    for name in _GLIF_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_reference_glif, name)
    for name in _HINDMARSH_ROSE_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_reference_hindmarsh_rose, name)
    for name in _HODGKIN_HUXLEY_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_reference_hodgkin_huxley, name)
    for name in _IBARZ_TANAKA_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_reference_ibarz_tanaka, name)
    for name in _IZHIKEVICH2007_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_reference_izhikevich2007, name)
    for name in _IZHIKEVICH_RS_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_reference_izhikevich_rs, name)
    assert cosim_support._MCKEAN_PARAMS is cosim_reference_mckean._MCKEAN_PARAMS
    for name in _MCKEAN_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_reference_mckean, name)
    assert (
        cosim_support._MIHALAS_NIEBUR_PARAMS
        is cosim_reference_mihalas_niebur._MIHALAS_NIEBUR_PARAMS
    )
    for name in _MIHALAS_NIEBUR_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_reference_mihalas_niebur, name)
    for name in _MORRIS_LECAR_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_reference_morris_lecar, name)
    for name in _PERFECT_INTEGRATOR_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_reference_perfect_integrator, name)
    for name in _PERNAROWSKI_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_reference_pernarowski, name)
    for name in _QUADRATIC_IF_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_reference_quadratic_if, name)
    for name in _RULKOV_MAP_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_reference_rulkov_map, name)
    for name in _THETA_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_reference_theta, name)
    for name in _TERMAN_WANG_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_reference_terman_wang, name)
    for name in _WILSON_HR_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_reference_wilson_hr, name)
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


def test_connor_stevens_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_CONNOR_STEVENS_NAMES)

    owner_tree = ast.parse(
        Path(cosim_reference_connor_stevens.__file__).read_text(encoding="utf-8")
    )
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_CONNOR_STEVENS_NAMES)


def test_runtime_dependency_is_one_way_and_surfaces_cannot_regrow() -> None:
    runtime_text = Path(cosim_runtime.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in runtime_text
    assert len(runtime_text.splitlines()) <= 180
    adex_text = Path(cosim_reference_adex.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in adex_text
    assert len(adex_text.splitlines()) <= 80
    conductance_rate_text = Path(cosim_reference_conductance_rates.__file__).read_text(
        encoding="utf-8"
    )
    assert "cosim_support" not in conductance_rate_text
    assert len(conductance_rate_text.splitlines()) <= 60
    connor_stevens_text = Path(cosim_reference_connor_stevens.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in connor_stevens_text
    assert len(connor_stevens_text.splitlines()) <= 140
    dpi_neuron_text = Path(cosim_reference_dpi_neuron.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in dpi_neuron_text
    assert len(dpi_neuron_text.splitlines()) <= 210
    exp_if_text = Path(cosim_reference_exp_if.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in exp_if_text
    assert len(exp_if_text.splitlines()) <= 80
    fitzhugh_nagumo_text = Path(cosim_reference_fitzhugh_nagumo.__file__).read_text(
        encoding="utf-8"
    )
    assert "cosim_support" not in fitzhugh_nagumo_text
    assert len(fitzhugh_nagumo_text.splitlines()) <= 90
    fitzhugh_rinzel_text = Path(cosim_reference_fitzhugh_rinzel.__file__).read_text(
        encoding="utf-8"
    )
    assert "cosim_support" not in fitzhugh_rinzel_text
    assert len(fitzhugh_rinzel_text.splitlines()) <= 100
    glif_text = Path(cosim_reference_glif.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in glif_text
    assert len(glif_text.splitlines()) <= 125
    hindmarsh_rose_text = Path(cosim_reference_hindmarsh_rose.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in hindmarsh_rose_text
    assert len(hindmarsh_rose_text.splitlines()) <= 100
    hodgkin_huxley_text = Path(cosim_reference_hodgkin_huxley.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in hodgkin_huxley_text
    assert len(hodgkin_huxley_text.splitlines()) <= 120
    ibarz_tanaka_text = Path(cosim_reference_ibarz_tanaka.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in ibarz_tanaka_text
    assert len(ibarz_tanaka_text.splitlines()) <= 100
    izhikevich2007_text = Path(cosim_reference_izhikevich2007.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in izhikevich2007_text
    assert len(izhikevich2007_text.splitlines()) <= 80
    izhikevich_rs_text = Path(cosim_reference_izhikevich_rs.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in izhikevich_rs_text
    assert len(izhikevich_rs_text.splitlines()) <= 70
    mckean_text = Path(cosim_reference_mckean.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in mckean_text
    assert len(mckean_text.splitlines()) <= 105
    mihalas_niebur_text = Path(cosim_reference_mihalas_niebur.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in mihalas_niebur_text
    assert len(mihalas_niebur_text.splitlines()) <= 155
    morris_lecar_text = Path(cosim_reference_morris_lecar.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in morris_lecar_text
    assert len(morris_lecar_text.splitlines()) <= 115
    assert (
        len(Path(cosim_reference_statistics.__file__).read_text(encoding="utf-8").splitlines())
        <= 55
    )
    perfect_text = Path(cosim_reference_perfect_integrator.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in perfect_text
    assert len(perfect_text.splitlines()) <= 65
    pernarowski_text = Path(cosim_reference_pernarowski.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in pernarowski_text
    assert len(pernarowski_text.splitlines()) <= 100
    quadratic_if_text = Path(cosim_reference_quadratic_if.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in quadratic_if_text
    assert len(quadratic_if_text.splitlines()) <= 30
    rulkov_map_text = Path(cosim_reference_rulkov_map.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in rulkov_map_text
    assert len(rulkov_map_text.splitlines()) <= 160
    theta_text = Path(cosim_reference_theta.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in theta_text
    assert len(theta_text.splitlines()) <= 40
    terman_wang_text = Path(cosim_reference_terman_wang.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in terman_wang_text
    assert len(terman_wang_text.splitlines()) <= 90
    wilson_hr_text = Path(cosim_reference_wilson_hr.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in wilson_hr_text
    assert len(wilson_hr_text.splitlines()) <= 90
    assert len(Path(cosim_support.__file__).read_text(encoding="utf-8").splitlines()) <= 520


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


def test_pernarowski_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_PERNAROWSKI_NAMES)

    owner_tree = ast.parse(Path(cosim_reference_pernarowski.__file__).read_text(encoding="utf-8"))
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_PERNAROWSKI_NAMES)


def test_glif_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_GLIF_NAMES)

    glif_tree = ast.parse(Path(cosim_reference_glif.__file__).read_text(encoding="utf-8"))
    glif_functions = {node.name for node in glif_tree.body if isinstance(node, ast.FunctionDef)}
    assert glif_functions == set(_GLIF_NAMES)


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


def test_exp_if_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_EXP_IF_NAMES)

    owner_tree = ast.parse(Path(cosim_reference_exp_if.__file__).read_text(encoding="utf-8"))
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_EXP_IF_NAMES)


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


def test_rulkov_map_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_RULKOV_MAP_NAMES)

    owner_tree = ast.parse(Path(cosim_reference_rulkov_map.__file__).read_text(encoding="utf-8"))
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_RULKOV_MAP_NAMES)


def test_theta_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_THETA_NAMES)

    theta_tree = ast.parse(Path(cosim_reference_theta.__file__).read_text(encoding="utf-8"))
    theta_functions = {node.name for node in theta_tree.body if isinstance(node, ast.FunctionDef)}
    assert theta_functions == set(_THETA_NAMES)


def test_terman_wang_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_TERMAN_WANG_NAMES)

    owner_tree = ast.parse(Path(cosim_reference_terman_wang.__file__).read_text(encoding="utf-8"))
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_TERMAN_WANG_NAMES)


def test_wilson_hr_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_WILSON_HR_NAMES)

    owner_tree = ast.parse(Path(cosim_reference_wilson_hr.__file__).read_text(encoding="utf-8"))
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_WILSON_HR_NAMES)
