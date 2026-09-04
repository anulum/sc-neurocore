# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (architecture_invariants) from former test_cosim_support_architecture.py

from __future__ import annotations

from tests.cosim_support_architecture_support import *  # noqa: F403


def test_legacy_surface_reexports_runtime_objects_without_wrappers() -> None:
    assert cosim_support.compile_to_verilog is cosim_rtl_spike_execution.compile_to_verilog
    for name in _RTL_SPIKE_EXECUTION_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_rtl_spike_execution, name)
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
    for name in _EXPONENTIAL_RELAXATION_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_reference_exponential_relaxation, name)
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
    for name in _LIF_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_reference_lif, name)
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
    for name in _WANG_BUZSAKI_NAMES:
        assert getattr(cosim_support, name) is getattr(cosim_reference_wang_buzsaki, name)
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


def test_rtl_spike_execution_has_exact_definition_ownership() -> None:
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_RTL_SPIKE_EXECUTION_NAMES)

    owner_tree = ast.parse(Path(cosim_rtl_spike_execution.__file__).read_text(encoding="utf-8"))
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_RTL_SPIKE_EXECUTION_NAMES)


def test_runtime_dependency_is_one_way_and_surfaces_cannot_regrow() -> None:
    rtl_spike_execution_text = Path(cosim_rtl_spike_execution.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in rtl_spike_execution_text
    assert len(rtl_spike_execution_text.splitlines()) <= 260
    runtime_text = Path(cosim_runtime.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in runtime_text
    assert len(runtime_text.splitlines()) <= 180
    adex_text = Path(cosim_reference_adex.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in adex_text
    assert len(adex_text.splitlines()) <= 120
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
    exponential_relaxation_text = Path(cosim_reference_exponential_relaxation.__file__).read_text(
        encoding="utf-8"
    )
    assert "cosim_support" not in exponential_relaxation_text
    assert len(exponential_relaxation_text.splitlines()) <= 40
    fitzhugh_nagumo_text = Path(cosim_reference_fitzhugh_nagumo.__file__).read_text(
        encoding="utf-8"
    )
    assert "cosim_support" not in fitzhugh_nagumo_text
    assert len(fitzhugh_nagumo_text.splitlines()) <= 110
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
    lif_text = Path(cosim_reference_lif.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in lif_text
    assert len(lif_text.splitlines()) <= 30
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
    wang_buzsaki_text = Path(cosim_reference_wang_buzsaki.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in wang_buzsaki_text
    assert len(wang_buzsaki_text.splitlines()) <= 135
    wilson_hr_text = Path(cosim_reference_wilson_hr.__file__).read_text(encoding="utf-8")
    assert "cosim_support" not in wilson_hr_text
    assert len(wilson_hr_text.splitlines()) <= 90
    assert len(Path(cosim_support.__file__).read_text(encoding="utf-8").splitlines()) <= 155
