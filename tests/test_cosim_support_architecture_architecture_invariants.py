# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (architecture_invariants) from former test_cosim_support_architecture.py

"""Pin the co-simulation facade's re-exports and its one-way module graph."""

from __future__ import annotations

import ast
from pathlib import Path
from types import ModuleType

from tests import (
    cosim_reference_adex,
    cosim_reference_conductance_rates,
    cosim_reference_connor_stevens,
    cosim_reference_dpi_neuron,
    cosim_reference_exp_if,
    cosim_reference_exponential_relaxation,
    cosim_reference_fitzhugh_nagumo,
    cosim_reference_fitzhugh_rinzel,
    cosim_reference_glif,
    cosim_reference_hindmarsh_rose,
    cosim_reference_hodgkin_huxley,
    cosim_reference_ibarz_tanaka,
    cosim_reference_izhikevich2007,
    cosim_reference_izhikevich_rs,
    cosim_reference_lif,
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
    cosim_reference_wang_buzsaki,
    cosim_reference_wilson_hr,
    cosim_rtl_spike_execution,
    cosim_runtime,
    cosim_support,
)
from tests.cosim_support_architecture_support import (
    _ADEX_NAMES,
    _CONDUCTANCE_RATE_NAMES,
    _CONNOR_STEVENS_NAMES,
    _DPI_NEURON_NAMES,
    _EXPONENTIAL_RELAXATION_NAMES,
    _EXP_IF_NAMES,
    _FITZHUGH_NAGUMO_NAMES,
    _FITZHUGH_RINZEL_NAMES,
    _GLIF_NAMES,
    _HINDMARSH_ROSE_NAMES,
    _HODGKIN_HUXLEY_NAMES,
    _IBARZ_TANAKA_NAMES,
    _IZHIKEVICH2007_NAMES,
    _IZHIKEVICH_RS_NAMES,
    _LIF_NAMES,
    _MCKEAN_NAMES,
    _MIHALAS_NIEBUR_NAMES,
    _MORRIS_LECAR_NAMES,
    _PERFECT_INTEGRATOR_NAMES,
    _PERNAROWSKI_NAMES,
    _QUADRATIC_IF_NAMES,
    _RTL_SPIKE_EXECUTION_NAMES,
    _RULKOV_MAP_NAMES,
    _RUNTIME_NAMES,
    _TERMAN_WANG_NAMES,
    _THETA_NAMES,
    _WANG_BUZSAKI_NAMES,
    _WILSON_HR_NAMES,
)

# Every module that owns part of the facade's surface; none may import the facade.
_FACADE_INDEPENDENT_OWNERS: tuple[ModuleType, ...] = (
    cosim_rtl_spike_execution,
    cosim_runtime,
    cosim_reference_adex,
    cosim_reference_conductance_rates,
    cosim_reference_connor_stevens,
    cosim_reference_dpi_neuron,
    cosim_reference_exp_if,
    cosim_reference_exponential_relaxation,
    cosim_reference_fitzhugh_nagumo,
    cosim_reference_fitzhugh_rinzel,
    cosim_reference_glif,
    cosim_reference_hindmarsh_rose,
    cosim_reference_hodgkin_huxley,
    cosim_reference_ibarz_tanaka,
    cosim_reference_izhikevich2007,
    cosim_reference_izhikevich_rs,
    cosim_reference_lif,
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
    cosim_reference_wang_buzsaki,
    cosim_reference_wilson_hr,
)


def test_legacy_surface_reexports_runtime_objects_without_wrappers() -> None:
    """The facade re-exports each owner's objects themselves, never a wrapper."""
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
    """The runtime owns exactly its functions, and the facade defines none of them."""
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
    """RTL spike execution owns exactly its functions, and the facade defines none."""
    facade_tree = ast.parse(Path(cosim_support.__file__).read_text(encoding="utf-8"))
    facade_functions = {node.name for node in facade_tree.body if isinstance(node, ast.FunctionDef)}
    assert facade_functions.isdisjoint(_RTL_SPIKE_EXECUTION_NAMES)

    owner_tree = ast.parse(Path(cosim_rtl_spike_execution.__file__).read_text(encoding="utf-8"))
    owner_functions = {node.name for node in owner_tree.body if isinstance(node, ast.FunctionDef)}
    assert owner_functions == set(_RTL_SPIKE_EXECUTION_NAMES)


def test_owners_are_facade_independent_and_the_facade_defines_nothing() -> None:
    """No owner module imports the facade, and the facade holds only imports."""
    for owner in _FACADE_INDEPENDENT_OWNERS:
        owner_path = Path(str(owner.__file__))
        assert "cosim_support" not in owner_path.read_text(encoding="utf-8"), owner_path.name
    facade_tree = ast.parse(Path(str(cosim_support.__file__)).read_text(encoding="utf-8"))
    assert ast.get_docstring(facade_tree) is not None
    assert all(isinstance(node, ast.Import | ast.ImportFrom) for node in facade_tree.body[1:])
