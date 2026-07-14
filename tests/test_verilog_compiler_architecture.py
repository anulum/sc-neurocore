# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verilog compiler responsibility architecture tests

"""Static contracts for the equation-to-Verilog responsibility split."""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from sc_neurocore.compiler import verilog_compiler


_COMPILER_DIR = Path(verilog_compiler.__file__).resolve().parent
_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULES = {
    "verilog_compiler": _COMPILER_DIR / "verilog_compiler.py",
    "_verilog_integrators": _COMPILER_DIR / "_verilog_integrators.py",
    "_verilog_neuron_core": _COMPILER_DIR / "_verilog_neuron_core.py",
    "_verilog_registered_module": _COMPILER_DIR / "_verilog_registered_module.py",
    "_verilog_folded_datapath": _COMPILER_DIR / "_verilog_folded_datapath.py",
}


def _top_level_definitions(path: Path) -> set[str]:
    """Return function and class names owned directly by one module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _local_dependencies(path: Path) -> set[str]:
    """Return imports from the five-module compiler responsibility graph."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cohort = set(_MODULES)
    dependencies: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            stem = node.module.rsplit(".", maxsplit=1)[-1]
            if stem in cohort:
                dependencies.add(stem)
    return dependencies


def test_facade_preserves_the_two_function_surface_and_module_identity() -> None:
    """Historical imports expose only both emitters under their original module."""
    assert verilog_compiler.__all__ == ["compile_to_datapath", "compile_to_verilog"]
    assert verilog_compiler.compile_to_verilog.__module__ == verilog_compiler.__name__
    assert verilog_compiler.compile_to_datapath.__module__ == verilog_compiler.__name__
    assert _top_level_definitions(_MODULES["verilog_compiler"]) == set()


def test_public_signatures_remain_explicit_and_stable() -> None:
    """The facade retains every established positional and keyword parameter."""
    assert tuple(inspect.signature(verilog_compiler.compile_to_verilog).parameters) == (
        "neuron",
        "module_name",
        "data_width",
        "fraction",
        "signed",
        "overflow",
        "rounding",
        "pipeline_stages",
        "pipeline_points",
    )
    assert tuple(inspect.signature(verilog_compiler.compile_to_datapath).parameters) == (
        "neuron",
        "module_name",
        "data_width",
        "fraction",
        "signed",
        "overflow",
        "rounding",
        "param_ports",
    )


def test_each_module_owns_one_compiler_responsibility() -> None:
    """Top-level definitions cannot drift back into a mixed compiler module."""
    assert _top_level_definitions(_MODULES["_verilog_integrators"]) == {
        "_emit_euler_deriv_wires",
        "_emit_exp_euler_deriv_wires",
        "_emit_gauss_seidel_deriv_wires",
        "_emit_map_deriv_wires",
        "_emit_rk4_deriv_wires",
    }
    assert _top_level_definitions(_MODULES["_verilog_neuron_core"]) == {
        "_NeuronCore",
        "_build_neuron_core",
        "_escape_threshold_wires",
    }
    assert _top_level_definitions(_MODULES["_verilog_registered_module"]) == {"compile_to_verilog"}
    assert _top_level_definitions(_MODULES["_verilog_folded_datapath"]) == {"compile_to_datapath"}


def test_compiler_dependency_graph_is_one_way() -> None:
    """Facade, emitters, shared core, and integrators form an acyclic graph."""
    assert _local_dependencies(_MODULES["verilog_compiler"]) == {
        "_verilog_folded_datapath",
        "_verilog_registered_module",
    }
    assert _local_dependencies(_MODULES["_verilog_registered_module"]) == {"_verilog_neuron_core"}
    assert _local_dependencies(_MODULES["_verilog_folded_datapath"]) == {"_verilog_neuron_core"}
    assert _local_dependencies(_MODULES["_verilog_neuron_core"]) == {"_verilog_integrators"}
    assert _local_dependencies(_MODULES["_verilog_integrators"]) == set()


def test_responsibility_modules_cannot_reaggregate_the_old_godfile() -> None:
    """Responsibility-specific ceilings prevent silent monolith reconstruction."""
    ceilings = {
        "verilog_compiler": 60,
        "_verilog_integrators": 450,
        "_verilog_neuron_core": 450,
        "_verilog_registered_module": 450,
        "_verilog_folded_datapath": 250,
    }
    for name, path in _MODULES.items():
        assert len(path.read_text(encoding="utf-8").splitlines()) <= ceilings[name]


def test_unwired_generated_polyglot_compiler_stubs_and_exports_are_absent() -> None:
    """Dead non-executable mirrors cannot masquerade as compiler backends."""
    removed = (
        "src/sc_neurocore/accel/go/services/equation_compiler/__init__.py",
        "src/sc_neurocore/accel/go/services/equation_compiler/equation_compiler.go",
        "src/sc_neurocore/accel/julia/compiler/equation_compiler.jl",
        "src/sc_neurocore/accel/mojo/kernels/equation_compiler.mojo",
        "src/sc_neurocore/accel/rust/safety/equation_compiler.rs",
    )
    assert all(not (_REPO_ROOT / relative).exists() for relative in removed)
    for crate_root in ("lib.rs", "mod.rs"):
        source = (_REPO_ROOT / "src/sc_neurocore/accel/rust/safety" / crate_root).read_text(
            encoding="utf-8"
        )
        assert "pub mod equation_compiler;" not in source
