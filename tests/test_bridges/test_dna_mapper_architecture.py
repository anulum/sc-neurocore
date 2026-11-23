# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Architecture contracts for the modular DNA mapper."""

from __future__ import annotations

import ast
from pathlib import Path

import sc_neurocore.bridges.dna_mapper as dna_mapper

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BRIDGES_ROOT = _REPO_ROOT / "src" / "sc_neurocore" / "bridges"
_FACADE = _BRIDGES_ROOT / "dna_mapper.py"
_FALSE_MIRRORS = (
    _REPO_ROOT / "src/sc_neurocore/accel/julia/bridges/dna_mapper.jl",
    _REPO_ROOT / "src/sc_neurocore/accel/mojo/kernels/dna_mapper.mojo",
    _REPO_ROOT / "src/sc_neurocore/accel/go/services/dna_mapper/dna_mapper.go",
)
_RUST_MIRROR = _REPO_ROOT / "src/sc_neurocore/accel/rust/safety/dna_mapper.rs"
_RESPONSIBILITY_MODULES = (
    "dna_analysis",
    "dna_bridge",
    "dna_compilers",
    "dna_encoding",
    "dna_io",
    "dna_lab",
    "dna_sequences",
    "dna_simulation",
    "dna_thermodynamics",
    "dna_types",
    "dna_views",
)
# The concern of each responsibility module, as the top-level classes and functions it owns.
# A new definition is placed by deciding which concern it belongs to, then listing it here.
_OWNED_DEFINITIONS: dict[str, frozenset[str]] = {
    "dna_analysis": frozenset(
        {
            "CrossHybridizationChecker",
            "GateOptimizer",
            "HairpinChecker",
            "TopologicalAnalyzer",
        }
    ),
    "dna_bridge": frozenset(
        {
            "BitstreamToDNA",
            "SCNetworkBridge",
        }
    ),
    "dna_compilers": frozenset(
        {
            "EnzymaticGateCompiler",
            "StrandDisplacementCompiler",
        }
    ),
    "dna_encoding": frozenset(
        {
            "DualRailEncoder",
            "GF4ErrorCorrection",
        }
    ),
    "dna_io": frozenset(
        {
            "export_fasta",
            "export_genbank",
            "export_json",
            "export_nupack_input",
        }
    ),
    "dna_lab": frozenset(
        {
            "PlateLayout",
            "estimate_cost",
            "generate_protocol",
        }
    ),
    "dna_sequences": frozenset(
        {
            "SequenceDesigner",
        }
    ),
    "dna_simulation": frozenset(
        {
            "ConcentrationOptimizer",
            "DegradationModel",
            "KineticSimulator",
            "NoiseModel",
            "SCPrecisionAnalyzer",
            "check_kinetics",
        }
    ),
    "dna_thermodynamics": frozenset(
        {
            "NUPACKInterface",
            "_can_pair",
            "_canonical_sequence",
            "_configure_nupack_backend",
            "_fallback_nupack_backend",
            "_fallback_pair_energy",
            "_fallback_pair_probability_matrix",
            "_fallback_secondary_structure",
            "_hairpin_loop_penalty",
        }
    ),
    "dna_types": frozenset(
        {
            "CompilationMethod",
            "DNACircuitDesign",
            "DNAGate",
            "DNAStrand",
            "GateType",
        }
    ),
    "dna_views": frozenset(
        {
            "visualize_circuit",
            "visualize_kinetics",
        }
    ),
}
_PUBLIC_NAMES = (
    "BitstreamToDNA",
    "CompilationMethod",
    "ConcentrationOptimizer",
    "CrossHybridizationChecker",
    "DNACircuitDesign",
    "DNAGate",
    "DNAStrand",
    "DegradationModel",
    "DualRailEncoder",
    "EnzymaticGateCompiler",
    "GF4ErrorCorrection",
    "GateOptimizer",
    "GateType",
    "HairpinChecker",
    "KineticSimulator",
    "NUPACKInterface",
    "NoiseModel",
    "PlateLayout",
    "SCNetworkBridge",
    "SCPrecisionAnalyzer",
    "SequenceDesigner",
    "StrandDisplacementCompiler",
    "TopologicalAnalyzer",
    "estimate_cost",
    "export_fasta",
    "export_genbank",
    "export_json",
    "export_nupack_input",
    "generate_protocol",
    "visualize_circuit",
    "visualize_kinetics",
)


def _module_path(name: str) -> Path:
    """Return the source path for one DNA responsibility module."""
    return _BRIDGES_ROOT / f"{name}.py"


def _dna_imports(path: Path) -> set[str]:
    """Return DNA responsibility modules imported by one source file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            module = node.module.rsplit(".", maxsplit=1)[-1]
            if module.startswith("dna_"):
                imports.add(module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.rsplit(".", maxsplit=1)[-1]
                if module.startswith("dna_"):
                    imports.add(module)
    return imports


def _assert_acyclic(graph: dict[str, set[str]]) -> None:
    """Fail with a readable path when the module graph contains a cycle."""
    visiting: list[str] = []
    visited: set[str] = set()

    def visit(module: str) -> None:
        if module in visited:
            return
        if module in visiting:
            start = visiting.index(module)
            cycle = visiting[start:] + [module]
            raise AssertionError(f"DNA module import cycle: {' -> '.join(cycle)}")
        visiting.append(module)
        for dependency in sorted(graph[module]):
            visit(dependency)
        visiting.pop()
        visited.add(module)

    for module in sorted(graph):
        visit(module)


def test_facade_stays_thin_and_definition_free() -> None:
    """Keep the public module a compatibility boundary that defines nothing of its own."""
    tree = ast.parse(_FACADE.read_text(encoding="utf-8"))
    definitions = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    assert definitions == []


def test_responsibility_modules_own_their_concerns_and_stay_acyclic() -> None:
    """Each module defines exactly its declared names, none twice, in an acyclic graph."""
    graph: dict[str, set[str]] = {}
    assert set(_OWNED_DEFINITIONS) == set(_RESPONSIBILITY_MODULES)
    owner: dict[str, str] = {}
    for module in _RESPONSIBILITY_MODULES:
        path = _module_path(module)
        assert path.is_file()
        tree = ast.parse(path.read_text(encoding="utf-8"))
        kinds = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        defined = {node.name for node in tree.body if isinstance(node, kinds)}
        assert defined == _OWNED_DEFINITIONS[module], module
        for definition in defined:
            assert definition not in owner, (
                f"{definition} is defined in {owner.get(definition)} and {module}"
            )
            owner[definition] = module
        imports = _dna_imports(path)
        assert "dna_mapper" not in imports
        graph[module] = imports & set(_RESPONSIBILITY_MODULES)

    _assert_acyclic(graph)


def test_public_surface_and_historical_module_identity_are_stable() -> None:
    """Pin the canonical export list and serialized object compatibility."""
    assert tuple(dna_mapper.__all__) == _PUBLIC_NAMES
    for name in _PUBLIC_NAMES:
        value = getattr(dna_mapper, name)
        assert value.__module__ == "sc_neurocore.bridges.dna_mapper"


def test_only_the_maintained_rust_mirror_is_shipped() -> None:
    """Do not reintroduce non-compiling or no-op generated mirror scaffolds."""
    assert _RUST_MIRROR.is_file()
    assert [path for path in _FALSE_MIRRORS if path.exists()] == []
