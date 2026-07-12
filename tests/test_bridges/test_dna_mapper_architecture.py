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
    "dna_sequences",
    "dna_simulation",
    "dna_thermodynamics",
    "dna_types",
)
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
    """Keep the public module as a compatibility and composition boundary."""
    source = _FACADE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    public_definitions = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and not node.name.startswith("_")
    ]

    assert len(source.splitlines()) <= 250
    assert public_definitions == []


def test_responsibility_modules_stay_bounded_and_acyclic() -> None:
    """Prevent a new GodFile or circular responsibility graph."""
    graph: dict[str, set[str]] = {}
    for module in _RESPONSIBILITY_MODULES:
        path = _module_path(module)
        assert path.is_file()
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 650
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
