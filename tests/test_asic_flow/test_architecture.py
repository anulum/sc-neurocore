# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ASIC-flow modular architecture contracts

"""Lock symbol ownership, compatibility, acyclicity, and honest boundaries."""

from __future__ import annotations

import ast
import hashlib
import json
import pickle
from pathlib import Path
from types import ModuleType

import pytest

from sc_neurocore.asic_flow import asic_flow as historical

ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = ROOT / "src/sc_neurocore/asic_flow"
ACCEL_DIR = ROOT / "src/sc_neurocore/accel"

SYMBOL_OWNERS = {
    "constraints": {
        "CDCCheckGenerator",
        "IOConstraintGenerator",
        "IOPin",
        "IRDropGenerator",
        "LECGenerator",
    },
    "decks": {
        "FloorplanGenerator",
        "GDSIIExporter",
        "PlaceRouteGenerator",
        "SDCGenerator",
        "SynthesisGenerator",
    },
    "design": {"DesignParams", "SCASICOptimisationConfig"},
    "estimation": {"DesignEstimate", "PreSynthEstimator"},
    "flow": {
        "ASICFlowBundle",
        "ASICFlowGenerator",
        "ASICFlowOutput",
        "generate_asic_flow_bundle",
    },
    "hierarchy": {"BlockConfig", "HierarchicalFlow"},
    "liberty_corners": {
        "CornerType",
        "DEFAULT_CORNERS",
        "PVTCorner",
    },
    "pdk": {
        "OpenSourcePDKResolver",
        "PDKConfig",
        "PDKResolution",
        "PDKType",
        "PDKValidationResult",
        "ResolvedPDKFiles",
        "validate_pdk",
        "validate_pdk_installation",
    },
    "readiness": {"TapeOutChecklist"},
    "signoff": {
        "DRCViolation",
        "MultiCornerAnalysis",
        "OCVConfig",
        "SignoffCheckResult",
        "SignoffGenerator",
        "SignoffSummary",
    },
}

# Module-private helpers each responsibility module defines beside its owned symbols.
PRIVATE_DEFINITIONS = {
    "flow": {
        "_build_asic_flow_manifest",
        "_design_to_manifest",
        "_formal_evidence_status",
        "_normalise_pdk_type",
        "_pdk_to_manifest",
    },
}
# Owned public names bound by assignment rather than defined as a class or function.
PUBLIC_CONSTANTS = {"DEFAULT_CORNERS"}


def _tree(path: Path) -> ast.Module:
    """Parse one tracked Python source file."""
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _top_level_symbols(path: Path) -> set[str]:
    """Return classes, functions, and named assignments defined by a module."""
    names: set[str] = set()
    for node in _tree(path).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(target.id for target in targets if isinstance(target, ast.Name))
    return names


def _module_dependencies(module_name: str) -> set[str]:
    """Return imports from other ASIC-flow responsibility modules."""
    dependencies: set[str] = set()
    for node in ast.walk(_tree(SOURCE_DIR / f"{module_name}.py")):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        prefix = "sc_neurocore.asic_flow."
        if node.module.startswith(prefix):
            dependency = node.module.removeprefix(prefix).split(".", maxsplit=1)[0]
            assert dependency in SYMBOL_OWNERS
            dependencies.add(dependency)
    return dependencies


def _assert_acyclic(graph: dict[str, set[str]]) -> None:
    """Raise an assertion with the active path when a dependency cycle exists."""
    complete: set[str] = set()
    active: list[str] = []

    def visit(module_name: str) -> None:
        if module_name in active:
            cycle = " -> ".join([*active[active.index(module_name) :], module_name])
            raise AssertionError(f"ASIC-flow import cycle: {cycle}")
        if module_name in complete:
            return
        active.append(module_name)
        for dependency in sorted(graph[module_name]):
            visit(dependency)
        active.pop()
        complete.add(module_name)

    for module_name in sorted(graph):
        visit(module_name)


def test_historical_facade_defines_no_implementation() -> None:
    """The historical module only re-exports responsibility-owned objects."""
    facade_tree = _tree(SOURCE_DIR / "asic_flow.py")
    definitions = [
        node.name
        for node in facade_tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    assert definitions == []
    assert set(historical.__all__) == set().union(*SYMBOL_OWNERS.values())


def test_public_symbols_have_one_responsibility_owner() -> None:
    """Every historical public symbol is defined by exactly one focused module."""
    actual_owners: dict[str, list[str]] = {}
    for module_name in SYMBOL_OWNERS:
        for symbol in _top_level_symbols(SOURCE_DIR / f"{module_name}.py"):
            if symbol in historical.__all__:
                actual_owners.setdefault(symbol, []).append(module_name)

    assert actual_owners == {
        symbol: [module_name]
        for module_name, symbols in SYMBOL_OWNERS.items()
        for symbol in symbols
    }


def test_responsibility_import_graph_is_acyclic() -> None:
    """Focused modules form a real directed acyclic import graph."""
    graph = {module_name: _module_dependencies(module_name) for module_name in SYMBOL_OWNERS}

    _assert_acyclic(graph)
    with pytest.raises(AssertionError, match=r"ASIC-flow import cycle: a -> b -> a"):
        _assert_acyclic({"a": {"b"}, "b": {"a"}})


def test_historical_identity_and_pickle_paths_are_stable() -> None:
    """Public definitions keep their former qualified import and pickle path."""
    for name in historical.__all__:
        if name == "DEFAULT_CORNERS":
            continue
        definition = getattr(historical, name)
        assert definition.__module__ == "sc_neurocore.asic_flow.asic_flow"
        assert pickle.loads(pickle.dumps(definition)) is definition


def test_package_facade_remains_intentionally_narrow() -> None:
    """The package root keeps the established one-command bundle API."""
    package = __import__("sc_neurocore.asic_flow", fromlist=["*"])
    assert isinstance(package, ModuleType)
    assert package.__all__ == ["ASICFlowBundle", "generate_asic_flow_bundle"]
    assert package.ASICFlowBundle is historical.ASICFlowBundle
    assert package.generate_asic_flow_bundle is historical.generate_asic_flow_bundle


def test_each_module_is_declared_and_defines_exactly_what_it_owns() -> None:
    """Every package module is declared, and each defines only its owned symbols and helpers."""
    declared = {"__init__", "asic_flow", *SYMBOL_OWNERS}
    assert {path.stem for path in SOURCE_DIR.glob("*.py")} == declared
    for module_name, symbols in SYMBOL_OWNERS.items():
        definitions = {
            node.name
            for node in _tree(SOURCE_DIR / f"{module_name}.py").body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        }
        expected = (symbols - PUBLIC_CONSTANTS) | PRIVATE_DEFINITIONS.get(module_name, set())
        assert definitions == expected, module_name


def test_nonfunctional_polyglot_mirrors_are_absent() -> None:
    """ASIC deck orchestration has no fabricated numerical acceleration mirror."""
    mirror_files = [
        path for path in ACCEL_DIR.rglob("*") if path.is_file() and "asic_flow" in path.parts
    ]
    rust_registry = ROOT / "src/sc_neurocore/accel/rust/safety/lib.rs"

    assert mirror_files == []
    assert "pub mod asic_flow;" not in rust_registry.read_text(encoding="utf-8")


def test_benchmark_evidence_is_bound_to_live_sources() -> None:
    """The committed benchmark hashes the live package, runner, and probe."""
    result_path = ROOT / "benchmarks/results/bench_asic_flow.json"
    benchmark_path = ROOT / "benchmarks/bench_asic_flow.py"
    probe_path = ROOT / "benchmarks/_asic_flow_probe.py"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    for path in sorted(SOURCE_DIR.glob("*.py")):
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")

    candidate = payload["variants"][1]
    comparison = payload["comparison"]
    assert candidate["source_sha256"] == digest.hexdigest()
    assert candidate["source_file_count"] == len(list(SOURCE_DIR.glob("*.py")))
    assert payload["benchmark_sha256"] == hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
    assert payload["probe_sha256"] == hashlib.sha256(probe_path.read_bytes()).hexdigest()
    assert payload["evidence_class"] == "local_source_behaviour_change_diagnostics"
    assert comparison["generated_output_byte_identical"] is False
    assert comparison["baseline_generated_sha256"] == (
        "ae901f9b10bdc61f0997964d6143568994625bdf89080f02cd58efbc83099653"
    )
    assert comparison["generated_sha256"] == (
        "7d604bbb4f2b37e4748b6df1336efdfc40ccc6d791936576c9de08b60b318d25"
    )
