# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary substrate architecture contracts

"""Enforce responsibility ownership and historical evolutionary API identity."""

from __future__ import annotations

import ast
import hashlib
import json
import pickle
from pathlib import Path
from types import ModuleType

import sc_neurocore.evo_substrate as package
import sc_neurocore.evo_substrate.evo_substrate as facade
from sc_neurocore.evo_substrate import (
    deployment,
    development,
    ecology,
    emission,
    fitness,
    genome,
    lineage,
    organism,
    replication,
    safety,
    selection,
    speciation,
    statistics,
    variation,
)
from sc_neurocore.fault_injection import (
    DegradationAction,
    DegradationPlan,
    FaultModel,
    GracefulDegradationPolicy,
)


REPO_ROOT = Path(__file__).parents[2]
PACKAGE_ROOT = REPO_ROOT / "src" / "sc_neurocore" / "evo_substrate"
RESPONSIBILITY_MODULES = (
    genome,
    organism,
    variation,
    fitness,
    safety,
    speciation,
    lineage,
    selection,
    ecology,
    development,
    statistics,
    emission,
    deployment,
    replication,
)


def _runtime_dependencies(module: ModuleType) -> set[str]:
    """Return focused imports executed outside type-checking blocks."""
    assert module.__file__ is not None
    module_path = Path(module.__file__).resolve()
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    type_only_lines: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            for child in ast.walk(node):
                if hasattr(child, "lineno") and hasattr(child, "end_lineno"):
                    type_only_lines.update(range(child.lineno, child.end_lineno + 1))
    module_names = {item.__name__.rsplit(".", maxsplit=1)[-1] for item in RESPONSIBILITY_MODULES}
    return {
        node.module.rsplit(".", maxsplit=1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and node.module.startswith("sc_neurocore.evo_substrate.")
        and node.lineno not in type_only_lines
        and node.module.rsplit(".", maxsplit=1)[-1] in module_names
    }


def _source_digest(paths: set[Path]) -> str:
    """Return the benchmark-compatible digest for repository source paths."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def test_facade_defines_no_new_behaviour() -> None:
    """Keep the historical module as imports plus compatibility metadata."""
    facade_path = PACKAGE_ROOT / "evo_substrate.py"
    lines = facade_path.read_text(encoding="utf-8").splitlines()
    tree = ast.parse("\n".join(lines))
    definitions = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    assert definitions == []
    assert len(lines) <= 180
    assert len(facade.__all__) == 56
    assert package.__all__ == facade.__all__


def test_responsibility_modules_exclusively_own_public_symbols() -> None:
    """Partition every historical symbol across one focused owner."""
    owners: dict[str, str] = {}
    for module in RESPONSIBILITY_MODULES:
        for symbol in module.__all__:
            assert symbol not in owners, (
                f"{symbol} owned by both {owners[symbol]} and {module.__name__}"
            )
            owners[symbol] = module.__name__
        assert module.__file__ is not None
        module_path = Path(module.__file__).resolve()
        assert len(module_path.read_text(encoding="utf-8").splitlines()) <= 350

    assert set(owners) == set(facade.__all__)
    assert not (Path(__file__).parent / "test_evo_substrate.py").exists()
    assert all(
        len(path.read_text(encoding="utf-8").splitlines()) <= 500
        for path in Path(__file__).parent.glob("test_*.py")
    )


def test_runtime_import_graph_is_acyclic() -> None:
    """Keep the focused runtime dependency graph explicit and acyclic."""
    expected = {
        "genome": set(),
        "organism": {"genome"},
        "variation": {"genome"},
        "fitness": {"genome"},
        "safety": {"genome", "organism"},
        "speciation": {"genome", "organism"},
        "lineage": {"organism"},
        "selection": {"fitness", "genome", "organism"},
        "ecology": {"organism"},
        "development": set(),
        "statistics": {"genome", "organism"},
        "emission": {"genome"},
        "deployment": {"organism"},
        "replication": {
            "ecology",
            "fitness",
            "genome",
            "lineage",
            "organism",
            "safety",
            "selection",
            "speciation",
            "statistics",
            "variation",
        },
    }
    dependencies = {
        module.__name__.rsplit(".", maxsplit=1)[-1]: _runtime_dependencies(module)
        for module in RESPONSIBILITY_MODULES
    }
    assert dependencies == expected

    resolved: set[str] = set()
    remaining = dict(dependencies)
    while remaining:
        ready = {name for name, deps in remaining.items() if deps <= resolved}
        assert ready, f"cyclic evolutionary-substrate dependency graph: {remaining}"
        resolved.update(ready)
        for name in ready:
            del remaining[name]


def test_public_symbols_keep_historical_pickle_identity() -> None:
    """Resolve every moved public object through its historical module path."""
    for name in facade.__all__:
        value = getattr(facade, name)
        assert value is getattr(package, name)
        assert value.__module__ == "sc_neurocore.evo_substrate.evo_substrate"
        assert pickle.loads(pickle.dumps(value)) is value


def test_legacy_fault_contract_reexports_remain_available() -> None:
    """Retain fault-plan objects used by historical replication integrations."""
    assert facade.DegradationAction is DegradationAction
    assert facade.DegradationPlan is DegradationPlan
    assert facade.FaultModel is FaultModel
    assert facade.GracefulDegradationPolicy is GracefulDegradationPolicy


def test_polyglot_runner_sources_remain_wired() -> None:
    """Keep every runner wired and benchmark evidence bound to current sources."""
    paths = (
        "crates/evo_substrate_core/Cargo.toml",
        "crates/evo_substrate_core/src/lib.rs",
        "crates/evo_substrate_core/src/runner.rs",
        "src/sc_neurocore/accel/go/evo_substrate/main.go",
        "src/sc_neurocore/accel/go/evo_substrate/runner.go",
        "src/sc_neurocore/accel/julia/evo_substrate/evo_runner.jl",
        "src/sc_neurocore/accel/julia/evo_substrate/evo_substrate.jl",
        "src/sc_neurocore/accel/mojo/kernels/evo_runner.mojo",
        "src/sc_neurocore/accel/mojo/kernels/evo_substrate.mojo",
        "benchmarks/bench_evo_substrate.py",
        "benchmarks/bench_evo_substrate_multilang.py",
    )
    assert all((REPO_ROOT / relative).is_file() for relative in paths)

    python_sources = set(PACKAGE_ROOT.glob("*.py"))
    python_evidence = json.loads(
        (REPO_ROOT / "benchmarks/results/bench_evo_substrate.json").read_text(encoding="utf-8")
    )
    assert python_evidence["source"]["source_file_count"] == len(python_sources)
    assert python_evidence["source"]["source_sha256"] == _source_digest(python_sources)

    multilang_sources = set(python_sources)
    for root in (
        REPO_ROOT / "crates/evo_substrate_core",
        REPO_ROOT / "src/sc_neurocore/accel/go/evo_substrate",
        REPO_ROOT / "src/sc_neurocore/accel/julia/evo_substrate",
    ):
        multilang_sources.update(
            path
            for path in root.rglob("*")
            if path.is_file()
            and "target" not in path.parts
            and path.suffix in {".py", ".rs", ".go", ".jl", ".toml"}
        )
    mojo = REPO_ROOT / "src/sc_neurocore/accel/mojo/kernels"
    multilang_sources.update(
        mojo / name
        for name in ("evo_substrate.mojo", "evo_substrate_bench.mojo", "evo_runner.mojo")
    )
    multilang_evidence = json.loads(
        (REPO_ROOT / "benchmarks/results/bench_evo_substrate_multilang.json").read_text(
            encoding="utf-8"
        )
    )
    assert multilang_evidence["source"]["source_file_count"] == len(multilang_sources)
    assert multilang_evidence["source"]["source_sha256"] == _source_digest(multilang_sources)
    assert multilang_evidence["unavailable"] == []
