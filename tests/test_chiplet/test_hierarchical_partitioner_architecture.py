# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hierarchical partitioner architecture contracts

"""Facade identity, dependency, size, and polyglot-authority contracts."""

from __future__ import annotations

import ast
import pickle
from pathlib import Path

from sc_neurocore.chiplet import hierarchical_balancing
from sc_neurocore.chiplet import hierarchical_boundary
from sc_neurocore.chiplet import hierarchical_core
from sc_neurocore.chiplet import hierarchical_graph
from sc_neurocore.chiplet import hierarchical_metrics
from sc_neurocore.chiplet import hierarchical_partitioner as facade
from sc_neurocore.chiplet import hierarchical_reporting
from tools.ci_workflow_inventory import ci_job_text


REPO_ROOT = Path(__file__).parents[2]
PACKAGE_ROOT = REPO_ROOT / "src" / "sc_neurocore" / "chiplet"
SPLIT_MODULES = (
    "hierarchical_backend_runtime",
    "hierarchical_backends",
    "hierarchical_balancing",
    "hierarchical_bisection",
    "hierarchical_boundary",
    "hierarchical_core",
    "hierarchical_graph",
    "hierarchical_metrics",
    "hierarchical_partitioner",
    "hierarchical_refinement",
    "hierarchical_reporting",
)
#: The documented one-way dependency DAG of the responsibility modules.
EXPECTED_DEPENDENCIES: dict[str, set[str]] = {
    "hierarchical_backend_runtime": set(),
    "hierarchical_backends": {
        "hierarchical_backend_runtime",
        "hierarchical_graph",
    },
    "hierarchical_balancing": {
        "hierarchical_boundary",
        "hierarchical_graph",
        "hierarchical_metrics",
    },
    "hierarchical_bisection": {"hierarchical_graph"},
    "hierarchical_boundary": {
        "hierarchical_graph",
        "hierarchical_metrics",
    },
    "hierarchical_core": {
        "hierarchical_backends",
        "hierarchical_bisection",
        "hierarchical_graph",
        "hierarchical_refinement",
    },
    "hierarchical_graph": set(),
    "hierarchical_metrics": {"hierarchical_graph"},
    "hierarchical_refinement": {"hierarchical_graph"},
    "hierarchical_reporting": {
        "hierarchical_boundary",
        "hierarchical_graph",
        "hierarchical_metrics",
    },
}


def test_facade_is_thin_and_defines_only_dynamic_diagnostics() -> None:
    """The historical module remains an import facade, not a second owner."""
    path = PACKAGE_ROOT / "hierarchical_partitioner.py"
    lines = path.read_text(encoding="utf-8").splitlines()
    tree = ast.parse("\n".join(lines))
    definitions = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    assert definitions == ["__getattr__"]
    assert len(facade.__all__) == 21
    assert len(facade.__all__) == len(set(facade.__all__))


def test_public_responsibilities_have_one_owner() -> None:
    """Every historical public symbol is owned by one focused module."""
    owners: dict[str, str] = {}
    for module in (
        hierarchical_graph,
        hierarchical_core,
        hierarchical_metrics,
        hierarchical_boundary,
        hierarchical_balancing,
        hierarchical_reporting,
    ):
        for symbol in module.__all__:
            assert symbol not in owners
            owners[symbol] = module.__name__
    assert set(owners) == set(facade.__all__)


# The concern of each split module, as the top-level classes and functions it owns.
# A new definition is placed by deciding which concern it belongs to, then listing it here.
OWNED_DEFINITIONS: dict[str, frozenset[str]] = {
    "hierarchical_backend_runtime": frozenset(
        {
            "_accel_path",
            "_ensure_go_kl_refine_loaded",
            "_ensure_julia_kl_refine_loaded",
            "_ensure_mojo_kl_refine_loaded",
        }
    ),
    "hierarchical_backends": frozenset(
        {
            "RefinementOwner",
            "decode_part_map",
            "dispatch_refine",
            "encode_csr",
            "refine_go",
            "refine_julia",
            "refine_mojo",
            "refine_rust",
        }
    ),
    "hierarchical_balancing": frozenset(
        {
            "CorrelationLoadBalancer",
            "LoadMetrics",
            "MigrationRecommendation",
            "RankMapper",
        }
    ),
    "hierarchical_bisection": frozenset(
        {
            "BisectionMixin",
        }
    ),
    "hierarchical_boundary": frozenset(
        {
            "BoundarySyncConfig",
            "BoundarySyncProtocol",
            "GhostCellManager",
        }
    ),
    "hierarchical_core": frozenset(
        {
            "HierarchicalPartitioner",
        }
    ),
    "hierarchical_graph": frozenset(
        {
            "CSRGraph",
            "CorrelationAwareGraph",
            "CorrelationEdge",
            "HierarchyLevel",
            "LFSRSeedAllocator",
        }
    ),
    "hierarchical_metrics": frozenset(
        {
            "_build_part_map",
            "calculate_boundary_scc",
            "calculate_comm_volume",
            "calculate_edge_cut",
            "calculate_imbalance_ratio",
            "calculate_mean_boundary_scc",
            "calculate_total_boundary_scc",
        }
    ),
    "hierarchical_partitioner": frozenset(
        {
            "__getattr__",
        }
    ),
    "hierarchical_refinement": frozenset(
        {
            "RefinementMixin",
        }
    ),
    "hierarchical_reporting": frozenset(
        {
            "PartitionReport",
            "build_partition_report",
        }
    ),
}
# The concerns a partitioner test file covers, as the first word after the common prefix.
TEST_CONCERNS = frozenset(
    {"architecture", "backends", "core", "dispatch", "graph", "metrics", "reporting", "runtime"}
)


def test_split_modules_own_their_concerns_and_are_licensed() -> None:
    """Each split module defines exactly its declared names, none twice, under the licence header."""
    assert set(OWNED_DEFINITIONS) == set(SPLIT_MODULES)
    kinds = (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    owner: dict[str, str] = {}
    for module_name in SPLIT_MODULES:
        source = (PACKAGE_ROOT / f"{module_name}.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        defined = {node.name for node in tree.body if isinstance(node, kinds)}
        assert defined == OWNED_DEFINITIONS[module_name], module_name
        for definition in defined:
            assert definition not in owner, (
                f"{definition} is defined in {owner.get(definition)} and {module_name}"
            )
            owner[definition] = module_name
        lines = source.splitlines()
        assert lines[0] == "# SPDX-License-Identifier: AGPL-3.0-or-later"
        assert any('"""' in line for line in lines[:20])


def test_each_partitioner_test_file_covers_one_concern() -> None:
    """Every ``test_hierarchical_partitioner_*`` file names the concern it tests."""
    tests = sorted(Path(__file__).parent.glob("test_hierarchical_partitioner_*.py"))
    assert tests
    for path in tests:
        concern = path.stem.removeprefix("test_hierarchical_partitioner_").split("_", maxsplit=1)[0]
        assert concern in TEST_CONCERNS, path.name


def _chiplet_dependencies(module_name: str) -> set[str]:
    """Return the responsibility modules that one responsibility module imports.

    Parameters
    ----------
    module_name : str
        A module in ``sc_neurocore.chiplet``.

    Returns
    -------
    set of str
        The names in ``EXPECTED_DEPENDENCIES`` it imports, by module or from the package.
    """
    tree = ast.parse((PACKAGE_ROOT / f"{module_name}.py").read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            tail = node.module.rsplit(".", maxsplit=1)[-1]
            if tail in EXPECTED_DEPENDENCIES:
                found.add(tail)
            if node.module == "sc_neurocore.chiplet":
                found.update(
                    alias.name for alias in node.names if alias.name in EXPECTED_DEPENDENCIES
                )
    return found


def test_focused_import_graph_is_acyclic() -> None:
    """The responsibility modules form the documented one-way dependency DAG."""
    dependencies = {name: _chiplet_dependencies(name) for name in EXPECTED_DEPENDENCIES}
    assert dependencies == EXPECTED_DEPENDENCIES

    resolved: set[str] = set()
    remaining = dict(dependencies)
    while remaining:
        ready = {name for name, deps in remaining.items() if deps <= resolved}
        assert ready, f"cyclic dependency graph: {remaining}"
        resolved.update(ready)
        for name in ready:
            del remaining[name]


def test_historical_qualified_names_identity_and_pickle_survive() -> None:
    """Moved public objects retain the facade's introspection and pickle path."""
    assert all(getattr(facade, symbol).__module__ == facade.__name__ for symbol in facade.__all__)
    values = [
        facade.CorrelationEdge(0, 1, scc_weight=0.25),
        facade.BoundarySyncConfig(max_boundary_scc_budget=0.2),
        facade.HierarchyLevel.DIE,
    ]
    for value in values:
        restored = pickle.loads(pickle.dumps(value))
        assert restored == value
        assert type(restored) is type(value)


def test_dynamic_backend_diagnostics_reject_unknown_names() -> None:
    """The facade exposes only the documented private runtime diagnostics."""
    assert isinstance(facade._HAS_RUST_KL_REFINE, bool)
    try:
        facade.__getattr__("_not_a_backend_diagnostic")
    except AttributeError as error:
        assert "no attribute" in str(error)
    else:
        raise AssertionError("unknown diagnostic did not raise AttributeError")


def test_false_mirrors_are_absent_and_real_kernels_remain() -> None:
    """Only executable KL-refinement implementations represent polyglot parity."""
    false_mirrors = (
        "src/sc_neurocore/accel/rust/safety/hierarchical_partitioner.rs",
        "src/sc_neurocore/accel/julia/chiplet/hierarchical_partitioner.jl",
        "src/sc_neurocore/accel/go/services/hierarchical_partitioner/hierarchical_partitioner.go",
        "src/sc_neurocore/accel/mojo/kernels/hierarchical_partitioner.mojo",
    )
    assert all(not (REPO_ROOT / path).exists() for path in false_mirrors)
    registry = REPO_ROOT / "src/sc_neurocore/accel/rust/safety/lib.rs"
    assert "pub mod hierarchical_partitioner;" not in registry.read_text(encoding="utf-8")

    maintained = {
        "engine/src/partition.rs": "pub fn kl_refine",
        "src/sc_neurocore/accel/julia/chiplet/kl_refine.jl": "module KLRefineAccel",
        "src/sc_neurocore/accel/go/partition/partition.go": "func klRefine",
        "src/sc_neurocore/accel/mojo/partition/partition.mojo": "def kl_refine_c",
    }
    for path, marker in maintained.items():
        assert marker in (REPO_ROOT / path).read_text(encoding="utf-8")


def test_ci_enforces_exact_hierarchical_partitioner_coverage() -> None:
    """Python 3.12 CI pins line and branch coverage for the full split."""
    workflow = ci_job_text("test")
    assert "Hierarchical partitioner exact coverage" in workflow
    assert "COVERAGE_FILE=.coverage-hierarchical-partitioner" in workflow
    assert "--include='src/sc_neurocore/chiplet/hierarchical*.py'" in workflow
    assert "tests/test_chiplet/test_hierarchical_partitioner_*.py" in workflow
    assert "--fail-under=100 --show-missing" in workflow
