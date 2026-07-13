# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bioware architecture regression tests

"""Prevent generated placeholder backends from masquerading as Bioware support."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import pickle

import sc_neurocore.bioware as package
from sc_neurocore.bioware import bioware as historical


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = REPO_ROOT / "src/sc_neurocore/bioware"
TEST_DIR = REPO_ROOT / "tests/test_bioware"
BENCHMARK_PATH = REPO_ROOT / "benchmarks/bench_bioware.py"
PROBE_PATH = REPO_ROOT / "benchmarks/_bioware_probe.py"
RESULT_PATH = REPO_ROOT / "benchmarks/results/bench_bioware.json"
CI_WORKFLOW_PATH = REPO_ROOT / ".github/workflows/ci.yml"
RESPONSIBILITY_MODULES = {
    path.stem for path in SOURCE_DIR.glob("bioware_*.py") if path.stem != "bioware"
}


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _module_dependencies(module_name: str) -> set[str]:
    dependencies: set[str] = set()
    for node in ast.walk(_tree(SOURCE_DIR / f"{module_name}.py")):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        dependency = node.module.rsplit(".", maxsplit=1)[-1]
        if dependency in RESPONSIBILITY_MODULES:
            dependencies.add(dependency)
    return dependencies


def _assert_acyclic(graph: dict[str, set[str]]) -> None:
    complete: set[str] = set()
    active: list[str] = []

    def visit(module_name: str) -> None:
        if module_name in active:
            cycle = " -> ".join([*active[active.index(module_name) :], module_name])
            raise AssertionError(f"Bioware import cycle: {cycle}")
        if module_name in complete:
            return
        active.append(module_name)
        for dependency in sorted(graph[module_name]):
            visit(dependency)
        active.pop()
        complete.add(module_name)

    for module_name in sorted(graph):
        visit(module_name)


def test_historical_facade_contains_no_implementation_definitions() -> None:
    definitions = {
        node.name
        for node in _tree(SOURCE_DIR / "bioware.py").body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }

    assert definitions == set()


def test_responsibility_import_graph_is_acyclic() -> None:
    graph = {
        module_name: _module_dependencies(module_name) for module_name in RESPONSIBILITY_MODULES
    }

    _assert_acyclic(graph)


def test_package_exports_reuse_historical_objects_and_qualified_names() -> None:
    for name in historical.__all__:
        definition = getattr(historical, name)
        assert getattr(package, name) is definition
        if hasattr(definition, "__module__"):
            assert definition.__module__ == "sc_neurocore.bioware.bioware"


def test_established_pickle_paths_remain_stable() -> None:
    for name in (
        "AEREvent",
        "BioHybridFrameResult",
        "BioHybridSession",
        "MEAConfig",
        "MEALayout",
        "SpikeDetector",
        "mea_fitness_hook",
    ):
        definition = getattr(historical, name)
        assert pickle.loads(pickle.dumps(definition)) is definition


def test_source_tests_and_benchmarks_remain_responsibility_sized() -> None:
    source_sizes = {
        path.name: len(path.read_text(encoding="utf-8").splitlines())
        for path in SOURCE_DIR.glob("*.py")
    }
    test_sizes = {
        path.name: len(path.read_text(encoding="utf-8").splitlines())
        for path in TEST_DIR.glob("*.py")
    }

    assert max(source_sizes.values()) <= 300, source_sizes
    assert max(test_sizes.values()) <= 300, test_sizes
    assert len(BENCHMARK_PATH.read_text(encoding="utf-8").splitlines()) <= 350
    assert len(PROBE_PATH.read_text(encoding="utf-8").splitlines()) <= 200


def test_benchmark_evidence_is_source_bound_and_byte_identical() -> None:
    payload = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    source_paths = sorted(SOURCE_DIR.glob("*.py"))
    expected_hashes = {
        path.relative_to(REPO_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source_paths
    }

    assert payload["benchmark_sha256"] == hashlib.sha256(BENCHMARK_PATH.read_bytes()).hexdigest()
    assert payload["probe_sha256"] == hashlib.sha256(PROBE_PATH.read_bytes()).hexdigest()
    assert payload["source_file_count"] == len(source_paths) == 12
    assert payload["source_hashes"] == expected_hashes
    assert payload["configuration"]["iterations"] == 30
    assert payload["configuration"]["warmups"] == 2
    assert payload["results"]["parent"]["sample_count"] == 30
    assert payload["results"]["candidate"]["sample_count"] == 30
    assert set(payload["comparison"]) >= {
        "import_ns",
        "max_rss_kib",
        "pipeline_ns",
        "subprocess_wall_ns",
    }
    expected = {
        "canonical_output_byte_identical": True,
        "canonical_sha256": "2491dc73a2de93a45a1cc944539c170b151403e42b973b18806143f318b7d669",
        "canonical_bytes": 6865,
        "spike_count": 7,
        "aer_event_count": 7,
        "bitstream_count": 4,
        "opto_pulse_count": 4,
    }
    assert {key: payload["comparison"][key] for key in expected} == expected


def test_ci_enforces_exact_bioware_coverage_on_python_312() -> None:
    """Keep the Bioware branch gate exact and independent of global omits."""
    workflow = CI_WORKFLOW_PATH.read_text(encoding="utf-8")
    step = workflow.split("- name: Bioware exact coverage", maxsplit=1)[1].split(
        "- name: Upload test results", maxsplit=1
    )[0]

    assert "if: matrix.python-version == '3.12'" in step
    assert "COVERAGE_FILE=.coverage-bioware" in step
    assert "--rcfile=/dev/null --branch" in step
    assert "--include='src/sc_neurocore/bioware/*.py'" in step
    assert "-m pytest tests/test_bioware -q" in step
    assert "--fail-under=100 --show-missing" in step


def test_bioware_has_no_placeholder_polyglot_mirrors() -> None:
    false_mirrors = (
        "src/sc_neurocore/accel/go/services/bioware/__init__.py",
        "src/sc_neurocore/accel/go/services/bioware/bioware.go",
        "src/sc_neurocore/accel/julia/bioware/__init__.py",
        "src/sc_neurocore/accel/julia/bioware/bioware.jl",
        "src/sc_neurocore/accel/mojo/kernels/bioware.mojo",
        "src/sc_neurocore/accel/rust/safety/bioware.rs",
    )

    assert all(not (REPO_ROOT / path).exists() for path in false_mirrors)


def test_rust_safety_registry_does_not_publish_bioware_placeholder() -> None:
    registries = (
        REPO_ROOT / "src/sc_neurocore/accel/rust/safety/mod.rs",
        REPO_ROOT / "src/sc_neurocore/accel/rust/safety/lib.rs",
    )

    assert all("pub mod bioware;" not in path.read_text(encoding="utf-8") for path in registries)
