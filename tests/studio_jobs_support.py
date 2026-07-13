# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio jobs architecture test support

"""Stable file and import-graph fixtures for Studio jobs architecture tests."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
JOBS_PACKAGE = "sc_neurocore.studio.platform"
JOBS_FACADE = f"{JOBS_PACKAGE}.jobs"
JOBS_IMPLEMENTATION_MODULES = (
    f"{JOBS_PACKAGE}.jobs_context",
    f"{JOBS_PACKAGE}.jobs_manager",
    f"{JOBS_PACKAGE}.jobs_manager_access",
    f"{JOBS_PACKAGE}.jobs_manager_process",
    f"{JOBS_PACKAGE}.jobs_manager_state",
    f"{JOBS_PACKAGE}.jobs_manager_thread",
    f"{JOBS_PACKAGE}.jobs_models",
    f"{JOBS_PACKAGE}.jobs_paths",
    f"{JOBS_PACKAGE}.jobs_process_protocol",
)
JOBS_SOURCE_PATHS = tuple(
    REPO_ROOT / "src" / Path(*module_name.split("."))
    for module_name in (JOBS_FACADE, *JOBS_IMPLEMENTATION_MODULES)
)
JOBS_SOURCE_PATHS = tuple(path.with_suffix(".py") for path in JOBS_SOURCE_PATHS)
JOBS_TEST_PATHS = tuple(
    REPO_ROOT / "tests" / filename
    for filename in (
        "test_studio_jobs.py",
        "test_studio_jobs_architecture.py",
        "test_studio_jobs_artifacts.py",
        "test_studio_jobs_context.py",
        "test_studio_jobs_thread.py",
        "test_studio_jobs_process.py",
        "test_studio_jobs_process_control.py",
        "test_studio_jobs_process_failures.py",
        "test_studio_jobs_routes.py",
    )
)
EXPECTED_JOBS_EXPORTS = (
    "JOBS_LIST_SCHEMA_VERSION",
    "JOBS_STATUS_SCHEMA_VERSION",
    "DEFAULT_STUDIO_JOB_MAX_ARTIFACT_BYTES",
    "StudioJobArtifact",
    "StudioJobArtifactPayload",
    "StudioJobArtifactUnavailable",
    "StudioJobCancelled",
    "StudioJobContext",
    "StudioJobExecutionModel",
    "StudioJobListSnapshot",
    "StudioJobManager",
    "StudioJobRecord",
    "StudioJobRejected",
    "StudioJobResourceProfile",
    "StudioJobStatus",
    "StudioJobStatusSnapshot",
    "StudioJobTask",
    "StudioProcessJobPayload",
)


def implementation_import_graph() -> dict[str, set[str]]:
    """Return direct imports between focused Studio jobs implementation modules."""

    graph: dict[str, set[str]] = {module_name: set() for module_name in JOBS_IMPLEMENTATION_MODULES}
    for module_name, path in zip(JOBS_IMPLEMENTATION_MODULES, JOBS_SOURCE_PATHS[1:]):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        type_only_imports = {
            id(child)
            for node in ast.walk(tree)
            if isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
            for child in ast.walk(node)
            if isinstance(child, ast.Import | ast.ImportFrom)
        }
        for node in ast.walk(tree):
            if id(node) in type_only_imports:
                continue
            if isinstance(node, ast.ImportFrom) and node.module in graph:
                graph[module_name].add(node.module)
            elif isinstance(node, ast.Import):
                graph[module_name].update(alias.name for alias in node.names if alias.name in graph)
    return graph


def assert_acyclic(graph: dict[str, set[str]]) -> None:
    """Assert that a direct-import graph admits a complete topological order."""

    remaining = {node: set(dependencies) for node, dependencies in graph.items()}
    while remaining:
        roots = {
            node
            for node, dependencies in remaining.items()
            if not dependencies.intersection(remaining)
        }
        assert roots, f"cyclic Studio jobs imports: {remaining}"
        for root in roots:
            remaining.pop(root)
