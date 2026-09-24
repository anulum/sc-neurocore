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
    f"{JOBS_PACKAGE}.jobs_worker_registration",
    f"{JOBS_PACKAGE}.jobs_process_state",
    f"{JOBS_PACKAGE}.jobs_purge_paths",
    f"{JOBS_PACKAGE}.jobs_worker_guard",
    f"{JOBS_PACKAGE}.jobs_purge_recovery",
    f"{JOBS_PACKAGE}.jobs_purge",
    f"{JOBS_PACKAGE}.jobs_worker_recovery",
    f"{JOBS_PACKAGE}.jobs_worker_custody",
    f"{JOBS_PACKAGE}.jobs_admission_recovery",
    f"{JOBS_PACKAGE}.jobs_ledger_creation",
    f"{JOBS_PACKAGE}.jobs_admission_schema",
    f"{JOBS_PACKAGE}.jobs_shared_admission",
    f"{JOBS_PACKAGE}.jobs_process_results",
    f"{JOBS_PACKAGE}.jobs_admission",
    f"{JOBS_PACKAGE}.jobs_context",
    f"{JOBS_PACKAGE}.jobs_ledger",
    f"{JOBS_PACKAGE}.jobs_ledger_reads",
    f"{JOBS_PACKAGE}.jobs_ledger_recovery",
    f"{JOBS_PACKAGE}.jobs_ledger_schema",
    f"{JOBS_PACKAGE}.jobs_ledger_rows",
    f"{JOBS_PACKAGE}.jobs_ledger_supervisor",
    f"{JOBS_PACKAGE}.jobs_ledger_writes",
    f"{JOBS_PACKAGE}.jobs_manager",
    f"{JOBS_PACKAGE}.jobs_manager_access",
    f"{JOBS_PACKAGE}.jobs_manager_custody",
    f"{JOBS_PACKAGE}.jobs_manager_process",
    f"{JOBS_PACKAGE}.jobs_manager_state",
    f"{JOBS_PACKAGE}.jobs_manager_supervision",
    f"{JOBS_PACKAGE}.jobs_manager_thread",
    f"{JOBS_PACKAGE}.jobs_models",
    f"{JOBS_PACKAGE}.jobs_paths",
    f"{JOBS_PACKAGE}.jobs_process_protocol",
    f"{JOBS_PACKAGE}.jobs_reaper",
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
        "test_studio_jobs_cancel_race.py",
        "test_studio_jobs_context.py",
        "test_studio_jobs_thread.py",
        "test_studio_jobs_process.py",
        "test_studio_jobs_process_seeds.py",
        "test_studio_jobs_process_control.py",
        "test_studio_jobs_process_failures.py",
        "test_studio_jobs_routes.py",
        "test_studio_jobs_admission.py",
        "test_studio_jobs_capacity_custody.py",
        "test_studio_jobs_capacity_recovery.py",
        "test_studio_jobs_orphan_lifetime.py",
        "test_studio_jobs_admission_startup.py",
        "test_studio_jobs_shared_admission.py",
        "test_studio_jobs_shared_queue.py",
        "test_studio_jobs_worker_registration.py",
        "test_studio_jobs_guard_cleanup.py",
        "test_studio_jobs_worker_recovery.py",
        "test_studio_jobs_purge_recovery.py",
        "test_studio_jobs_purge_retry.py",
        "test_studio_jobs_purge_moves.py",
        "test_studio_jobs_purge_source_identity.py",
        "test_studio_jobs_purge_durability.py",
        "test_studio_jobs_purge_migration.py",
        "test_studio_jobs_purge_commit.py",
        "test_studio_jobs_purge_conflicts.py",
        "test_studio_jobs_admission_migration.py",
        "test_studio_jobs_admission_transactions.py",
        "test_studio_jobs_ledger.py",
        "test_studio_jobs_ledger_artifacts.py",
        "test_studio_jobs_ledger_state.py",
        "test_studio_jobs_peer_cancellation.py",
        "test_studio_jobs_supervisor_recovery.py",
        "test_studio_jobs_ledger_reads.py",
        "test_studio_jobs_ledger_ordering.py",
        "test_studio_jobs_ledger_recovery.py",
        "test_studio_jobs_reaping.py",
        "test_studio_jobs_restart_recovery.py",
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


__all__ = [
    "REPO_ROOT",
    "JOBS_PACKAGE",
    "JOBS_FACADE",
    "JOBS_IMPLEMENTATION_MODULES",
    "JOBS_SOURCE_PATHS",
    "JOBS_TEST_PATHS",
    "EXPECTED_JOBS_EXPORTS",
    "implementation_import_graph",
    "assert_acyclic",
    "ast",
    "Path",
]
