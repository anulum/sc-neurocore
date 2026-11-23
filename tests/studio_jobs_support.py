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
# Top-level classes and functions each module owns. A name is defined in one
# module only, and the facade re-exports without defining anything.
JOBS_OWNED_DEFINITIONS: dict[str, tuple[str, ...]] = {
    JOBS_FACADE: (),
    f"{JOBS_PACKAGE}.jobs_worker_registration": (
        "start_worker_registration",
        "await_worker_registration",
    ),
    f"{JOBS_PACKAGE}.jobs_process_state": (
        "process_exited",
        "group_survivors",
        "group_is_gone",
    ),
    f"{JOBS_PACKAGE}.jobs_purge_paths": (
        "sync_directory",
        "move_without_replace",
    ),
    f"{JOBS_PACKAGE}.jobs_worker_guard": (
        "_await_ready",
        "arm_worker_guard",
        "_stop_group",
        "main",
    ),
    f"{JOBS_PACKAGE}.jobs_purge_recovery": (
        "PurgeCustody",
        "_matches",
        "_clear_directory",
        "_remove_owned_directory",
        "_recover_phase",
        "recover_purges",
    ),
    f"{JOBS_PACKAGE}.jobs_purge": (
        "_forget_purged_job",
        "purge_terminal_job",
    ),
    f"{JOBS_PACKAGE}.jobs_worker_recovery": ("worker_group_stopped",),
    f"{JOBS_PACKAGE}.jobs_worker_custody": ("register_worker",),
    f"{JOBS_PACKAGE}.jobs_admission_recovery": ("recover_stopped_reservations",),
    f"{JOBS_PACKAGE}.jobs_ledger_creation": ("create_job",),
    f"{JOBS_PACKAGE}.jobs_admission_schema": (
        "migrate_purge_journal",
        "migrate_purge_phases",
        "migrate_worker_custody",
        "migrate_admission",
        "migrate_storage_admission_replay",
    ),
    f"{JOBS_PACKAGE}.jobs_shared_admission": ("SharedJobAdmission",),
    f"{JOBS_PACKAGE}.jobs_process_results": (
        "_ProcessWorkerResult",
        "_load_process_result",
        "_load_process_artifacts",
        "_parse_process_result",
        "_parse_process_artifacts",
    ),
    f"{JOBS_PACKAGE}.jobs_admission": (
        "StudioJobQueueFull",
        "AdmissionSnapshot",
        "StudioJobAdmission",
    ),
    f"{JOBS_PACKAGE}.jobs_context": ("StudioJobContext",),
    f"{JOBS_PACKAGE}.jobs_ledger": ("StudioJobLedger",),
    f"{JOBS_PACKAGE}.jobs_ledger_reads": (
        "read_pending_purge_count",
        "read_purge_snapshot",
        "read_record",
        "read_records",
        "read_transitions",
        "read_live_rows",
    ),
    f"{JOBS_PACKAGE}.jobs_ledger_recovery": (
        "StudioJobReconciliation",
        "_expired",
        "reconcile_ledger",
    ),
    f"{JOBS_PACKAGE}.jobs_ledger_schema": (
        "StudioJobSubmission",
        "migrate",
        "migrate_training_event_data",
    ),
    f"{JOBS_PACKAGE}.jobs_ledger_rows": (
        "StudioJobLedgerCorrupt",
        "json_or_none",
        "artifacts_from_json",
        "artifacts_to_json",
        "record_from_row",
        "training_config_from_json",
        "_validated_training_config",
    ),
    f"{JOBS_PACKAGE}.jobs_ledger_supervisor": (
        "supervisor_identity",
        "_process_start_token",
        "supervisor_is_alive",
    ),
    f"{JOBS_PACKAGE}.jobs_ledger_writes": (
        "transition_job",
        "heartbeat_job",
        "require_purgeable",
        "delete_job",
    ),
    f"{JOBS_PACKAGE}.jobs_manager": ("StudioJobManager",),
    f"{JOBS_PACKAGE}.jobs_manager_access": (
        "_cancel_job",
        "_wait_for_job",
        "_get_job_record",
        "_list_job_records",
        "_list_job_snapshot",
        "_purge_terminal_job",
        "_commit_supervised_update",
        "_read_declared_artifact",
        "_read_live_artifact",
        "_job_manager_status",
    ),
    f"{JOBS_PACKAGE}.jobs_manager_custody": ("StudioJobCustody",),
    f"{JOBS_PACKAGE}.jobs_manager_process": (
        "_submit_process_job",
        "_send_process_control_command",
        "_write_seed_inputs",
    ),
    f"{JOBS_PACKAGE}.jobs_manager_state": ("_StudioJobManagerState",),
    f"{JOBS_PACKAGE}.jobs_manager_supervision": ("StudioJobSupervision",),
    f"{JOBS_PACKAGE}.jobs_manager_thread": (
        "_submit_thread_job",
        "_run_thread_supervised",
    ),
    f"{JOBS_PACKAGE}.jobs_models": (
        "StudioJobRejected",
        "StudioJobRefused",
        "StudioJobConflict",
        "StudioJobCancelled",
        "StudioJobArtifactUnavailable",
        "StudioJobArtifactRefused",
        "StudioJobArtifact",
        "StudioJobRecord",
        "StudioJobResourceProfile",
        "StudioJobStatusSnapshot",
        "StudioJobListSnapshot",
        "StudioJobArtifactPayload",
        "StudioJobPurgeRecord",
        "StudioJobPurgeSnapshot",
    ),
    f"{JOBS_PACKAGE}.jobs_paths": (
        "_find_artifact",
        "_normalize_artifact_lookup_path",
        "_resolve_job_directory",
        "_resolve_confined_nested_child",
        "_relative_path_candidate",
        "_is_confined_path",
        "_resolve_confined_child",
    ),
    f"{JOBS_PACKAGE}.jobs_process_protocol": (
        "_process_worker_environment",
        "_validate_process_task_path",
        "_json_payload",
        "_terminate_process",
        "_unreaped_error",
        "_run_process_supervised",
    ),
    f"{JOBS_PACKAGE}.jobs_reaper": (
        "ReapReport",
        "process_group_of",
        "_signal_group",
        "_wait_for_group",
        "reap_process_group",
        "_collect",
        "_terminate_direct_child",
    ),
}
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
    "JOBS_OWNED_DEFINITIONS",
    "EXPECTED_JOBS_EXPORTS",
    "implementation_import_graph",
    "assert_acyclic",
    "ast",
    "Path",
]
