# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio local job sandbox facade

"""Stable historical imports for the modular Studio job sandbox."""

from sc_neurocore.studio.platform.jobs_context import StudioJobContext
from sc_neurocore.studio.platform.jobs_manager import StudioJobManager
from sc_neurocore.studio.platform.jobs_models import (
    DEFAULT_STUDIO_JOB_MAX_ARTIFACT_BYTES,
    JOBS_LIST_SCHEMA_VERSION,
    JOBS_STATUS_SCHEMA_VERSION,
    STUDIO_CONTROL_COMMAND_FILE as STUDIO_CONTROL_COMMAND_FILE,
    STUDIO_CONTROL_DIR as STUDIO_CONTROL_DIR,
    STUDIO_CONTROL_SEED_DIR as STUDIO_CONTROL_SEED_DIR,
    STUDIO_JOB_ID_PATTERN as STUDIO_JOB_ID_PATTERN,
    STUDIO_SEED_INPUT_DIR as STUDIO_SEED_INPUT_DIR,
    UTC as UTC,
    JsonValue as JsonValue,
    StudioJobArtifact,
    StudioJobArtifactPayload,
    StudioJobArtifactUnavailable,
    StudioJobCancelled,
    StudioJobExecutionModel,
    StudioJobListSnapshot,
    StudioJobRecord,
    StudioJobRejected,
    StudioJobResourceProfile,
    StudioJobStatus,
    StudioJobStatusSnapshot,
    StudioJobTask,
    StudioProcessJobPayload,
)
from sc_neurocore.studio.platform.jobs_paths import (
    _find_artifact as _find_artifact,
    _is_confined_path as _is_confined_path,
    _normalize_artifact_lookup_path as _normalize_artifact_lookup_path,
    _relative_path_candidate as _relative_path_candidate,
    _resolve_confined_child as _resolve_confined_child,
    _resolve_confined_nested_child as _resolve_confined_nested_child,
    _resolve_job_directory as _resolve_job_directory,
)
from sc_neurocore.studio.platform.jobs_process_protocol import (
    _ProcessWorkerResult as _ProcessWorkerResult,
    _json_payload as _json_payload,
    _load_process_artifacts as _load_process_artifacts,
    _load_process_result as _load_process_result,
    _parse_process_artifacts as _parse_process_artifacts,
    _parse_process_result as _parse_process_result,
    _process_worker_environment as _process_worker_environment,
    _terminate_process as _terminate_process,
    _validate_process_task_path as _validate_process_task_path,
)

_PUBLIC_CLASSES = (
    StudioJobRejected,
    StudioJobCancelled,
    StudioJobArtifactUnavailable,
    StudioJobArtifact,
    StudioJobRecord,
    StudioJobResourceProfile,
    StudioJobStatusSnapshot,
    StudioJobListSnapshot,
    StudioJobArtifactPayload,
    StudioJobContext,
    StudioJobManager,
)
for _public_class in _PUBLIC_CLASSES:
    _public_class.__module__ = __name__
del _public_class

__all__ = [
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
]
