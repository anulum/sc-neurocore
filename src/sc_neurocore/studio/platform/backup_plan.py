# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio backup planning

"""Backup and restore planning manifest for SC-NeuroCore Studio."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from sc_neurocore.studio.platform.settings import (
    StudioRuntimeSettings,
    build_default_studio_runtime_settings,
)

STUDIO_BACKUP_PLAN_SCHEMA_VERSION = "studio.backup-plan.v1"

StudioBackupTargetKind = Literal["file", "directory"]


@dataclass(frozen=True, slots=True)
class StudioBackupPlanItem:
    """One durable Studio state target that must be backed up.

    Parameters
    ----------
    item_id:
        Stable item identifier for operator automation.
    target_kind:
        Expected filesystem object kind.
    description:
        Operator-facing state description.
    source_label:
        Path-free source label, normally an environment variable name or a
        documented Studio default.
    configured:
        Whether the target is configured for this runtime.
    exists:
        Whether the current target exists.
    required:
        Whether this target is required by the active deployment profile.
    backup_actions:
        Path-free actions for capturing the target.
    restore_actions:
        Path-free actions for restoring the target.
    local_path:
        Optional resolved path. This is emitted only when the operator requests
        local-path disclosure for an internal handoff.
    """

    item_id: str
    target_kind: StudioBackupTargetKind
    description: str
    source_label: str
    configured: bool
    exists: bool
    required: bool
    backup_actions: tuple[str, ...]
    restore_actions: tuple[str, ...]
    local_path: Path | None = None

    def to_public_dict(self, *, include_local_paths: bool = False) -> dict[str, object]:
        """Return a JSON-serializable item payload."""

        payload: dict[str, object] = {
            "backup_actions": list(self.backup_actions),
            "configured": self.configured,
            "description": self.description,
            "exists": self.exists,
            "item_id": self.item_id,
            "required": self.required,
            "restore_actions": list(self.restore_actions),
            "source_label": self.source_label,
            "target_kind": self.target_kind,
        }
        if include_local_paths and self.local_path is not None:
            payload["local_path"] = str(self.local_path)
        return payload


@dataclass(frozen=True, slots=True)
class StudioBackupPlan:
    """Machine-readable Studio backup and restore plan.

    Parameters
    ----------
    deployment_profile:
        Active Studio deployment profile.
    items:
        Durable state targets that an operator backup must capture.
    include_local_paths:
        Whether public serialization should include resolved local paths.
    schema_version:
        Stable schema identifier.
    """

    deployment_profile: str
    items: tuple[StudioBackupPlanItem, ...]
    include_local_paths: bool = False
    schema_version: str = STUDIO_BACKUP_PLAN_SCHEMA_VERSION

    @property
    def missing_required_count(self) -> int:
        """Return the number of required targets that are not configured."""

        return sum(1 for item in self.items if item.required and not item.configured)

    @property
    def missing_existing_count(self) -> int:
        """Return the number of configured targets that do not exist yet."""

        return sum(1 for item in self.items if item.configured and not item.exists)

    @property
    def ready_for_restore_drill(self) -> bool:
        """Return whether all required targets are configured and present."""

        return self.missing_required_count == 0 and self.missing_existing_count == 0

    def to_public_dict(self) -> dict[str, object]:
        """Return a JSON-serializable backup-plan payload."""

        return {
            "deployment_profile": self.deployment_profile,
            "include_local_paths": self.include_local_paths,
            "items": [
                item.to_public_dict(include_local_paths=self.include_local_paths)
                for item in self.items
            ],
            "missing_existing_count": self.missing_existing_count,
            "missing_required_count": self.missing_required_count,
            "ready_for_restore_drill": self.ready_for_restore_drill,
            "schema_version": self.schema_version,
        }


def build_studio_backup_plan(
    settings: StudioRuntimeSettings | None = None,
    *,
    include_local_paths: bool = False,
    project_root: Path | None = None,
) -> StudioBackupPlan:
    """Build the Studio durable-state backup and restore plan.

    Parameters
    ----------
    settings:
        Runtime settings that define identity, audit, and job state locations.
        When omitted, settings are read from the current environment.
    include_local_paths:
        Include resolved local paths in the serialized plan. Keep this disabled
        for deployment logs that may leave the host.
    project_root:
        Optional Studio project workspace root for tests and embedded tools.

    Returns
    -------
    StudioBackupPlan
        Backup and restore manifest for the active Studio runtime profile.
    """

    runtime_settings = (
        build_default_studio_runtime_settings() if settings is None else settings
    )
    required_for_production = runtime_settings.deployment_profile == "production"
    resolved_project_root = (
        Path.home() / ".sc-neurocore" / "studio" / "projects"
        if project_root is None
        else project_root
    ).expanduser().resolve()
    items = (
        _file_item(
            item_id="identity_file",
            description="Studio service-account and browser-user identity store.",
            source_label="SC_NEUROCORE_STUDIO_IDENTITY_FILE",
            configured_path=runtime_settings.identity_file_path,
            required=required_for_production,
            backup_actions=(
                "Capture the configured identity file before account changes.",
                "Store the copy in an encrypted backup target with restricted operator access.",
            ),
            restore_actions=(
                "Restore the identity file before launching Studio.",
                "Set SC_NEUROCORE_STUDIO_IDENTITY_FILE to the restored file location.",
            ),
        ),
        _file_item(
            item_id="audit_log",
            description="Append-only Studio route-policy and identity audit log.",
            source_label="SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH",
            configured_path=runtime_settings.audit_log_path,
            required=required_for_production,
            backup_actions=(
                "Capture the active audit log and retained rotated audit files.",
                "Preserve append order and file metadata for incident review.",
            ),
            restore_actions=(
                "Restore audit files before incident reconstruction or service restart.",
                "Set SC_NEUROCORE_STUDIO_AUDIT_LOG_PATH to the restored active log location.",
            ),
        ),
        _directory_item(
            item_id="job_root",
            description="Studio job records, bounded worker directories, and artifacts.",
            source_label="SC_NEUROCORE_STUDIO_JOB_ROOT",
            configured_path=runtime_settings.job_root_path,
            required=required_for_production,
            backup_actions=(
                "Capture the configured job root after stopping active worker submissions.",
                "Keep manifest-declared artifacts with their SHA-256 metadata.",
            ),
            restore_actions=(
                "Restore the job root before replaying job evidence or serving artifacts.",
                "Set SC_NEUROCORE_STUDIO_JOB_ROOT to the restored directory location.",
            ),
        ),
        _explicit_directory_item(
            item_id="project_workspace",
            description="Saved Studio project JSON workspaces.",
            source_label="Studio project workspace default",
            path=resolved_project_root,
            required=False,
            backup_actions=(
                "Capture saved project JSON files during regular Studio state backups.",
                "Exclude virtual environments, build trees, and dependency caches.",
            ),
            restore_actions=(
                "Restore project JSON files before opening saved Studio workspaces.",
                "Verify project names through the Studio project list endpoint after restore.",
            ),
        ),
    )
    return StudioBackupPlan(
        deployment_profile=runtime_settings.deployment_profile,
        include_local_paths=include_local_paths,
        items=items,
    )


def _file_item(
    *,
    item_id: str,
    description: str,
    source_label: str,
    configured_path: str | None,
    required: bool,
    backup_actions: tuple[str, ...],
    restore_actions: tuple[str, ...],
) -> StudioBackupPlanItem:
    path = _configured_path(configured_path)
    return StudioBackupPlanItem(
        item_id=item_id,
        target_kind="file",
        description=description,
        source_label=source_label,
        configured=path is not None,
        exists=path.is_file() if path is not None else False,
        required=required,
        backup_actions=backup_actions,
        restore_actions=restore_actions,
        local_path=path,
    )


def _directory_item(
    *,
    item_id: str,
    description: str,
    source_label: str,
    configured_path: str | None,
    required: bool,
    backup_actions: tuple[str, ...],
    restore_actions: tuple[str, ...],
) -> StudioBackupPlanItem:
    path = _configured_path(configured_path)
    return StudioBackupPlanItem(
        item_id=item_id,
        target_kind="directory",
        description=description,
        source_label=source_label,
        configured=path is not None,
        exists=path.is_dir() if path is not None else False,
        required=required,
        backup_actions=backup_actions,
        restore_actions=restore_actions,
        local_path=path,
    )


def _explicit_directory_item(
    *,
    item_id: str,
    description: str,
    source_label: str,
    path: Path,
    required: bool,
    backup_actions: tuple[str, ...],
    restore_actions: tuple[str, ...],
) -> StudioBackupPlanItem:
    resolved_path = path.expanduser().resolve()
    return StudioBackupPlanItem(
        item_id=item_id,
        target_kind="directory",
        description=description,
        source_label=source_label,
        configured=True,
        exists=resolved_path.is_dir(),
        required=required,
        backup_actions=backup_actions,
        restore_actions=restore_actions,
        local_path=resolved_path,
    )


def _configured_path(configured_path: str | None) -> Path | None:
    if configured_path is None:
        return None
    return Path(configured_path).expanduser().resolve()


__all__ = [
    "STUDIO_BACKUP_PLAN_SCHEMA_VERSION",
    "StudioBackupPlan",
    "StudioBackupPlanItem",
    "StudioBackupTargetKind",
    "build_studio_backup_plan",
]
