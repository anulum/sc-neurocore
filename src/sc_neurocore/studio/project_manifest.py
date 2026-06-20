# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio project save manifests

"""Path-free save manifests for SC-NeuroCore Studio projects."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

STUDIO_PROJECT_SAVE_SCHEMA_VERSION = "studio.project-save.v1"

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


@dataclass(frozen=True, slots=True)
class StudioProjectSaveManifest:
    """Path-free metadata returned after persisting a Studio project.

    Parameters
    ----------
    name:
        Sanitized project name.
    saved_at:
        Unix timestamp stored in the project payload.
    version:
        Studio project payload version.
    state_sha256:
        SHA-256 digest of the canonical project state JSON.
    project_sha256:
        SHA-256 digest of the canonical full project payload JSON.
    evidence_classification:
        Stable evidence lane label for saved project workspaces.
    """

    name: str
    saved_at: float
    version: str
    state_sha256: str
    project_sha256: str
    evidence_classification: str = "project_workspace"

    def to_public_dict(self) -> dict[str, JsonValue]:
        """Return the public, path-free project save response."""

        return {
            "evidence_classification": self.evidence_classification,
            "name": self.name,
            "project_sha256": self.project_sha256,
            "saved_at": self.saved_at,
            "schema_version": STUDIO_PROJECT_SAVE_SCHEMA_VERSION,
            "state_sha256": self.state_sha256,
            "version": self.version,
        }


def build_project_save_manifest(
    *,
    name: str,
    saved_at: float,
    version: str,
    state: Mapping[str, Any],
    project_payload: Mapping[str, Any],
) -> StudioProjectSaveManifest:
    """Build digest-backed metadata for a persisted Studio project.

    Parameters
    ----------
    name:
        Sanitized project name.
    saved_at:
        Unix timestamp stored in the project payload.
    version:
        Studio project payload version.
    state:
        Project state object persisted under ``state``.
    project_payload:
        Full persisted project payload.

    Returns
    -------
    StudioProjectSaveManifest
        Path-free metadata suitable for API responses, logs, and evidence
        manifests.

    Raises
    ------
    ValueError
        If the state or payload cannot be encoded as portable JSON.
    """

    return StudioProjectSaveManifest(
        name=name,
        saved_at=saved_at,
        version=version,
        state_sha256=_sha256_json(state),
        project_sha256=_sha256_json(project_payload),
    )


def dump_project_payload(payload: Mapping[str, Any]) -> str:
    """Return the durable JSON representation for a saved project payload.

    The writer keeps the human-readable indentation used by existing Studio
    project files, while rejecting non-standard JSON values such as NaN and
    Infinity so saved projects remain portable across runtimes.
    """

    try:
        return json.dumps(payload, indent=2, default=str, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Project payload must be portable JSON.") from exc


def _sha256_json(payload: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 digest over canonical JSON."""

    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            default=str,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Project payload must be portable JSON.") from exc
    return hashlib.sha256(encoded).hexdigest()
