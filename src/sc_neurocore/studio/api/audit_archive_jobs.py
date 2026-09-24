# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quarantine archive process adapters

"""Materialise exported quarantine snapshots through named process tasks."""

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, JsonValue

from sc_neurocore.studio.platform.audit_quarantine_archive import (
    write_studio_audit_quarantine_archive,
    write_studio_audit_quarantine_restore,
)
from sc_neurocore.studio.platform.jobs_context import StudioJobContext


class _ArchiveRequest(BaseModel):
    """Exact JSON envelope carrying the API's already-exported audit snapshot."""

    model_config = ConfigDict(extra="forbid", strict=True)
    quarantine_export: dict[str, JsonValue]


class _RestoreRequest(BaseModel):
    """Exact JSON envelope; domain validation remains with the restore writer."""

    model_config = ConfigDict(extra="forbid", strict=True)
    archive: dict[str, JsonValue]
    manifest: dict[str, JsonValue] | None


def execute_quarantine_archive_task(
    context: StudioJobContext, payload: Mapping[str, object]
) -> dict[str, object]:
    """Write a quarantine export snapshot through the existing archive owner.

    Parameters
    ----------
    context : StudioJobContext
        Registered worker's bounded artefact context.
    payload : mapping
        Exactly ``quarantine_export`` containing the path-free JSON export.
        No audit sink, ledger path, clock override or callable is accepted.

    Returns
    -------
    dict[str, object]
        Existing archive receipt, manifest and summary. Artefact bytes and
        digests are produced by the original writer in this job's directory.

    Raises
    ------
    ValueError
        Envelope, export schema or artefact byte limits are invalid.
    """
    request = _ArchiveRequest.model_validate(dict(payload))
    return dict(
        write_studio_audit_quarantine_archive(
            context, quarantine_export=request.quarantine_export
        ).to_public_dict()
    )


def execute_quarantine_restore_task(
    context: StudioJobContext, payload: Mapping[str, object]
) -> dict[str, object]:
    """Validate and materialise a quarantine archive without updating the live sink.

    Parameters
    ----------
    context : StudioJobContext
        Registered worker's bounded artefact context.
    payload : mapping
        Exactly ``archive`` and ``manifest``; the latter may be null. The
        existing writer revalidates their schema and digest relationship.

    Returns
    -------
    dict[str, object]
        Existing restore receipt naming generated JSONL and manifest artefacts.

    Raises
    ------
    ValueError
        Envelope or archive/manifest validation fails, or artefacts exceed limits.
    """
    request = _RestoreRequest.model_validate(dict(payload))
    return dict(
        write_studio_audit_quarantine_restore(
            context, archive_payload=request.archive, manifest_payload=request.manifest
        ).to_public_dict()
    )
