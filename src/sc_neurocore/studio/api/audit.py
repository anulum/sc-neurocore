# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio audit and evidence routes

"""Expose audit export, quarantine lifecycle, and evidence-bundle adapters."""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, HTTPException, Query, Request

from sc_neurocore.studio.api.common import _safe
from sc_neurocore.studio.api.runtime import StudioApiContext
from sc_neurocore.studio.api.schemas import (
    StudioAuditQuarantineArchivePurgeRequest,
    StudioAuditQuarantineArchiveRequest,
    StudioAuditQuarantineArchiveRestoreRequest,
    StudioAuditQuarantineArchiveValidateRequest,
    StudioEvidenceBundleRequest,
)
from sc_neurocore.studio.platform import (
    STUDIO_AUDIT_QUARANTINE_RESTORE_OWNER,
    AuditExportValue,
    AuditSinkError,
    JsonlAuditSink,
    StudioJobContext,
    build_studio_audit_quarantine_archive_retention_plan,
    purge_studio_audit_quarantine_archive_prune_candidates,
    validate_studio_audit_quarantine_archive,
    write_studio_audit_quarantine_archive,
    write_studio_audit_quarantine_restore,
    write_studio_evidence_bundle,
)
from sc_neurocore.studio.project import load_project


def build_audit_router(context: StudioApiContext) -> APIRouter:
    """Build the audit and evidence router over shared Studio runtime state."""
    router = APIRouter()
    settings = context.settings
    studio_audit_sink = context.studio_audit_sink
    studio_job_manager = context.studio_job_manager

    @router.get("/api/studio/audit/status")
    def api_studio_audit_status() -> dict[str, bool | int | str | None]:
        """Return path-free health for the configured Studio audit sink."""
        return studio_audit_sink.status().to_public_dict()

    @router.get("/api/studio/audit/export")
    def api_studio_audit_export(limit: int = 100) -> dict[str, AuditExportValue]:
        """Return a bounded, path-free audit export for Studio administrators."""
        if not isinstance(studio_audit_sink, JsonlAuditSink):
            raise HTTPException(status_code=409, detail="audit_export_unavailable")
        if limit < 1 or limit > 1000:
            raise HTTPException(status_code=422, detail="Audit export limit must be 1..1000")
        try:
            return studio_audit_sink.export_recent(limit=limit).to_public_dict()
        except AuditSinkError as exc:
            raise HTTPException(status_code=503, detail="audit_export_failed") from exc

    @router.get("/api/studio/audit/quarantine/export")
    def api_studio_audit_quarantine_export(
        limit: int = 100,
    ) -> dict[str, AuditExportValue]:
        """Return path-free quarantined audit rows for incident handoff."""
        if not isinstance(studio_audit_sink, JsonlAuditSink):
            raise HTTPException(status_code=409, detail="audit_export_unavailable")
        if limit < 1 or limit > 1000:
            raise HTTPException(
                status_code=422,
                detail="Audit quarantine export limit must be 1..1000",
            )
        try:
            return studio_audit_sink.export_quarantine(limit=limit).to_public_dict()
        except AuditSinkError as exc:
            raise HTTPException(
                status_code=503,
                detail="audit_quarantine_export_failed",
            ) from exc

    @router.post("/api/studio/audit/quarantine/archive")
    def api_studio_audit_quarantine_archive(
        archive_request: StudioAuditQuarantineArchiveRequest,
        request: Request,
    ) -> dict[str, object]:
        """Write quarantined audit rows into a confined Studio evidence job."""
        if not isinstance(studio_audit_sink, JsonlAuditSink):
            raise HTTPException(status_code=409, detail="audit_export_unavailable")
        try:
            quarantine_export = studio_audit_sink.export_quarantine(
                limit=archive_request.limit
            ).to_public_dict()
        except AuditSinkError as exc:
            raise HTTPException(
                status_code=503,
                detail="audit_quarantine_export_failed",
            ) from exc
        request_id = getattr(request.state, "studio_request_id", None)

        def task(context: StudioJobContext) -> dict[str, object]:
            result = write_studio_audit_quarantine_archive(
                context,
                quarantine_export=quarantine_export,
            ).to_public_dict()
            return cast(dict[str, object], result)

        submitted = studio_job_manager.submit(
            kind="evidence",
            owner="studio-audit-quarantine",
            request_id=request_id if isinstance(request_id, str) else None,
            task=task,
        )
        completed = studio_job_manager.wait(
            submitted.job_id,
            timeout_seconds=settings.job_default_timeout_seconds + 1.0,
        )
        if completed.status == "completed" and completed.result is not None:
            result = dict(completed.result)
            result["job_id"] = completed.job_id
            result["artifacts"] = [artifact.to_public_dict() for artifact in completed.artifacts]
            return result
        if completed.status in {"pending", "running", "cancelling"}:
            raise HTTPException(status_code=503, detail="studio_job_wait_exceeded")
        if completed.status == "timed_out":
            raise HTTPException(status_code=504, detail="studio_job_timed_out")
        raise HTTPException(status_code=500, detail="studio_job_failed")

    @router.post("/api/studio/audit/quarantine/archive/validate")
    def api_studio_audit_quarantine_archive_validate(
        validate_request: StudioAuditQuarantineArchiveValidateRequest,
    ) -> dict[str, object]:
        """Validate a path-free quarantine archive before import handling."""
        result = validate_studio_audit_quarantine_archive(
            validate_request.archive,
            manifest_payload=validate_request.manifest,
        ).to_public_dict()
        return cast(dict[str, object], result)

    @router.get("/api/studio/audit/quarantine/archive/retention")
    def api_studio_audit_quarantine_archive_retention(
        retain_latest: int = Query(default=10, ge=1, le=1000),
    ) -> dict[str, object]:
        """Return a path-free quarantine archive retention plan."""
        result = build_studio_audit_quarantine_archive_retention_plan(
            studio_job_manager.list_records(),
            retain_latest=retain_latest,
        ).to_public_dict()
        return cast(dict[str, object], result)

    @router.post("/api/studio/audit/quarantine/archive/restore")
    def api_studio_audit_quarantine_archive_restore(
        restore_request: StudioAuditQuarantineArchiveRestoreRequest,
        request: Request,
    ) -> dict[str, object]:
        """Materialize a validated archive as confined restore artifacts."""
        validation = validate_studio_audit_quarantine_archive(
            restore_request.archive,
            manifest_payload=restore_request.manifest,
        )
        if not validation.valid:
            raise HTTPException(
                status_code=422,
                detail={
                    "errors": list(validation.errors),
                    "schema_version": validation.schema_version,
                },
            )
        request_id = getattr(request.state, "studio_request_id", None)

        def task(context: StudioJobContext) -> dict[str, object]:
            result = write_studio_audit_quarantine_restore(
                context,
                archive_payload=restore_request.archive,
                manifest_payload=restore_request.manifest,
            ).to_public_dict()
            return cast(dict[str, object], result)

        submitted = studio_job_manager.submit(
            kind="evidence",
            owner=STUDIO_AUDIT_QUARANTINE_RESTORE_OWNER,
            request_id=request_id if isinstance(request_id, str) else None,
            task=task,
        )
        completed = studio_job_manager.wait(
            submitted.job_id,
            timeout_seconds=settings.job_default_timeout_seconds + 1.0,
        )
        if completed.status == "completed" and completed.result is not None:
            result = dict(completed.result)
            result["job_id"] = completed.job_id
            result["artifacts"] = [artifact.to_public_dict() for artifact in completed.artifacts]
            return result
        if completed.status in {"pending", "running", "cancelling"}:
            raise HTTPException(status_code=503, detail="studio_job_wait_exceeded")
        if completed.status == "timed_out":
            raise HTTPException(status_code=504, detail="studio_job_timed_out")
        raise HTTPException(status_code=500, detail="studio_job_failed")

    @router.post("/api/studio/audit/quarantine/archive/purge")
    def api_studio_audit_quarantine_archive_purge(
        purge_request: StudioAuditQuarantineArchivePurgeRequest,
    ) -> dict[str, object]:
        """Purge archive jobs marked as retention prune candidates."""
        result = purge_studio_audit_quarantine_archive_prune_candidates(
            studio_job_manager.list_records(),
            purge_job=studio_job_manager.purge_terminal_record,
            retain_latest=purge_request.retain_latest,
        ).to_public_dict()
        return cast(dict[str, object], result)

    @router.post("/api/studio/evidence/bundle")
    def api_studio_evidence_bundle(
        export_request: StudioEvidenceBundleRequest,
        request: Request,
    ) -> dict[str, object]:
        """Create a path-confined evidence bundle as a Studio worker artifact."""
        project_payload: dict[str, Any] | None = None
        if export_request.project_name is not None:
            project_name = export_request.project_name
            loaded_project = _safe(lambda: load_project(project_name))
            if "error" in loaded_project:
                raise HTTPException(status_code=404, detail=loaded_project["error"])
            project_payload = loaded_project
        job_records = []
        for job_id in export_request.job_ids:
            try:
                job_records.append(studio_job_manager.record(job_id))
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="job_not_found") from exc
        audit_export: dict[str, AuditExportValue] | None = None
        if export_request.include_audit:
            if not isinstance(studio_audit_sink, JsonlAuditSink):
                raise HTTPException(status_code=409, detail="audit_export_unavailable")
            try:
                audit_export = studio_audit_sink.export_recent(
                    limit=export_request.audit_limit
                ).to_public_dict()
            except AuditSinkError as exc:
                raise HTTPException(status_code=503, detail="audit_export_failed") from exc

        request_id = getattr(request.state, "studio_request_id", None)

        def task(context: StudioJobContext) -> dict[str, object]:
            result = write_studio_evidence_bundle(
                context,
                project_payload=project_payload,
                simulation_payloads=tuple(export_request.simulation_results),
                analysis_payloads=tuple(export_request.analysis_results),
                model_scan_payloads=tuple(export_request.model_scan_results),
                weight_restore_payloads=tuple(export_request.weight_restore_results),
                weight_restore_attach_payloads=tuple(export_request.weight_restore_attach_results),
                default_flow_runs=tuple(export_request.default_flow_runs),
                default_flow_attestations=tuple(export_request.default_flow_attestations),
                job_records=tuple(job_records),
                artifact_reader=studio_job_manager.read_artifact,
                audit_export=audit_export,
                command_replay=export_request.command_replay,
            ).to_public_dict()
            return cast(dict[str, object], result)

        submitted = studio_job_manager.submit(
            kind="evidence",
            owner="studio-evidence",
            request_id=request_id if isinstance(request_id, str) else None,
            task=task,
        )
        completed = studio_job_manager.wait(
            submitted.job_id,
            timeout_seconds=settings.job_default_timeout_seconds + 1.0,
        )
        if completed.status == "completed" and completed.result is not None:
            result = dict(completed.result)
            result["job_id"] = completed.job_id
            result["artifacts"] = [artifact.to_public_dict() for artifact in completed.artifacts]
            return result
        if completed.status in {"pending", "running", "cancelling"}:
            raise HTTPException(status_code=503, detail="studio_job_wait_exceeded")
        if completed.status == "timed_out":
            raise HTTPException(status_code=504, detail="studio_job_timed_out")
        raise HTTPException(status_code=500, detail="studio_job_failed")

    return router
