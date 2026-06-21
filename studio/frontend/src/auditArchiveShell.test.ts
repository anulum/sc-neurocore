import { describe, expect, it } from "vitest";

import type {
  StudioAuditExport,
  StudioAuditQuarantineArchivePurgeResult,
  StudioAuditQuarantineArchiveResult,
  StudioAuditQuarantineArchiveRestoreResult,
  StudioAuditQuarantineArchiveRetentionEntry,
  StudioAuditQuarantineArchiveRetentionPlan,
  StudioAuditQuarantineArchiveSummary,
  StudioAuditQuarantineArchiveValidation,
  StudioAuditStatus,
  StudioJobListResponse,
  StudioJobStatus,
  StudioOperatorStatus,
} from "./api/client";
import {
  auditArchiveCreatedState,
  auditArchivePurgedState,
  auditArchiveRestoredState,
  auditArchiveRetentionLoadedState,
  auditArchiveValidationLoadedState,
} from "./auditShell";

function auditStatus(overrides: Partial<StudioAuditStatus> = {}): StudioAuditStatus {
  return {
    configured: overrides.configured ?? true,
    healthy: overrides.healthy ?? true,
    last_error: overrides.last_error ?? null,
    path_configured: overrides.path_configured ?? true,
    sink_type: overrides.sink_type ?? "jsonl",
  };
}

function auditExport(overrides: Partial<StudioAuditExport> = {}): StudioAuditExport {
  return {
    configured: overrides.configured ?? true,
    event_count: overrides.event_count ?? 1,
    events: overrides.events ?? [],
    schema_version: overrides.schema_version ?? "studio.audit.export.v1",
    sink_type: overrides.sink_type ?? "jsonl",
    truncated: overrides.truncated ?? false,
  };
}

function jobStatus(overrides: Partial<StudioJobStatus> = {}): StudioJobStatus {
  return {
    active_count: overrides.active_count ?? 1,
    allowed_kinds: overrides.allowed_kinds ?? ["audit_quarantine_archive"],
    completed_count: overrides.completed_count ?? 2,
    configured: overrides.configured ?? true,
    failed_count: overrides.failed_count ?? 0,
    process_count: overrides.process_count ?? 0,
    resource_profiles: overrides.resource_profiles ?? [],
    schema_version: overrides.schema_version ?? "studio.jobs.status.v1",
    thread_count: overrides.thread_count ?? 1,
    timed_out_count: overrides.timed_out_count ?? 0,
  };
}

function jobList(overrides: Partial<StudioJobListResponse> = {}): StudioJobListResponse {
  return {
    jobs: overrides.jobs ?? [],
    schema_version: overrides.schema_version ?? "studio.jobs.list.v1",
  };
}

function operatorStatus(overrides: Partial<StudioOperatorStatus> = {}): StudioOperatorStatus {
  return {
    audit: overrides.audit ?? auditStatus(),
    browser_login: overrides.browser_login ?? {
      active_bucket_count: 0,
      cooldown_seconds: 60,
      failure_window_seconds: 300,
      locked_bucket_count: 0,
      max_failures: 5,
      max_retry_after_seconds: 0,
    },
    capabilities: overrides.capabilities ?? {
      degraded_count: 0,
      experimental_count: 0,
      healthy_count: 4,
      stable_count: 4,
      total_count: 4,
      unavailable_count: 0,
    },
    deployment_profile: overrides.deployment_profile ?? "production",
    identity: overrides.identity ?? {
      configured: true,
      header_principal_allowed: false,
      mode: "service_account",
    },
    jobs: overrides.jobs ?? jobStatus(),
    resource_limits: overrides.resource_limits ?? {
      eda_process_cpu_seconds: null,
      eda_process_limits_supported: false,
      eda_process_memory_bytes: null,
      job_default_timeout_seconds: 600,
      job_max_artifact_bytes: 1048576,
    },
    route_policies: overrides.route_policies ?? {
      admin_count: 2,
      authenticated_count: 5,
      enforced: true,
      protected_audit_action_count: 2,
      protected_count: 7,
      protected_routes_audited: true,
      public_count: 3,
      total_count: 10,
    },
    schema_version: overrides.schema_version ?? "studio.operator.status.v1",
  };
}

function archiveSummary(
  overrides: Partial<StudioAuditQuarantineArchiveSummary> = {},
): StudioAuditQuarantineArchiveSummary {
  return {
    archive_artifact_count: overrides.archive_artifact_count ?? 2,
    event_count: overrides.event_count ?? 6,
    quarantine_reason: overrides.quarantine_reason ?? "retention",
    reason_counts: overrides.reason_counts ?? { retention: 6 },
    retained_event_count: overrides.retained_event_count ?? 6,
    source_schema_version: overrides.source_schema_version ?? "studio.audit.export.v1",
    truncated: overrides.truncated ?? false,
  };
}

function archiveResult(
  overrides: Partial<StudioAuditQuarantineArchiveResult> = {},
): StudioAuditQuarantineArchiveResult {
  return {
    archive_id: overrides.archive_id ?? "archive-1",
    artifact_paths: overrides.artifact_paths ?? ["audit/archive.json"],
    artifacts: overrides.artifacts ?? [],
    job_id: overrides.job_id ?? "job-archive",
    manifest: overrides.manifest ?? { archive_id: "archive-1" },
    schema_version: overrides.schema_version ?? "studio.audit.quarantine-archive.v1",
    summary: overrides.summary ?? archiveSummary(),
  };
}

function archiveRetentionEntry(
  overrides: Partial<StudioAuditQuarantineArchiveRetentionEntry> = {},
): StudioAuditQuarantineArchiveRetentionEntry {
  return {
    archive_id: overrides.archive_id ?? "archive-1",
    artifact_paths: overrides.artifact_paths ?? ["audit/archive.json"],
    created_at_utc: overrides.created_at_utc ?? "2026-06-21T12:00:00Z",
    disposition: overrides.disposition ?? "retain",
    event_count: overrides.event_count ?? 6,
    finished_at_utc: overrides.finished_at_utc ?? "2026-06-21T12:00:10Z",
    job_id: overrides.job_id ?? "job-archive",
    retained_event_count: overrides.retained_event_count ?? 6,
    summary: overrides.summary ?? archiveSummary(),
  };
}

function retentionPlan(
  overrides: Partial<StudioAuditQuarantineArchiveRetentionPlan> = {},
): StudioAuditQuarantineArchiveRetentionPlan {
  return {
    archive_count: overrides.archive_count ?? 1,
    entries: overrides.entries ?? [archiveRetentionEntry()],
    prune_candidate_count: overrides.prune_candidate_count ?? 0,
    retain_count: overrides.retain_count ?? 1,
    retain_latest: overrides.retain_latest ?? 10,
    schema_version: overrides.schema_version ?? "studio.audit.quarantine-retention.v1",
    skipped_record_count: overrides.skipped_record_count ?? 0,
  };
}

function archiveValidation(
  overrides: Partial<StudioAuditQuarantineArchiveValidation> = {},
): StudioAuditQuarantineArchiveValidation {
  return {
    archive_id: overrides.archive_id ?? "archive-1",
    errors: overrides.errors ?? [],
    schema_version: overrides.schema_version ?? "studio.audit.quarantine-validation.v1",
    summary: overrides.summary ?? archiveSummary(),
    valid: overrides.valid ?? true,
    warnings: overrides.warnings ?? [],
  };
}

function archivePurge(
  overrides: Partial<StudioAuditQuarantineArchivePurgeResult> = {},
): StudioAuditQuarantineArchivePurgeResult {
  return {
    purged_archive_count: overrides.purged_archive_count ?? 1,
    purged_entries: overrides.purged_entries ?? [
      archiveRetentionEntry({ archive_id: "archive-old", disposition: "prune_candidate" }),
    ],
    retained_archive_count: overrides.retained_archive_count ?? 1,
    retained_entries: overrides.retained_entries ?? [archiveRetentionEntry()],
    retain_latest: overrides.retain_latest ?? 10,
    schema_version: overrides.schema_version ?? "studio.audit.quarantine-purge.v1",
    skipped_record_count: overrides.skipped_record_count ?? 0,
  };
}

function archiveRestore(
  overrides: Partial<StudioAuditQuarantineArchiveRestoreResult> = {},
): StudioAuditQuarantineArchiveRestoreResult {
  return {
    archive_id: overrides.archive_id ?? "archive-restore",
    artifact_paths: overrides.artifact_paths ?? ["audit/restore.json"],
    artifacts: overrides.artifacts ?? [],
    job_id: overrides.job_id ?? "job-restore",
    manifest: overrides.manifest ?? { archive_id: "archive-restore" },
    schema_version: overrides.schema_version ?? "studio.audit.quarantine-restore.v1",
    summary: overrides.summary ?? {
      ...archiveSummary(),
      restore_artifact_count: 2,
      restored_at_utc: "2026-06-21T12:10:00Z",
    },
  };
}

describe("audit archive shell state contract", () => {
  it("builds archive creation patches with refreshed operator state", () => {
    const archive = archiveResult();
    const exported = auditExport();
    const operator = operatorStatus({
      audit: auditStatus({ healthy: false, last_error: "archive warning" }),
      jobs: jobStatus({ completed_count: 3 }),
    });
    const jobs = jobList();

    expect(auditArchiveCreatedState(archive, exported, operator, jobs)).toEqual({
      auditArchive: archive,
      auditError: null,
      auditExport: exported,
      auditLoading: false,
      auditStatus: operator.audit,
      jobRecords: jobs.jobs,
      jobStatus: operator.jobs,
      operatorStatus: operator,
    });
  });

  it("builds archive retention and validation patches", () => {
    const plan = retentionPlan();
    const validation = archiveValidation({ warnings: ["extra manifest key ignored"] });

    expect(auditArchiveRetentionLoadedState(plan)).toEqual({
      auditArchiveRetention: plan,
      auditError: null,
      auditLoading: false,
    });
    expect(auditArchiveValidationLoadedState(validation)).toEqual({
      auditArchiveValidation: validation,
      auditError: null,
      auditLoading: false,
    });
  });

  it("builds archive restore and purge patches with refreshed operator state", () => {
    const restored = archiveRestore();
    const purged = archivePurge();
    const plan = retentionPlan({ prune_candidate_count: 1 });
    const operator = operatorStatus({ jobs: jobStatus({ completed_count: 4 }) });
    const jobs = jobList();

    expect(auditArchiveRestoredState(restored, operator, jobs)).toEqual({
      auditArchiveRestore: restored,
      auditArchiveValidation: null,
      auditError: null,
      auditLoading: false,
      auditStatus: operator.audit,
      jobRecords: jobs.jobs,
      jobStatus: operator.jobs,
      operatorStatus: operator,
    });
    expect(auditArchivePurgedState(purged, plan, operator, jobs)).toEqual({
      auditArchivePurge: purged,
      auditArchiveRetention: plan,
      auditError: null,
      auditLoading: false,
      auditStatus: operator.audit,
      jobRecords: jobs.jobs,
      jobStatus: operator.jobs,
      operatorStatus: operator,
    });
  });
});
