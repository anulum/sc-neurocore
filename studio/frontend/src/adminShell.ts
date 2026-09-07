// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

/**
 * The Admin panel's whole state, derived in one pass from what the server said.
 *
 * Everything an operator reads in that panel is a string built here. The
 * component renders the model and decides nothing: no formatting, no
 * fallbacks, no arithmetic. That is what makes the panel testable without a
 * browser, and it is why every "unknown", "none" and "unavailable" in the
 * interface is a value chosen in this file rather than an empty render.
 *
 * The recurring pattern is that a missing answer and a zero answer must not
 * look alike. A count the server did not send reads `unknown`; a count it sent
 * as zero reads `0`. An operator judging a deployment needs to tell "nothing
 * happened" from "I could not ask", and a panel that renders both as blank
 * takes that distinction away.
 *
 * Where the operator status carries a block the older panel fields also carry
 * — audit health, job counts, capability counts — the operator status wins and
 * the separate fetch is the fallback, because the operator route answers them
 * all from one consistent snapshot.
 */

import type {
  StudioAuditEvent,
  StudioAuditExport,
  StudioAuditQuarantineArchivePurgeResult,
  StudioAuditQuarantineArchiveResult,
  StudioAuditQuarantineArchiveRetentionPlan,
  StudioAuditQuarantineArchiveRestoreResult,
  StudioAuditQuarantineArchiveValidation,
  StudioAuditStatus,
  StudioCapability,
  StudioEvidenceBundleResponse,
  StudioIdentityBrowserUser,
  StudioIdentityServiceAccount,
  StudioJobRecord,
  StudioJobStatus,
  StudioOperatorStatus,
} from "./api/client";
import { summarizeAuditExport } from "./auditShell";
import { buildStudioReadinessModel, type StudioReadinessModel } from "./studioReadiness";

/**
 * Everything the panel is given: each server payload, each error, each loading
 * flag. A payload that has not arrived is `null` rather than absent, so the
 * builder distinguishes "not fetched" from "fetched and empty".
 */
export interface AdminShellInput {
  auditError: string | null;
  auditArchive: StudioAuditQuarantineArchiveResult | null;
  auditArchivePurge: StudioAuditQuarantineArchivePurgeResult | null;
  auditArchiveRetention: StudioAuditQuarantineArchiveRetentionPlan | null;
  auditArchiveRestore: StudioAuditQuarantineArchiveRestoreResult | null;
  auditArchiveValidation: StudioAuditQuarantineArchiveValidation | null;
  auditExport: StudioAuditExport | null;
  auditStatus: StudioAuditStatus | null;
  capabilities: StudioCapability[];
  evidenceBundle: StudioEvidenceBundleResponse | null;
  evidenceBundleError: string | null;
  evidenceBundleLoading: boolean;
  jobRecords: StudioJobRecord[];
  jobStatus: StudioJobStatus | null;
  identityBrowserUsers: StudioIdentityBrowserUser[];
  identityServiceAccounts: StudioIdentityServiceAccount[];
  operatorStatus: StudioOperatorStatus | null;
}

/** The audit row: counts by category, health, and the latest action seen. */
export interface AdminAuditModel {
  denied: number;
  error: string | null;
  browserAuth: number;
  browserAuthAllowed: number;
  browserAuthDenied: number;
  healthLabel: "ready" | "unhealthy";
  identityLifecycle: number;
  identityLifecycleAllowed: number;
  identityLifecycleDenied: number;
  lastError: string | null;
  latestAction: string | null;
  latestBrowserAuthAction: string | null;
  latestIdentityLifecycleAction: string | null;
  sinkType: string;
  total: number;
  truncated: boolean;
}

/** The capability row: how many are registered and how many are unhealthy. */
export interface AdminCapabilityModel {
  registered: number;
  unhealthy: number;
  healthLabel: "ready" | "degraded";
}

/** The job-queue row: counts, configuration, and what the queue accepts. */
export interface AdminJobModel {
  active: number;
  allowedKinds: string;
  completed: number;
  configured: boolean;
  failed: number;
  healthLabel: "ready" | "attention" | "unconfigured";
  processCount: number;
  resourceProfiles: string[];
  threadCount: number;
  timedOut: number;
}

/** One job in the recent-jobs table, with its artefacts already counted. */
export interface AdminJobRecordModel {
  artifactCount: number;
  artifactPaths: string;
  createdAt: string;
  evidenceArtifactCount: number;
  error: string | null;
  executionModel: string;
  finishedAt: string;
  jobId: string;
  kind: string;
  owner: string;
  status: StudioJobRecord["status"];
}

/** The evidence-bundle panel: the bundle, its artefacts and its manifest. */
export interface AdminEvidenceBundleModel {
  artifactCount: number;
  artifacts: AdminEvidenceBundleArtifactModel[];
  bundleId: string;
  entries: AdminEvidenceBundleEntryModel[];
  entryTypes: string;
  error: string | null;
  evidenceClasses: string;
  jobId: string;
  loading: boolean;
  manifestEntryCount: number;
  sourceJobs: string;
}

/**
 * One artefact of a bundle, with both the raw digest and the short label the
 * table shows. Both are kept: the label is for reading, the digest is for
 * comparing.
 */
export interface AdminEvidenceBundleArtifactModel {
  relativePath: string;
  sha256: string;
  sha256Label: string;
  sizeBytes: number | null;
  sizeLabel: string;
}

/** One manifest entry, reduced to the four fields the table displays. */
export interface AdminEvidenceBundleEntryModel {
  classification: string;
  detail: string;
  index: number;
  source: string;
  type: string;
}

/**
 * The audit-archive panel, which shows five independent results at once: the
 * archive just written, the retention plan, the last purge, a validation and a
 * restore. Each has its own "none" so a reader can see which of the five has
 * actually been run.
 */
export interface AdminAuditArchiveModel {
  archiveCount: number;
  archivedEventCount: number;
  archiveId: string;
  artifactCount: number;
  error: string | null;
  latestEntries: AdminAuditArchiveEntryModel[];
  lastPurge: string;
  pruneCandidateCount: number;
  purgedArchiveCount: number;
  reasonCounts: string;
  retainedArchiveCount: number;
  retainCount: number;
  retainLatest: number;
  restoreArchiveId: string;
  restoreArtifactCount: number;
  restoreJobId: string;
  restoreRows: number;
  skippedRecordCount: number;
  validationArchiveId: string;
  validationErrors: string;
  validationStatus: "not checked" | "valid" | "invalid";
  validationWarnings: string;
}

/** One archive in the retention plan, and whether it is a prune candidate. */
export interface AdminAuditArchiveEntryModel {
  archiveId: string;
  disposition: "retain" | "prune_candidate";
  eventCount: number;
  finishedAt: string;
  jobId: string;
  retainedEventCount: number;
}

/** One service account, sorted and labelled for the identity table. */
export interface AdminIdentityAccountModel {
  active: boolean;
  activeLabel: "active" | "disabled";
  expiresAt: string;
  principalId: string;
  rolesText: string;
}

/** One browser user, sorted and labelled for the identity table. */
export interface AdminIdentityBrowserUserModel {
  active: boolean;
  activeLabel: "active" | "disabled";
  expiresAt: string;
  principalId: string;
  rolesText: string;
  username: string;
}

/**
 * The deployment's own settings, every one a string.
 *
 * They are strings because each has an `unknown` state that no number can
 * express, and because the panel shows them beside their units — seconds,
 * bytes, `supported` — rather than raw.
 */
export interface AdminOperatorModel {
  browserLoginActiveBuckets: string;
  browserLoginCooldown: string;
  browserLoginLockedBuckets: string;
  browserLoginLimit: string;
  browserLoginMaxRetryAfter: string;
  browserLoginWindow: string;
  deploymentProfile: "development" | "production" | "unknown";
  edaCpuLimit: string;
  edaMemoryLimit: string;
  edaLimitSupport: "supported" | "unsupported" | "unknown";
  identityMode: string;
  jobArtifactLimit: string;
  jobTimeout: string;
  routePolicyAuditLabel: "audited" | "incomplete" | "unknown";
  routePolicyInventory: string;
  routePolicyLabel: "enforced" | "disabled" | "unknown";
  schemaVersion: string;
}

/** The whole panel: one field per section, all derived, none live. */
export interface AdminShellModel {
  audit: AdminAuditModel;
  auditArchive: AdminAuditArchiveModel;
  capabilities: AdminCapabilityModel;
  evidenceBundle: AdminEvidenceBundleModel;
  jobs: AdminJobModel;
  jobRecords: AdminJobRecordModel[];
  identityBrowserUsers: AdminIdentityBrowserUserModel[];
  identityAccounts: AdminIdentityAccountModel[];
  operator: AdminOperatorModel;
  readiness: StudioReadinessModel;
  recentAuditEvents: StudioAuditEvent[];
  unhealthyCapabilities: StudioCapability[];
}

/**
 * Build the operator-facing Admin panel state from the server's payloads.
 *
 * @param input - Everything that has arrived, and what has not.
 * @returns The whole panel's state.
 */
export function buildAdminShellModel(input: AdminShellInput): AdminShellModel {
  const auditSummary = summarizeAuditExport(input.auditExport);
  const unhealthyCapabilities = input.capabilities.filter((capability) => !capability.healthy);
  const recentAuditEvents = input.auditExport?.events.slice(-8).reverse() ?? [];
  const auditStatus = input.operatorStatus?.audit ?? input.auditStatus;
  const jobStatus = input.operatorStatus?.jobs ?? input.jobStatus;
  const operatorCapabilities = input.operatorStatus?.capabilities;

  return {
    audit: {
      denied: auditSummary.denied,
      error: input.auditError,
      browserAuth: auditSummary.browserAuth,
      browserAuthAllowed: auditSummary.browserAuthAllowed,
      browserAuthDenied: auditSummary.browserAuthDenied,
      healthLabel: auditStatus?.healthy === false ? "unhealthy" : "ready",
      identityLifecycle: auditSummary.identityLifecycle,
      identityLifecycleAllowed: auditSummary.identityLifecycleAllowed,
      identityLifecycleDenied: auditSummary.identityLifecycleDenied,
      lastError: auditStatus?.last_error ?? null,
      latestAction: auditSummary.latestAction,
      latestBrowserAuthAction: auditSummary.latestBrowserAuthAction,
      latestIdentityLifecycleAction: auditSummary.latestIdentityLifecycleAction,
      sinkType: auditStatus?.sink_type ?? auditSummary.sinkType,
      total: auditSummary.total,
      truncated: auditSummary.truncated,
    },
    auditArchive: buildAuditArchiveModel(
      input.auditArchive,
      input.auditArchiveRetention,
      input.auditArchivePurge,
      input.auditArchiveValidation,
      input.auditArchiveRestore,
      input.auditError,
    ),
    capabilities: {
      registered: operatorCapabilities?.total_count ?? input.capabilities.length,
      unhealthy: operatorCapabilities?.unavailable_count ?? unhealthyCapabilities.length,
      healthLabel: (operatorCapabilities?.unavailable_count ?? unhealthyCapabilities.length) === 0
        ? "ready" : "degraded",
    },
    jobs: buildJobModel(jobStatus),
    evidenceBundle: buildEvidenceBundleModel(
      input.evidenceBundle,
      input.evidenceBundleError,
      input.evidenceBundleLoading,
    ),
    jobRecords: buildJobRecords(input.jobRecords),
    identityBrowserUsers: buildIdentityBrowserUsers(input.identityBrowserUsers),
    identityAccounts: buildIdentityAccounts(input.identityServiceAccounts),
    operator: buildOperatorModel(input.operatorStatus),
    readiness: buildStudioReadinessModel(input.operatorStatus),
    recentAuditEvents,
    unhealthyCapabilities,
  };
}

/**
 * Build the audit-archive panel from whichever of its five results exist.
 *
 * @param archive - The archive just written, or `null`.
 * @param retention - The retention plan, or `null`.
 * @param purge - The last purge's result, or `null`.
 * @param validation - The last validation, or `null`.
 * @param restore - The last restore, or `null`.
 * @param error - The audit error to surface, or `null`.
 * @returns The panel's state.
 */
function buildAuditArchiveModel(
  archive: StudioAuditQuarantineArchiveResult | null,
  retention: StudioAuditQuarantineArchiveRetentionPlan | null,
  purge: StudioAuditQuarantineArchivePurgeResult | null,
  validation: StudioAuditQuarantineArchiveValidation | null,
  restore: StudioAuditQuarantineArchiveRestoreResult | null,
  error: string | null,
): AdminAuditArchiveModel {
  const entries = retention?.entries ?? [];
  return {
    archiveCount: retention?.archive_count ?? 0,
    archivedEventCount: archive?.summary.event_count ?? 0,
    archiveId: archive?.archive_id ?? "none",
    artifactCount: archive?.artifact_paths.length ?? 0,
    error,
    latestEntries: entries.slice(0, 5).map((entry) => ({
      archiveId: entry.archive_id,
      disposition: entry.disposition,
      eventCount: entry.event_count,
      finishedAt: entry.finished_at_utc ?? "unfinished",
      jobId: entry.job_id,
      retainedEventCount: entry.retained_event_count,
    })),
    lastPurge: purge === null
      ? "none"
      : `${purge.purged_archive_count} purged / ${purge.retained_archive_count} retained`,
    pruneCandidateCount: retention?.prune_candidate_count ?? 0,
    purgedArchiveCount: purge?.purged_archive_count ?? 0,
    reasonCounts: formatCounts(archive?.summary.reason_counts),
    retainedArchiveCount: purge?.retained_archive_count ?? 0,
    retainCount: retention?.retain_count ?? 0,
    retainLatest: retention?.retain_latest ?? purge?.retain_latest ?? 10,
    restoreArchiveId: restore?.archive_id ?? "none",
    restoreArtifactCount: restore?.summary.restore_artifact_count ?? restore?.artifact_paths.length ?? 0,
    restoreJobId: restore?.job_id ?? "none",
    restoreRows: restore?.summary.event_count ?? 0,
    skippedRecordCount: retention?.skipped_record_count ?? purge?.skipped_record_count ?? 0,
    validationArchiveId: validation?.archive_id ?? "none",
    validationErrors: validation === null ? "none" : validation.errors.join(", ") || "none",
    validationStatus: validation === null ? "not checked" : validation.valid ? "valid" : "invalid",
    validationWarnings: validation === null ? "none" : validation.warnings.join(", ") || "none",
  };
}

/**
 * Build the evidence-bundle panel.
 *
 * The summary's counts are preferred over counting the arrays, because the
 * server counts what it wrote and the browser can only count what it was
 * sent; they differ if a response was truncated.
 *
 * @param evidenceBundle - The bundle, or `null` when none has been built.
 * @param error - The error to surface, or `null`.
 * @param loading - Whether a bundle is being built now.
 * @returns The panel's state.
 */
function buildEvidenceBundleModel(
  evidenceBundle: StudioEvidenceBundleResponse | null,
  error: string | null,
  loading: boolean,
): AdminEvidenceBundleModel {
  const entries = evidenceBundle?.manifest.entries;
  const summary = evidenceBundle?.summary;
  return {
    artifactCount: summary?.artifact_path_count ?? evidenceBundle?.artifact_paths.length ?? 0,
    artifacts: buildEvidenceBundleArtifacts(evidenceBundle),
    bundleId: evidenceBundle?.bundle_id ?? "none",
    entries: buildEvidenceBundleEntries(evidenceBundle),
    entryTypes: formatCounts(summary?.entry_type_counts),
    error,
    evidenceClasses: formatCounts(summary?.evidence_classification_counts),
    jobId: evidenceBundle?.job_id ?? "none",
    loading,
    manifestEntryCount: summary?.entry_count ?? (Array.isArray(entries) ? entries.length : 0),
    sourceJobs: formatSourceJobs(summary?.source_job_count, summary?.source_job_kind_counts),
  };
}

/**
 * List a bundle's artefacts, joining each path to its metadata.
 *
 * The path list is authoritative: an artefact named there but missing from
 * the metadata is still shown, with its digest and size as `unknown`,
 * because hiding it would hide the inconsistency.
 *
 * @param evidenceBundle - The bundle, or `null`.
 * @returns One row per artefact path, in the order the bundle lists them.
 */
function buildEvidenceBundleArtifacts(
  evidenceBundle: StudioEvidenceBundleResponse | null,
): AdminEvidenceBundleArtifactModel[] {
  if (evidenceBundle === null) {
    return [];
  }
  const artifactMetadata = new Map(
    evidenceBundle.artifacts.map((artifact) => [artifact.relative_path, artifact]),
  );
  return evidenceBundle.artifact_paths.map((relativePath) => {
    const artifact = artifactMetadata.get(relativePath);
    const sha256 = artifact?.sha256 ?? "unknown";
    return {
      relativePath,
      sha256,
      sha256Label: sha256 === "unknown" ? "unknown" : sha256.slice(0, 12),
      sizeBytes: artifact?.size_bytes ?? null,
      sizeLabel: artifact === undefined ? "unknown" : formatBytes(artifact.size_bytes),
    };
  });
}

/**
 * List a manifest's entries, skipping anything that is not a record.
 *
 * Manifest entries are open-ended by design -- each evidence class writes
 * its own fields -- so they arrive as unknown records and are read field by
 * field rather than typed.
 *
 * @param evidenceBundle - The bundle, or `null`.
 * @returns One row per readable entry, indexed as the manifest orders them.
 */
function buildEvidenceBundleEntries(
  evidenceBundle: StudioEvidenceBundleResponse | null,
): AdminEvidenceBundleEntryModel[] {
  const entries = evidenceBundle?.manifest.entries;
  if (!Array.isArray(entries)) {
    return [];
  }
  return entries
    .filter(isRecord)
    .map((entry, index) => ({
      classification: textField(entry, "evidence_classification") ?? "unclassified",
      detail: formatEvidenceBundleEntryDetail(entry),
      index,
      source: formatEvidenceBundleEntrySource(entry),
      type: textField(entry, "type") ?? "unknown",
    }));
}

/**
 * Render a count map as `name:count` pairs, alphabetically.
 *
 * Zero counts are dropped: a map listing every possible name with most at
 * zero is unreadable, and the names that matter are the ones with a count.
 *
 * @param counts - The map, or `undefined` when the server sent none.
 * @returns The rendered pairs, or `none`.
 */
function formatCounts(counts: Record<string, number> | undefined): string {
  if (counts === undefined) {
    return "none";
  }
  const parts = Object.entries(counts)
    .filter(([, count]) => count > 0)
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([name, count]) => `${name}:${count}`);
  return parts.length > 0 ? parts.join(", ") : "none";
}

/**
 * Render how many jobs a bundle drew from, and of what kinds.
 *
 * @param sourceJobCount - The count, or `undefined`.
 * @param sourceJobKindCounts - The counts by kind, or `undefined`.
 * @returns The count alone, or the count and its kinds.
 */
function formatSourceJobs(
  sourceJobCount: number | undefined,
  sourceJobKindCounts: Record<string, number> | undefined,
): string {
  const count = sourceJobCount ?? 0;
  const kinds = formatCounts(sourceJobKindCounts);
  return kinds === "none" ? `${count}` : `${count} - ${kinds}`;
}

/**
 * Name where a manifest entry came from.
 *
 * The fields are tried in order of how specific they are: the job that
 * produced it, then what it says about itself, then the route that would
 * replay it, then the path it sits at.
 *
 * @param entry - The manifest entry.
 * @returns The most specific source it carries.
 */
function formatEvidenceBundleEntrySource(entry: Record<string, unknown>): string {
  const sourceJobId = textField(entry, "source_job_id");
  if (sourceJobId !== null) {
    return `job ${sourceJobId}`;
  }
  return textField(entry, "source")
    ?? textField(entry, "replay_route")
    ?? textField(entry, "bundle_path")
    ?? "bundle";
}

/**
 * Say the one further thing worth showing about a manifest entry.
 *
 * @param entry - The manifest entry.
 * @returns Its artefact path, bundle path, replay route or short digest,
 *   whichever it carries first.
 */
function formatEvidenceBundleEntryDetail(entry: Record<string, unknown>): string {
  const artifactPath = textField(entry, "source_job_artifact_path");
  if (artifactPath !== null) {
    return artifactPath;
  }
  const bundlePath = textField(entry, "bundle_path");
  if (bundlePath !== null) {
    return bundlePath;
  }
  const replayRoute = textField(entry, "replay_route");
  if (replayRoute !== null) {
    return replayRoute;
  }
  const sha256 = textField(entry, "sha256");
  if (sha256 !== null) {
    return `sha ${sha256.slice(0, 12)}`;
  }
  return "manifest entry";
}

/**
 * Read a field that must be a non-empty string to count as present.
 *
 * @param entry - The record.
 * @param key - The field.
 * @returns The string, or `null` when it is absent, empty or another type.
 */
function textField(entry: Record<string, unknown>, key: string): string | null {
  const value = entry[key];
  return typeof value === "string" && value.length > 0 ? value : null;
}

/**
 * Whether a value is a plain object.
 *
 * @param value - The value.
 * @returns Whether it is one.
 */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * List browser users by username, without disturbing the caller's array.
 *
 * @param users - The users, in whatever order they arrived.
 * @returns The rows, sorted by username.
 */
function buildIdentityBrowserUsers(
  users: StudioIdentityBrowserUser[],
): AdminIdentityBrowserUserModel[] {
  return users
    .slice()
    .sort((left, right) => left.username.localeCompare(right.username))
    .map((user) => ({
      active: user.active,
      activeLabel: user.active ? "active" : "disabled",
      expiresAt: user.expires_at_utc ?? "never",
      principalId: user.principal_id,
      rolesText: user.roles.join(", "),
      username: user.username,
    }));
}

/**
 * List service accounts by principal, without disturbing the caller's array.
 *
 * @param accounts - The accounts, in whatever order they arrived.
 * @returns The rows, sorted by principal.
 */
function buildIdentityAccounts(
  accounts: StudioIdentityServiceAccount[],
): AdminIdentityAccountModel[] {
  return accounts
    .slice()
    .sort((left, right) => left.principal_id.localeCompare(right.principal_id))
    .map((account) => ({
      active: account.active,
      activeLabel: account.active ? "active" : "disabled",
      expiresAt: account.expires_at_utc ?? "never",
      principalId: account.principal_id,
      rolesText: account.roles.join(", "),
    }));
}

/**
 * Build the job-queue row.
 *
 * A queue that has not answered is `unconfigured` rather than empty: zero
 * jobs and no queue are different states and the label says which.
 *
 * @param jobStatus - The queue's status, or `null`.
 * @returns The row's state.
 */
function buildJobModel(jobStatus: StudioJobStatus | null): AdminJobModel {
  if (jobStatus === null) {
    return {
      active: 0,
      allowedKinds: "unavailable",
      completed: 0,
      configured: false,
      failed: 0,
      healthLabel: "unconfigured",
      processCount: 0,
      resourceProfiles: [],
      threadCount: 0,
      timedOut: 0,
    };
  }
  const needsAttention = jobStatus.failed_count > 0 || jobStatus.timed_out_count > 0;
  return {
    active: jobStatus.active_count,
    allowedKinds: jobStatus.allowed_kinds.join(", "),
    completed: jobStatus.completed_count,
    configured: jobStatus.configured,
    failed: jobStatus.failed_count,
    healthLabel: !jobStatus.configured ? "unconfigured" : needsAttention ? "attention" : "ready",
    processCount: jobStatus.process_count,
    resourceProfiles: jobStatus.resource_profiles.map((profile) =>
      `${profile.kind}: ${profile.default_timeout_seconds}s, ${profile.max_artifact_bytes} bytes, ${profile.execution_models.join("+")}`,
    ),
    threadCount: jobStatus.thread_count,
    timedOut: jobStatus.timed_out_count,
  };
}

/**
 * Take the eight most recent jobs, newest first.
 *
 * @param records - The jobs, oldest first as the server lists them.
 * @returns The rows, newest first.
 */
function buildJobRecords(records: StudioJobRecord[]): AdminJobRecordModel[] {
  return records
    .slice(-8)
    .reverse()
    .map((record) => {
      const artifactPaths = record.artifacts.map((artifact) => artifact.relative_path);
      return {
        artifactCount: record.artifacts.length,
        artifactPaths: artifactPaths.length > 0 ? artifactPaths.join(", ") : "none",
        createdAt: record.created_at_utc,
        evidenceArtifactCount: artifactPaths.filter(isEvidenceArtifactPath).length,
        error: record.error,
        executionModel: record.execution_model,
        finishedAt: record.finished_at_utc ?? "running",
        jobId: record.job_id,
        kind: record.kind,
        owner: record.owner,
        status: record.status,
      };
    });
}

/**
 * Whether an artefact path names an evidence document.
 *
 * @param path - The artefact's relative path.
 * @returns Whether its filename is `evidence.json` or ends in
 *   `-evidence.json`.
 */
function isEvidenceArtifactPath(path: string): boolean {
  const parts = path.split("/");
  const filename = parts[parts.length - 1] ?? path;
  return filename === "evidence.json" || filename.endsWith("-evidence.json");
}

/**
 * Build the deployment-settings row.
 *
 * Every field is `unknown` when the operator status has not arrived, which
 * is why the whole row is strings.
 *
 * @param operatorStatus - The status, or `null`.
 * @returns The row's state.
 */
function buildOperatorModel(operatorStatus: StudioOperatorStatus | null): AdminOperatorModel {
  if (operatorStatus === null) {
    return {
      browserLoginCooldown: "unknown",
      browserLoginActiveBuckets: "unknown",
      browserLoginLockedBuckets: "unknown",
      browserLoginLimit: "unknown",
      browserLoginMaxRetryAfter: "unknown",
      browserLoginWindow: "unknown",
      deploymentProfile: "unknown",
      edaCpuLimit: "unknown",
      edaMemoryLimit: "unknown",
      edaLimitSupport: "unknown",
      identityMode: "unknown",
      jobArtifactLimit: "unknown",
      jobTimeout: "unknown",
      routePolicyAuditLabel: "unknown",
      routePolicyInventory: "unknown",
      routePolicyLabel: "unknown",
      schemaVersion: "unavailable",
    };
  }
  // The three blocks below are required fields of the operator status, and the
  // browser-login one used to be read through six `=== undefined` guards while
  // the other two were dereferenced directly one line later. Those guards
  // bought nothing: a response missing one block would throw on the next, so
  // they defended one field of an answer nothing validates. Response
  // validation at the API boundary is a real gap and is recorded as its own
  // unit; a guard on one block of one route is not a substitute for it.
  const browserLogin = operatorStatus.browser_login;
  const limits = operatorStatus.resource_limits;
  const routePolicies = operatorStatus.route_policies;
  return {
    browserLoginActiveBuckets: `${browserLogin.active_bucket_count}`,
    browserLoginCooldown: formatSeconds(browserLogin.cooldown_seconds),
    browserLoginLockedBuckets: `${browserLogin.locked_bucket_count}`,
    browserLoginLimit: `${browserLogin.max_failures}`,
    browserLoginMaxRetryAfter: formatSeconds(browserLogin.max_retry_after_seconds),
    browserLoginWindow: formatSeconds(browserLogin.failure_window_seconds),
    deploymentProfile: operatorStatus.deployment_profile,
    edaCpuLimit: formatSeconds(limits.eda_process_cpu_seconds),
    edaMemoryLimit: formatBytes(limits.eda_process_memory_bytes),
    edaLimitSupport: limits.eda_process_limits_supported ? "supported" : "unsupported",
    identityMode: operatorStatus.identity.mode,
    jobArtifactLimit: formatBytes(limits.job_max_artifact_bytes),
    jobTimeout: formatSeconds(limits.job_default_timeout_seconds),
    routePolicyAuditLabel: routePolicies.protected_routes_audited ? "audited" : "incomplete",
    routePolicyInventory:
      `${routePolicies.total_count} total / ${routePolicies.protected_count} protected`,
    routePolicyLabel: routePolicies.enforced ? "enforced" : "disabled",
    schemaVersion: operatorStatus.schema_version,
  };
}

/**
 * Render a duration in seconds, or say it is unlimited.
 *
 * @param value - The seconds, or `null` for no limit.
 * @returns The rendered duration.
 */
function formatSeconds(value: number | null): string {
  if (value === null) {
    return "unbounded";
  }
  return Number.isInteger(value) ? `${value}s` : `${value.toFixed(1)}s`;
}

/**
 * Render a size in bytes, or say it is unlimited.
 *
 * @param value - The bytes, or `null` for no limit.
 * @returns The rendered size.
 */
function formatBytes(value: number | null): string {
  if (value === null) {
    return "unbounded";
  }
  const gib = 1024 * 1024 * 1024;
  const mib = 1024 * 1024;
  if (value >= gib && value % gib === 0) {
    return `${value / gib} GiB`;
  }
  if (value >= mib && value % mib === 0) {
    return `${value / mib} MiB`;
  }
  return `${value} B`;
}
