import type {
  StudioAuditEvent,
  StudioAuditExport,
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

export interface AdminShellInput {
  auditError: string | null;
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

export interface AdminAuditModel {
  denied: number;
  error: string | null;
  healthLabel: "ready" | "unhealthy";
  identityLifecycle: number;
  lastError: string | null;
  latestAction: string | null;
  latestIdentityLifecycleAction: string | null;
  sinkType: string;
  total: number;
  truncated: boolean;
}

export interface AdminCapabilityModel {
  registered: number;
  unhealthy: number;
  healthLabel: "ready" | "degraded";
}

export interface AdminJobModel {
  active: number;
  allowedKinds: string;
  completed: number;
  configured: boolean;
  failed: number;
  healthLabel: "ready" | "attention" | "unconfigured";
  resourceProfiles: string[];
  timedOut: number;
}

export interface AdminJobRecordModel {
  artifactCount: number;
  artifactPaths: string;
  createdAt: string;
  evidenceArtifactCount: number;
  error: string | null;
  finishedAt: string;
  jobId: string;
  kind: string;
  owner: string;
  status: StudioJobRecord["status"];
}

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

export interface AdminEvidenceBundleArtifactModel {
  relativePath: string;
  sha256: string;
  sha256Label: string;
  sizeBytes: number | null;
  sizeLabel: string;
}

export interface AdminEvidenceBundleEntryModel {
  classification: string;
  detail: string;
  index: number;
  source: string;
  type: string;
}

export interface AdminIdentityAccountModel {
  active: boolean;
  activeLabel: "active" | "disabled";
  expiresAt: string;
  principalId: string;
  rolesText: string;
}

export interface AdminIdentityBrowserUserModel {
  active: boolean;
  activeLabel: "active" | "disabled";
  expiresAt: string;
  principalId: string;
  rolesText: string;
  username: string;
}

export interface AdminOperatorModel {
  browserLoginCooldown: string;
  browserLoginLimit: string;
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

export interface AdminShellModel {
  audit: AdminAuditModel;
  capabilities: AdminCapabilityModel;
  evidenceBundle: AdminEvidenceBundleModel;
  jobs: AdminJobModel;
  jobRecords: AdminJobRecordModel[];
  identityBrowserUsers: AdminIdentityBrowserUserModel[];
  identityAccounts: AdminIdentityAccountModel[];
  operator: AdminOperatorModel;
  recentAuditEvents: StudioAuditEvent[];
  unhealthyCapabilities: StudioCapability[];
}

/** Build the operator-facing Admin panel state from backend contract payloads. */
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
      healthLabel: auditStatus?.healthy === false ? "unhealthy" : "ready",
      identityLifecycle: auditSummary.identityLifecycle,
      lastError: auditStatus?.last_error ?? null,
      latestAction: auditSummary.latestAction,
      latestIdentityLifecycleAction: auditSummary.latestIdentityLifecycleAction,
      sinkType: auditStatus?.sink_type ?? auditSummary.sinkType,
      total: auditSummary.total,
      truncated: auditSummary.truncated,
    },
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
    recentAuditEvents,
    unhealthyCapabilities,
  };
}

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

function formatSourceJobs(
  sourceJobCount: number | undefined,
  sourceJobKindCounts: Record<string, number> | undefined,
): string {
  const count = sourceJobCount ?? 0;
  const kinds = formatCounts(sourceJobKindCounts);
  return kinds === "none" ? `${count}` : `${count} - ${kinds}`;
}

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

function textField(entry: Record<string, unknown>, key: string): string | null {
  const value = entry[key];
  return typeof value === "string" && value.length > 0 ? value : null;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

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

function buildJobModel(jobStatus: StudioJobStatus | null): AdminJobModel {
  if (jobStatus === null) {
    return {
      active: 0,
      allowedKinds: "unavailable",
      completed: 0,
      configured: false,
      failed: 0,
      healthLabel: "unconfigured",
      resourceProfiles: [],
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
    resourceProfiles: jobStatus.resource_profiles.map((profile) =>
      `${profile.kind}: ${profile.default_timeout_seconds}s, ${profile.max_artifact_bytes} bytes, ${profile.execution_models.join("+")}`,
    ),
    timedOut: jobStatus.timed_out_count,
  };
}

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
        finishedAt: record.finished_at_utc ?? "running",
        jobId: record.job_id,
        kind: record.kind,
        owner: record.owner,
        status: record.status,
      };
    });
}

function isEvidenceArtifactPath(path: string): boolean {
  const parts = path.split("/");
  const filename = parts.length > 0 ? parts[parts.length - 1] : path;
  return filename === "evidence.json" || filename.endsWith("-evidence.json");
}

function buildOperatorModel(operatorStatus: StudioOperatorStatus | null): AdminOperatorModel {
  if (operatorStatus === null) {
    return {
      browserLoginCooldown: "unknown",
      browserLoginLimit: "unknown",
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
  const browserLogin = operatorStatus.browser_login;
  const limits = operatorStatus.resource_limits;
  const routePolicies = operatorStatus.route_policies;
  return {
    browserLoginCooldown: formatSeconds(browserLogin.cooldown_seconds),
    browserLoginLimit: `${browserLogin.max_failures}`,
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

function formatSeconds(value: number | null): string {
  if (value === null) {
    return "unbounded";
  }
  return Number.isInteger(value) ? `${value}s` : `${value.toFixed(1)}s`;
}

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
