import type {
  StudioAuditEvent,
  StudioAuditExport,
  StudioAuditStatus,
  StudioCapability,
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
  lastError: string | null;
  latestAction: string | null;
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
  timedOut: number;
}

export interface AdminJobRecordModel {
  artifactCount: number;
  createdAt: string;
  error: string | null;
  finishedAt: string;
  jobId: string;
  kind: string;
  owner: string;
  status: StudioJobRecord["status"];
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
  deploymentProfile: "development" | "production" | "unknown";
  edaCpuLimit: string;
  edaMemoryLimit: string;
  edaLimitSupport: "supported" | "unsupported" | "unknown";
  identityMode: string;
  jobArtifactLimit: string;
  jobTimeout: string;
  routePolicyLabel: "enforced" | "disabled" | "unknown";
  schemaVersion: string;
}

export interface AdminShellModel {
  audit: AdminAuditModel;
  capabilities: AdminCapabilityModel;
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
      lastError: auditStatus?.last_error ?? null,
      latestAction: auditSummary.latestAction,
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
    jobRecords: buildJobRecords(input.jobRecords),
    identityBrowserUsers: buildIdentityBrowserUsers(input.identityBrowserUsers),
    identityAccounts: buildIdentityAccounts(input.identityServiceAccounts),
    operator: buildOperatorModel(input.operatorStatus),
    recentAuditEvents,
    unhealthyCapabilities,
  };
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
    timedOut: jobStatus.timed_out_count,
  };
}

function buildJobRecords(records: StudioJobRecord[]): AdminJobRecordModel[] {
  return records
    .slice(-8)
    .reverse()
    .map((record) => ({
      artifactCount: record.artifacts.length,
      createdAt: record.created_at_utc,
      error: record.error,
      finishedAt: record.finished_at_utc ?? "running",
      jobId: record.job_id,
      kind: record.kind,
      owner: record.owner,
      status: record.status,
    }));
}

function buildOperatorModel(operatorStatus: StudioOperatorStatus | null): AdminOperatorModel {
  if (operatorStatus === null) {
    return {
      deploymentProfile: "unknown",
      edaCpuLimit: "unknown",
      edaMemoryLimit: "unknown",
      edaLimitSupport: "unknown",
      identityMode: "unknown",
      jobArtifactLimit: "unknown",
      jobTimeout: "unknown",
      routePolicyLabel: "unknown",
      schemaVersion: "unavailable",
    };
  }
  const limits = operatorStatus.resource_limits;
  return {
    deploymentProfile: operatorStatus.deployment_profile,
    edaCpuLimit: formatSeconds(limits.eda_process_cpu_seconds),
    edaMemoryLimit: formatBytes(limits.eda_process_memory_bytes),
    edaLimitSupport: limits.eda_process_limits_supported ? "supported" : "unsupported",
    identityMode: operatorStatus.identity.mode,
    jobArtifactLimit: formatBytes(limits.job_max_artifact_bytes),
    jobTimeout: formatSeconds(limits.job_default_timeout_seconds),
    routePolicyLabel: operatorStatus.route_policies.enforced ? "enforced" : "disabled",
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
