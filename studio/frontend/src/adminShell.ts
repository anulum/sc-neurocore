import type {
  StudioAuditEvent,
  StudioAuditExport,
  StudioAuditStatus,
  StudioCapability,
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

export interface AdminOperatorModel {
  deploymentProfile: "development" | "production" | "unknown";
  identityMode: string;
  routePolicyLabel: "enforced" | "disabled" | "unknown";
  schemaVersion: string;
}

export interface AdminShellModel {
  audit: AdminAuditModel;
  capabilities: AdminCapabilityModel;
  jobs: AdminJobModel;
  jobRecords: AdminJobRecordModel[];
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
    operator: buildOperatorModel(input.operatorStatus),
    recentAuditEvents,
    unhealthyCapabilities,
  };
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
      identityMode: "unknown",
      routePolicyLabel: "unknown",
      schemaVersion: "unavailable",
    };
  }
  return {
    deploymentProfile: operatorStatus.deployment_profile,
    identityMode: operatorStatus.identity.mode,
    routePolicyLabel: operatorStatus.route_policies.enforced ? "enforced" : "disabled",
    schemaVersion: operatorStatus.schema_version,
  };
}
