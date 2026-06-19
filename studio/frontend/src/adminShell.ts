import type {
  StudioAuditEvent,
  StudioAuditExport,
  StudioAuditStatus,
  StudioCapability,
  StudioJobStatus,
} from "./api/client";
import { summarizeAuditExport } from "./auditShell";

export interface AdminShellInput {
  auditError: string | null;
  auditExport: StudioAuditExport | null;
  auditStatus: StudioAuditStatus | null;
  capabilities: StudioCapability[];
  jobStatus: StudioJobStatus | null;
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

export interface AdminShellModel {
  audit: AdminAuditModel;
  capabilities: AdminCapabilityModel;
  jobs: AdminJobModel;
  recentAuditEvents: StudioAuditEvent[];
  unhealthyCapabilities: StudioCapability[];
}

/** Build the operator-facing Admin panel state from backend contract payloads. */
export function buildAdminShellModel(input: AdminShellInput): AdminShellModel {
  const auditSummary = summarizeAuditExport(input.auditExport);
  const unhealthyCapabilities = input.capabilities.filter((capability) => !capability.healthy);
  const recentAuditEvents = input.auditExport?.events.slice(-8).reverse() ?? [];

  return {
    audit: {
      denied: auditSummary.denied,
      error: input.auditError,
      healthLabel: input.auditStatus?.healthy === false ? "unhealthy" : "ready",
      lastError: input.auditStatus?.last_error ?? null,
      latestAction: auditSummary.latestAction,
      sinkType: input.auditStatus?.sink_type ?? auditSummary.sinkType,
      total: auditSummary.total,
      truncated: auditSummary.truncated,
    },
    capabilities: {
      registered: input.capabilities.length,
      unhealthy: unhealthyCapabilities.length,
      healthLabel: unhealthyCapabilities.length === 0 ? "ready" : "degraded",
    },
    jobs: buildJobModel(input.jobStatus),
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
