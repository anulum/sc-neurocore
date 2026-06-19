import type { StudioAuditEvent, StudioAuditExport, StudioAuditStatus, StudioCapability } from "./api/client";
import { summarizeAuditExport } from "./auditShell";

export interface AdminShellInput {
  auditError: string | null;
  auditExport: StudioAuditExport | null;
  auditStatus: StudioAuditStatus | null;
  capabilities: StudioCapability[];
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

export interface AdminShellModel {
  audit: AdminAuditModel;
  capabilities: AdminCapabilityModel;
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
    recentAuditEvents,
    unhealthyCapabilities,
  };
}
