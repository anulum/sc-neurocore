import { describe, expect, it } from "vitest";

import type {
  StudioAuditExport,
  StudioAuditStatus,
  StudioCapability,
  StudioJobStatus,
} from "./api/client";
import { buildAdminShellModel } from "./adminShell";

function capability(overrides: Partial<StudioCapability> = {}): StudioCapability {
  return {
    capability_id: overrides.capability_id ?? "studio.simulation_workbench",
    title: overrides.title ?? "Simulation Workbench",
    summary: overrides.summary ?? "Run simulations.",
    status: overrides.status ?? "stable",
    healthy: overrides.healthy ?? true,
    message: overrides.message ?? "Ready.",
    requirements: overrides.requirements ?? [],
    evidence: overrides.evidence ?? ["contract_test"],
    ui_placement: overrides.ui_placement ?? "Workbench",
    docs_path: overrides.docs_path ?? "docs/studio/index.md",
  };
}

const auditStatus: StudioAuditStatus = {
  configured: true,
  healthy: false,
  last_error: "AuditPathIsDirectory",
  path_configured: true,
  sink_type: "jsonl",
};

const auditExport: StudioAuditExport = {
  configured: true,
  event_count: 2,
  events: [
    {
      action: "studio.simulation.run",
      decision: "allow",
      event_hash: "hash-1",
      previous_event_hash: null,
      principal_id: "operator-1",
      reason: "authorized",
      request_id: "req-1",
      route: "/api/simulate",
      schema_version: "studio.audit.v1",
      timestamp_utc: "2026-06-19T20:00:00Z",
    },
    {
      action: "studio.audit.export",
      decision: "deny",
      event_hash: "hash-2",
      previous_event_hash: "hash-1",
      principal_id: "operator-2",
      reason: "missing_admin_role",
      request_id: "req-2",
      route: "/api/studio/audit/export",
      schema_version: "studio.audit.v1",
      timestamp_utc: "2026-06-19T20:01:00Z",
    },
  ],
  schema_version: "studio.audit.export.v1",
  sink_type: "jsonl",
  truncated: true,
};

const jobStatus: StudioJobStatus = {
  active_count: 1,
  allowed_kinds: ["compiler", "synthesis", "training"],
  completed_count: 4,
  configured: true,
  failed_count: 0,
  schema_version: "studio.jobs.status.v1",
  timed_out_count: 1,
};

describe("admin shell model", () => {
  it("aggregates audit and capability health for the operator view", () => {
    const model = buildAdminShellModel({
      auditError: "Audit export failed",
      auditExport,
      auditStatus,
      capabilities: [
        capability(),
        capability({
          capability_id: "studio.synthesis_dashboard",
          title: "Synthesis Dashboard",
          status: "unavailable",
          healthy: false,
          message: "Yosys unavailable.",
        }),
      ],
      jobStatus,
    });

    expect(model.audit).toEqual({
      denied: 1,
      error: "Audit export failed",
      healthLabel: "unhealthy",
      lastError: "AuditPathIsDirectory",
      latestAction: "studio.audit.export",
      sinkType: "jsonl",
      total: 2,
      truncated: true,
    });
    expect(model.capabilities).toEqual({
      registered: 2,
      unhealthy: 1,
      healthLabel: "degraded",
    });
    expect(model.jobs).toEqual({
      active: 1,
      allowedKinds: "compiler, synthesis, training",
      completed: 4,
      configured: true,
      failed: 0,
      healthLabel: "attention",
      timedOut: 1,
    });
    expect(model.unhealthyCapabilities).toHaveLength(1);
    expect(model.recentAuditEvents.map((event) => event.action)).toEqual([
      "studio.audit.export",
      "studio.simulation.run",
    ]);
  });
});
