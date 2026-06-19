import { describe, expect, it } from "vitest";

import type {
  StudioAuditExport,
  StudioAuditStatus,
  StudioCapability,
  StudioJobRecord,
  StudioJobStatus,
  StudioOperatorStatus,
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

const jobRecord: StudioJobRecord = {
  artifacts: [
    {
      relative_path: "reports/result.txt",
      sha256: "a".repeat(64),
      size_bytes: 12,
    },
  ],
  created_at_utc: "2026-06-19T20:02:00Z",
  error: null,
  finished_at_utc: "2026-06-19T20:03:00Z",
  job_id: "sj_1234",
  kind: "synthesis",
  owner: "operator-1",
  request_id: "req-3",
  result: { ok: true },
  started_at_utc: "2026-06-19T20:02:01Z",
  status: "completed",
};

const operatorStatus: StudioOperatorStatus = {
  audit: auditStatus,
  capabilities: {
    degraded_count: 0,
    experimental_count: 2,
    healthy_count: 1,
    stable_count: 1,
    total_count: 2,
    unavailable_count: 1,
  },
  deployment_profile: "production",
  identity: {
    configured: true,
    header_principal_allowed: false,
    mode: "service_account",
  },
  jobs: jobStatus,
  route_policies: { enforced: true },
  schema_version: "studio.operator.status.v1",
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
      jobRecords: [jobRecord],
      jobStatus,
      operatorStatus,
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
    expect(model.jobRecords).toEqual([
      {
        artifactCount: 1,
        createdAt: "2026-06-19T20:02:00Z",
        error: null,
        finishedAt: "2026-06-19T20:03:00Z",
        jobId: "sj_1234",
        kind: "synthesis",
        owner: "operator-1",
        status: "completed",
      },
    ]);
    expect(model.operator).toEqual({
      deploymentProfile: "production",
      identityMode: "service_account",
      routePolicyLabel: "enforced",
      schemaVersion: "studio.operator.status.v1",
    });
    expect(model.unhealthyCapabilities).toHaveLength(1);
    expect(model.recentAuditEvents.map((event) => event.action)).toEqual([
      "studio.audit.export",
      "studio.simulation.run",
    ]);
  });
});
