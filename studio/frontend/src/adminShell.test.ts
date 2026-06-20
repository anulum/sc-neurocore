import { describe, expect, it } from "vitest";

import type {
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
  allowed_kinds: ["compiler", "evidence", "synthesis", "training"],
  completed_count: 4,
  configured: true,
  failed_count: 0,
  resource_profiles: [
    {
      default_timeout_seconds: 3,
      execution_models: ["thread", "process"],
      kind: "compiler",
      max_artifact_bytes: 16777216,
    },
  ],
  schema_version: "studio.jobs.status.v1",
  timed_out_count: 1,
};

const evidenceBundle: StudioEvidenceBundleResponse = {
  artifact_paths: [
    "evidence/audit-export.json",
    "evidence/manifest.json",
  ],
  artifacts: [
    {
      relative_path: "evidence/audit-export.json",
      sha256: "b".repeat(64),
      size_bytes: 128,
    },
    {
      relative_path: "evidence/manifest.json",
      sha256: "c".repeat(64),
      size_bytes: 256,
    },
  ],
  bundle_id: "seb_sj_evidence",
  job_id: "sj_evidence",
  manifest: {
    entries: [
      {
        bundle_path: "evidence/audit-export.json",
        evidence_classification: "audit",
        type: "audit_export",
      },
      {
        bundle_path: "evidence/jobs/sj_compile/artifacts/compiler/compile-evidence.json",
        evidence_classification: "compile",
        source_job_artifact_path: "compiler/compile-evidence.json",
        source_job_id: "sj_compile",
        type: "action_evidence",
      },
      {
        sha256: "d".repeat(64),
        type: "manifest",
      },
    ],
  },
  schema_version: "studio.evidence-bundle.v1",
  summary: {
    artifact_path_count: 2,
    entry_count: 3,
    entry_type_counts: { action_evidence: 1, audit_export: 1, manifest: 1 },
    evidence_classification_counts: { compile: 1 },
    source_job_count: 1,
    source_job_kind_counts: { compiler: 1 },
    source_job_owner_counts: { "studio-compiler": 1 },
  },
};

const jobRecord: StudioJobRecord = {
  artifacts: [
    {
      relative_path: "reports/result.txt",
      sha256: "a".repeat(64),
      size_bytes: 12,
    },
    {
      relative_path: "compiler/compile-evidence.json",
      sha256: "b".repeat(64),
      size_bytes: 256,
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

const identityServiceAccount: StudioIdentityServiceAccount = {
  active: true,
  expires_at_utc: null,
  principal_id: "svc-admin",
  roles: ["studio.admin", "studio.viewer"],
};

const identityBrowserUser: StudioIdentityBrowserUser = {
  active: false,
  expires_at_utc: "2030-01-01T00:00:00Z",
  principal_id: "human-operator",
  roles: ["studio.viewer"],
  username: "operator",
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
  resource_limits: {
    eda_process_cpu_seconds: 120,
    eda_process_limits_supported: true,
    eda_process_memory_bytes: 2147483648,
    job_default_timeout_seconds: 300,
    job_max_artifact_bytes: 16777216,
  },
  route_policies: {
    admin_count: 17,
    authenticated_count: 54,
    enforced: true,
    protected_audit_action_count: 71,
    protected_count: 71,
    protected_routes_audited: true,
    public_count: 22,
    total_count: 93,
  },
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
      evidenceBundle,
      evidenceBundleError: null,
      evidenceBundleLoading: false,
      identityBrowserUsers: [identityBrowserUser],
      identityServiceAccounts: [identityServiceAccount],
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
      allowedKinds: "compiler, evidence, synthesis, training",
      completed: 4,
      configured: true,
      failed: 0,
      healthLabel: "attention",
      resourceProfiles: ["compiler: 3s, 16777216 bytes, thread+process"],
      timedOut: 1,
    });
    expect(model.evidenceBundle).toEqual({
      artifactCount: 2,
      artifacts: [
        {
          relativePath: "evidence/audit-export.json",
          sha256: "b".repeat(64),
          sha256Label: "bbbbbbbbbbbb",
          sizeBytes: 128,
          sizeLabel: "128 B",
        },
        {
          relativePath: "evidence/manifest.json",
          sha256: "c".repeat(64),
          sha256Label: "cccccccccccc",
          sizeBytes: 256,
          sizeLabel: "256 B",
        },
      ],
      bundleId: "seb_sj_evidence",
      entries: [
        {
          classification: "audit",
          detail: "evidence/audit-export.json",
          index: 0,
          source: "evidence/audit-export.json",
          type: "audit_export",
        },
        {
          classification: "compile",
          detail: "compiler/compile-evidence.json",
          index: 1,
          source: "job sj_compile",
          type: "action_evidence",
        },
        {
          classification: "unclassified",
          detail: "sha dddddddddddd",
          index: 2,
          source: "bundle",
          type: "manifest",
        },
      ],
      entryTypes: "action_evidence:1, audit_export:1, manifest:1",
      error: null,
      evidenceClasses: "compile:1",
      jobId: "sj_evidence",
      loading: false,
      manifestEntryCount: 3,
      sourceJobs: "1 - compiler:1",
    });
    expect(model.jobRecords).toEqual([
      {
        artifactCount: 2,
        artifactPaths: "reports/result.txt, compiler/compile-evidence.json",
        createdAt: "2026-06-19T20:02:00Z",
        evidenceArtifactCount: 1,
        error: null,
        finishedAt: "2026-06-19T20:03:00Z",
        jobId: "sj_1234",
        kind: "synthesis",
        owner: "operator-1",
        status: "completed",
      },
    ]);
    expect(model.identityAccounts).toEqual([
      {
        active: true,
        activeLabel: "active",
        expiresAt: "never",
        principalId: "svc-admin",
        rolesText: "studio.admin, studio.viewer",
      },
    ]);
    expect(model.identityBrowserUsers).toEqual([
      {
        active: false,
        activeLabel: "disabled",
        expiresAt: "2030-01-01T00:00:00Z",
        principalId: "human-operator",
        rolesText: "studio.viewer",
        username: "operator",
      },
    ]);
    expect(model.operator).toEqual({
      deploymentProfile: "production",
      edaCpuLimit: "120s",
      edaLimitSupport: "supported",
      edaMemoryLimit: "2 GiB",
      identityMode: "service_account",
      jobArtifactLimit: "16 MiB",
      jobTimeout: "300s",
      routePolicyAuditLabel: "audited",
      routePolicyInventory: "93 total / 71 protected",
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
