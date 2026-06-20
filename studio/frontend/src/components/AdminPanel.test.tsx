import { renderToStaticMarkup } from "react-dom/server";
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
} from "../api/client";
import { buildAdminShellModel } from "../adminShell";
import AdminPanelView from "./AdminPanelView";

function capability(overrides: Partial<StudioCapability> = {}): StudioCapability {
  return {
    capability_id: overrides.capability_id ?? "studio.capability_registry",
    title: overrides.title ?? "Capability Registry",
    summary: overrides.summary ?? "Registry.",
    status: overrides.status ?? "stable",
    healthy: overrides.healthy ?? true,
    message: overrides.message ?? "Ready.",
    requirements: overrides.requirements ?? [],
    evidence: overrides.evidence ?? ["contract_test"],
    ui_placement: overrides.ui_placement ?? "Admin",
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
  event_count: 1,
  events: [
    {
      action: "studio.audit.export",
      decision: "deny",
      event_hash: "hash-1",
      previous_event_hash: null,
      principal_id: "operator-1",
      reason: "missing_admin_role",
      request_id: "req-1",
      route: "/api/studio/audit/export",
      schema_version: "studio.audit.v1",
      timestamp_utc: "2026-06-19T20:01:00Z",
    },
  ],
  schema_version: "studio.audit.export.v1",
  sink_type: "jsonl",
  truncated: false,
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
      relative_path: "evidence/manifest.json",
      sha256: "b".repeat(64),
      size_bytes: 256,
    },
  ],
  bundle_id: "seb_sj_evidence",
  job_id: "sj_evidence",
  manifest: {
    entries: [{ type: "audit_export" }, { type: "manifest" }],
  },
  schema_version: "studio.evidence-bundle.v1",
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

describe("AdminPanel", () => {
  it("renders audit health, denied events, and degraded capabilities", () => {
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

    const html = renderToStaticMarkup(
      <AdminPanelView
        auditLoading={false}
        model={model}
        onCreateEvidenceBundle={async () => undefined}
        onCreateIdentityBrowserUser={async () => undefined}
        onLoadAuditExport={async () => undefined}
        onLoadAuditStatus={async () => undefined}
        onLoadIdentityServiceAccounts={async () => undefined}
        onLoadJobStatus={async () => undefined}
        onLoadOperatorStatus={async () => undefined}
        onRotateIdentityBrowserUserPassword={async () => undefined}
        onUpdateIdentityBrowserUser={async () => undefined}
        onUpdateIdentityServiceAccount={async () => undefined}
      />,
    );

    expect(html).toContain("Operator");
    expect(html).toContain("production");
    expect(html).toContain("service_account");
    expect(html).toContain("enforced");
    expect(html).toContain("93 total / 71 protected");
    expect(html).toContain("audited");
    expect(html).toContain("120s");
    expect(html).toContain("2 GiB");
    expect(html).toContain("16 MiB");
    expect(html).toContain("Audit");
    expect(html).toContain("jsonl");
    expect(html).toContain("unhealthy");
    expect(html).toContain("1");
    expect(html).toContain("missing_admin_role");
    expect(html).toContain("Synthesis Dashboard");
    expect(html).toContain("Yosys unavailable.");
    expect(html).toContain("Jobs");
    expect(html).toContain("Identity");
    expect(html).toContain("svc-admin");
    expect(html).toContain("studio.admin, studio.viewer");
    expect(html).toContain("New browser username");
    expect(html).toContain("New browser principal");
    expect(html).toContain("New browser secret");
    expect(html).toContain("Create browser user");
    expect(html).toContain("operator");
    expect(html).toContain("human-operator - 2030-01-01T00:00:00Z");
    expect(html).toContain("operator new secret");
    expect(html).toContain("studio.viewer");
    expect(html).not.toContain("token_sha256");
    expect(html).not.toContain("password_pbkdf2_sha256");
    expect(html).toContain("attention");
    expect(html).toContain("compiler, evidence, synthesis, training");
    expect(html).toContain("compiler: 3s, 16777216 bytes, thread+process");
    expect(html).toContain("synthesis - sj_1234");
    expect(html).toContain("operator-1");
    expect(html).toContain("Evidence");
    expect(html).toContain("seb_sj_evidence");
    expect(html).toContain("Evidence job IDs");
    expect(html).toContain("Evidence simulation JSON");
    expect(html).toContain("Evidence analysis JSON");
    expect(html).toContain("Evidence default-flow run JSON");
    expect(html).toContain("Evidence default-flow attestation JSON");
    expect(html).toContain("Create evidence bundle");
  });
});
