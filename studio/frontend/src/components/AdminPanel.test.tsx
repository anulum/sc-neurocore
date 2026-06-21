import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type {
  StudioAuditExport,
  StudioAuditQuarantineArchivePurgeResult,
  StudioAuditQuarantineArchiveResult,
  StudioAuditQuarantineArchiveRetentionPlan,
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
  event_count: 3,
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
    {
      action: "studio.auth.login",
      decision: "deny",
      event_hash: "hash-auth",
      previous_event_hash: "hash-1",
      principal_id: null,
      reason: "invalid_browser_login",
      request_id: "req-auth",
      route: "/api/studio/auth/login",
      schema_version: "studio.audit.v1",
      timestamp_utc: "2026-06-19T20:01:30Z",
    },
    {
      action: "studio.identity.browser_user.password.rotate",
      decision: "allow",
      event_hash: "hash-2",
      previous_event_hash: "hash-auth",
      principal_id: "operator-1",
      reason: "authorized",
      request_id: "req-2",
      route: "/api/studio/identity/browser-users/operator/password",
      schema_version: "studio.audit.v1",
      timestamp_utc: "2026-06-19T20:02:00Z",
    },
  ],
  schema_version: "studio.audit.export.v1",
  sink_type: "jsonl",
  truncated: false,
};

const auditArchive: StudioAuditQuarantineArchiveResult = {
  archive_id: "saqa_sj_archive",
  artifact_paths: [
    "evidence/audit-quarantine/archive.json",
    "evidence/audit-quarantine/manifest.json",
  ],
  artifacts: [
    {
      relative_path: "evidence/audit-quarantine/archive.json",
      sha256: "a".repeat(64),
      size_bytes: 1024,
    },
  ],
  job_id: "sj_archive",
  manifest: { schema_version: "studio.audit-quarantine-archive.v1" },
  schema_version: "studio.audit-quarantine-archive.v1",
  summary: {
    archive_artifact_count: 2,
    event_count: 3,
    quarantine_reason: "legacy_or_corrupt_retained_rows",
    reason_counts: { chain_broken: 1, legacy_row: 2 },
    retained_event_count: 8,
    source_schema_version: "studio.audit.quarantine.export.v1",
    truncated: false,
  },
};

const auditArchiveRetention: StudioAuditQuarantineArchiveRetentionPlan = {
  archive_count: 2,
  entries: [
    {
      archive_id: "saqa_sj_new",
      artifact_paths: ["evidence/audit-quarantine/archive.json"],
      created_at_utc: "2026-06-21T10:00:00Z",
      disposition: "retain",
      event_count: 3,
      finished_at_utc: "2026-06-21T10:00:01Z",
      job_id: "sj_new",
      retained_event_count: 8,
      summary: auditArchive.summary,
    },
    {
      archive_id: "saqa_sj_old",
      artifact_paths: ["evidence/audit-quarantine/archive.json"],
      created_at_utc: "2026-06-21T09:00:00Z",
      disposition: "prune_candidate",
      event_count: 2,
      finished_at_utc: "2026-06-21T09:00:01Z",
      job_id: "sj_old",
      retained_event_count: 7,
      summary: auditArchive.summary,
    },
  ],
  prune_candidate_count: 1,
  retain_count: 1,
  retain_latest: 1,
  schema_version: "studio.audit-quarantine-archive.retention.v1",
  skipped_record_count: 0,
};

const auditArchivePurge: StudioAuditQuarantineArchivePurgeResult = {
  purged_archive_count: 1,
  purged_entries: [auditArchiveRetention.entries[1]],
  retained_archive_count: 1,
  retained_entries: [auditArchiveRetention.entries[0]],
  retain_latest: 1,
  schema_version: "studio.audit-quarantine-archive.purge.v1",
  skipped_record_count: 0,
};

const jobStatus: StudioJobStatus = {
  active_count: 1,
  allowed_kinds: ["compiler", "evidence", "synthesis", "training"],
  completed_count: 4,
  configured: true,
  failed_count: 0,
  process_count: 3,
  resource_profiles: [
    {
      default_timeout_seconds: 3,
      execution_models: ["thread", "process"],
      kind: "compiler",
      max_artifact_bytes: 16777216,
    },
  ],
  schema_version: "studio.jobs.status.v1",
  thread_count: 2,
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
    entries: [
      {
        bundle_path: "evidence/audit-export.json",
        evidence_classification: "audit",
        type: "audit_export",
      },
      {
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
  execution_model: "process",
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
  browser_login: {
    active_bucket_count: 2,
    cooldown_seconds: 900,
    failure_window_seconds: 300,
    locked_bucket_count: 1,
    max_retry_after_seconds: 120,
    max_failures: 5,
  },
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
    authenticated_count: 56,
    enforced: true,
    protected_audit_action_count: 73,
    protected_count: 73,
    protected_routes_audited: true,
    public_count: 22,
    total_count: 95,
  },
  schema_version: "studio.operator.status.v1",
};

describe("AdminPanel", () => {
  it("renders audit health, denied events, and degraded capabilities", () => {
    const model = buildAdminShellModel({
      auditArchive,
      auditArchivePurge,
      auditArchiveRetention,
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
        onCreateAuditArchive={async () => undefined}
        onCreateEvidenceBundle={async () => undefined}
        onCreateIdentityBrowserUser={async () => undefined}
        onDownloadEvidenceArtifact={async () => undefined}
        onLoadAuditExport={async () => undefined}
        onLoadAuditArchiveRetention={async () => undefined}
        onLoadAuditStatus={async () => undefined}
        onLoadIdentityServiceAccounts={async () => undefined}
        onLoadJobStatus={async () => undefined}
        onLoadOperatorStatus={async () => undefined}
        onPurgeAuditArchiveRetention={async () => undefined}
        onRotateIdentityBrowserUserPassword={async () => undefined}
        onUpdateIdentityBrowserUser={async () => undefined}
        onUpdateIdentityServiceAccount={async () => undefined}
      />,
    );

    expect(html).toContain("Operator");
    expect(html).toContain("production");
    expect(html).toContain("service_account");
    expect(html).toContain("enforced");
    expect(html).toContain("95 total / 73 protected");
    expect(html).toContain("audited");
    expect(html).toContain("120s");
    expect(html).toContain("2 GiB");
    expect(html).toContain("16 MiB");
    expect(html).toContain("Login limit");
    expect(html).toContain("Login window");
    expect(html).toContain("Login cooldown");
    expect(html).toContain("Login buckets");
    expect(html).toContain("Login locked");
    expect(html).toContain("Max retry");
    expect(html).toContain("900s");
    expect(html).toContain("Audit");
    expect(html).toContain("Browser auth");
    expect(html).toContain("Auth allowed");
    expect(html).toContain("Auth denied");
    expect(html).toContain("Latest auth");
    expect(html).toContain("studio.auth.login");
    expect(html).toContain("Identity lifecycle");
    expect(html).toContain("Identity allowed");
    expect(html).toContain("Identity denied");
    expect(html).toContain("studio.identity.browser_user.password.rotate");
    expect(html).toContain("jsonl");
    expect(html).toContain("unhealthy");
    expect(html).toContain("1");
    expect(html).toContain("missing_admin_role");
    expect(html).toContain("Audit archive");
    expect(html).toContain("saqa_sj_archive");
    expect(html).toContain("chain_broken:1, legacy_row:2");
    expect(html).toContain("Prune candidates");
    expect(html).toContain("1 purged / 1 retained");
    expect(html).toContain("saqa_sj_new");
    expect(html).toContain("saqa_sj_old");
    expect(html).toContain("Review audit archive retention");
    expect(html).toContain("Purge audit archive prune candidates");
    expect(html).toContain("Create audit quarantine archive");
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
    expect(html).not.toContain("/tmp/");
    expect(html).toContain("attention");
    expect(html).toContain("Process");
    expect(html).toContain("Thread");
    expect(html).toContain("compiler, evidence, synthesis, training");
    expect(html).toContain("compiler: 3s, 16777216 bytes, thread+process");
    expect(html).toContain("synthesis - sj_1234");
    expect(html).toContain("operator-1");
    expect(html).toContain("process");
    expect(html).toContain("2 artifacts - 1 evidence");
    expect(html).toContain("reports/result.txt, compiler/compile-evidence.json");
    expect(html).toContain("Add sj_1234 to evidence bundle");
    expect(html).toContain("Evidence");
    expect(html).toContain("seb_sj_evidence");
    expect(html).toContain("Evidence job IDs");
    expect(html).toContain("Evidence simulation JSON");
    expect(html).toContain("Evidence analysis JSON");
    expect(html).toContain("Evidence default-flow run JSON");
    expect(html).toContain("Evidence default-flow attestation JSON");
    expect(html).toContain("action_evidence:1, audit_export:1, manifest:1");
    expect(html).toContain("compile:1");
    expect(html).toContain("1 - compiler:1");
    expect(html).toContain("audit - evidence/audit-export.json");
    expect(html).toContain("job sj_compile");
    expect(html).toContain("compile - compiler/compile-evidence.json");
    expect(html).toContain("unclassified - sha dddddddddddd");
    expect(html).toContain("evidence/manifest.json");
    expect(html).toContain("256 B - sha bbbbbbbbbbbb");
    expect(html).toContain("Download evidence artifact evidence/manifest.json");
    expect(html).toContain("Create evidence bundle");
  });
});
