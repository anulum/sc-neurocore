import { expect, test, type Page } from "@playwright/test";

test.setTimeout(60_000);

function capability(overrides: Record<string, unknown>): Record<string, unknown> {
  return {
    capability_id: "studio.capability_registry",
    docs_path: "docs/studio/index.md",
    evidence: ["contract_test"],
    healthy: true,
    message: "Capability is available.",
    requirements: [{ available: true, detail: "registry active", name: "studio.platform" }],
    status: "stable",
    summary: "Typed inventory for Studio capabilities, requirements, and evidence.",
    title: "Capability Registry",
    ui_placement: "Admin",
    ...overrides,
  };
}

function registry(capabilities: Record<string, unknown>[]): Record<string, unknown> {
  return { capabilities };
}

const capabilityRegistryContract = capability({});

const simulationCapability = capability({
  capability_id: "studio.simulation_workbench",
  summary: "Simulation traces and spike statistics.",
  title: "Simulation Workbench",
  ui_placement: "Trace",
});

const analysisUnavailable = capability({
  capability_id: "studio.analysis_suite",
  evidence: ["static_inventory"],
  healthy: false,
  message: "Analysis endpoints are unavailable.",
  requirements: [{ available: false, detail: "analysis endpoint disabled", name: "analysis" }],
  status: "unavailable",
  summary: "Trace analysis and sweep tools.",
  title: "Analysis Suite",
  ui_placement: "Analysis",
});

const synthesisUnavailable = capability({
  capability_id: "studio.synthesis_dashboard",
  evidence: ["static_inventory"],
  healthy: false,
  message: "Synthesis tools are unavailable.",
  requirements: [{ available: false, detail: "yosys unavailable", name: "yosys" }],
  status: "unavailable",
  summary: "FPGA synthesis and place-and-route tools.",
  title: "Synthesis Dashboard",
  ui_placement: "FPGA",
});

const synthesisCapability = capability({
  capability_id: "studio.synthesis_dashboard",
  evidence: ["target_provenance_matrix"],
  summary: "FPGA synthesis and target provenance.",
  title: "Synthesis Dashboard",
  ui_placement: "FPGA",
});

const compilerCapability = capability({
  capability_id: "studio.compiler_inspector",
  evidence: ["compile_traceability"],
  summary: "IR and RTL inspection.",
  title: "Compiler Inspector",
  ui_placement: "IR",
});

const capabilityRegistry = {
  capabilities: [
    capabilityRegistryContract,
  ],
};

const auditStatus = {
  configured: true,
  healthy: true,
  last_error: null,
  path_configured: true,
  sink_type: "jsonl",
};

const auditExport = {
  configured: true,
  event_count: 1,
  events: [
    {
      action: "studio.operator.status.read",
      decision: "allow",
      event_hash: "event-hash-1",
      previous_event_hash: null,
      principal_id: "svc-admin",
      reason: "authorized",
      request_id: "req-browser-1",
      route: "/api/studio/operator/status",
      schema_version: "studio.audit.v1",
      timestamp_utc: "2026-06-20T00:00:00Z",
    },
  ],
  schema_version: "studio.audit.export.v1",
  sink_type: "jsonl",
  truncated: false,
};

const auditArchiveSummary = {
  archive_artifact_count: 2,
  event_count: 3,
  quarantine_reason: "legacy_or_corrupt_retained_rows",
  reason_counts: { chain_broken: 1, legacy_row: 2 },
  retained_event_count: 8,
  source_schema_version: "studio.audit.quarantine.export.v1",
  truncated: false,
};

const auditArchiveResult = {
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
  summary: auditArchiveSummary,
};

const auditArchiveValidation = {
  archive_id: "saqa_sj_archive",
  errors: [],
  schema_version: "studio.audit-quarantine-archive.validation.v1",
  summary: auditArchiveSummary,
  valid: true,
  warnings: ["manifest_recomputed"],
};

const auditArchiveRestore = {
  archive_id: "saqa_sj_archive",
  artifact_paths: [
    "evidence/audit-quarantine/restore.jsonl",
    "evidence/audit-quarantine/restore-manifest.json",
  ],
  artifacts: [
    {
      relative_path: "evidence/audit-quarantine/restore.jsonl",
      sha256: "e".repeat(64),
      size_bytes: 512,
    },
    {
      relative_path: "evidence/audit-quarantine/restore-manifest.json",
      sha256: "f".repeat(64),
      size_bytes: 384,
    },
  ],
  job_id: "sj_restore",
  manifest: { schema_version: "studio.audit-quarantine-archive.restore.v1" },
  schema_version: "studio.audit-quarantine-archive.restore.v1",
  summary: {
    ...auditArchiveSummary,
    restore_artifact_count: 2,
    restored_at_utc: "2026-06-21T11:00:00Z",
  },
};

const auditArchiveRetention = {
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
      summary: auditArchiveSummary,
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
      summary: auditArchiveSummary,
    },
  ],
  prune_candidate_count: 1,
  retain_count: 1,
  retain_latest: 1,
  schema_version: "studio.audit-quarantine-archive.retention.v1",
  skipped_record_count: 0,
};

const auditArchiveRetentionAfterPurge = {
  ...auditArchiveRetention,
  archive_count: 1,
  entries: [auditArchiveRetention.entries[0]],
  prune_candidate_count: 0,
  retain_count: 1,
};

const auditArchivePurge = {
  purged_archive_count: 1,
  purged_entries: [auditArchiveRetention.entries[1]],
  retained_archive_count: 1,
  retained_entries: [auditArchiveRetention.entries[0]],
  retain_latest: 1,
  schema_version: "studio.audit-quarantine-archive.purge.v1",
  skipped_record_count: 0,
};

const projectSaveResult = {
  evidence_classification: "project_workspace",
  name: "saved-network",
  project_sha256: "b".repeat(64),
  saved_at: 1782010000,
  schema_version: "studio.project-save.v1",
  state_sha256: "a".repeat(64),
  version: "studio.project.v1",
};

const jobStatus = {
  active_count: 1,
  allowed_kinds: ["compiler", "synthesis", "training"],
  completed_count: 7,
  configured: true,
  failed_count: 0,
  process_count: 1,
  resource_profiles: [
    {
      default_timeout_seconds: 120,
      execution_models: ["thread", "process"],
      kind: "compiler",
      max_artifact_bytes: 16777216,
    },
  ],
  schema_version: "studio.jobs.status.v1",
  thread_count: 0,
  timed_out_count: 0,
};

const jobList = {
  jobs: [],
  schema_version: "studio.jobs.list.v1",
};

const artifactJobList = {
  jobs: [
    {
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
      created_at_utc: "2026-06-20T01:00:00Z",
      error: null,
      execution_model: "process",
      finished_at_utc: "2026-06-20T01:01:00Z",
      job_id: "sj_artifact",
      kind: "compiler",
      owner: "svc-admin",
      request_id: "req-artifact",
      result: { ok: true },
      started_at_utc: "2026-06-20T01:00:01Z",
      status: "completed",
    },
  ],
  schema_version: "studio.jobs.list.v1",
};

const operatorStatus = {
  audit: auditStatus,
  browser_login: {
    active_bucket_count: 0,
    cooldown_seconds: 900,
    failure_window_seconds: 300,
    locked_bucket_count: 0,
    max_failures: 5,
    max_retry_after_seconds: 0,
  },
  capabilities: {
    degraded_count: 0,
    experimental_count: 0,
    healthy_count: 1,
    stable_count: 1,
    total_count: 1,
    unavailable_count: 0,
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

interface ApiMockSequence {
  sequence: (object | ApiMockBinary)[];
}

interface ApiMockBinary {
  binaryBody: string;
  contentType: string;
}

type ApiMockPayload = object | ApiMockSequence | ApiMockBinary;

interface ApiDispatcher {
  bodies: (path: string) => unknown[];
  headers: (path: string) => Record<string, string>[];
  requests: (path: string) => number;
}

async function installApiDispatcher(
  page: Page,
  mocks: Map<string, ApiMockPayload>,
): Promise<ApiDispatcher> {
  await page.unrouteAll({ behavior: "ignoreErrors" });
  const bodies = new Map<string, unknown[]>();
  const counts = new Map<string, number>();
  const headers = new Map<string, Record<string, string>[]>();
  await page.route((url) => url.pathname.startsWith("/api/"), async (route) => {
    const url = new URL(route.request().url());
    const path = `${url.pathname}${url.search}`;
    const count = counts.get(path) ?? 0;
    counts.set(path, count + 1);
    const recordedHeaders = headers.get(path) ?? [];
    recordedHeaders.push(route.request().headers());
    headers.set(path, recordedHeaders);
    const postData = route.request().postData();
    if (postData !== null) {
      const recorded = bodies.get(path) ?? [];
      try {
        recorded.push(JSON.parse(postData) as unknown);
      } catch {
        recorded.push(postData);
      }
      bodies.set(path, recorded);
    }
    const payload = mocks.get(path);
    if (payload === undefined) {
      await route.fulfill({
        contentType: "application/json",
        json: { detail: "unmocked Studio browser test route" },
        status: 404,
      });
      return;
    }
    const selectedPayload = "sequence" in payload
      ? payload.sequence[Math.min(count, payload.sequence.length - 1)]
      : payload;
    if ("binaryBody" in selectedPayload) {
      await route.fulfill({
        body: selectedPayload.binaryBody,
        contentType: selectedPayload.contentType,
        status: 200,
      });
      return;
    }
    await route.fulfill({
      contentType: "application/json",
      json: selectedPayload,
      status: 200,
    });
  });
  return {
    bodies: (path: string) => bodies.get(path) ?? [],
    headers: (path: string) => headers.get(path) ?? [],
    requests: (path: string) => counts.get(path) ?? 0,
  };
}

function defaultApiMocks(): Map<string, ApiMockPayload> {
  return new Map<string, ApiMockPayload>([
    ["/api/studio/capabilities", capabilityRegistry],
    ["/api/studio/audit/status", auditStatus],
    ["/api/studio/audit/export?limit=100", auditExport],
    ["/api/studio/jobs/status", jobStatus],
    ["/api/studio/jobs", jobList],
    ["/api/studio/auth/session", {
      authenticated: true,
      principal_id: "svc-admin",
      roles: ["studio.admin", "studio.viewer"],
    }],
    ["/api/studio/evidence/bundle", {
      artifact_paths: [
        "evidence/simulations/000.json",
        "evidence/analyses/000.json",
        "evidence/default-flows/runs/000.json",
        "evidence/default-flows/attestations/000.json",
        "evidence/manifest.json",
      ],
      artifacts: [
        {
          relative_path: "evidence/simulations/000.json",
          sha256: "c".repeat(64),
          size_bytes: 256,
        },
        {
          relative_path: "evidence/analyses/000.json",
          sha256: "d".repeat(64),
          size_bytes: 192,
        },
        {
          relative_path: "evidence/default-flows/runs/000.json",
          sha256: "e".repeat(64),
          size_bytes: 512,
        },
        {
          relative_path: "evidence/default-flows/attestations/000.json",
          sha256: "f".repeat(64),
          size_bytes: 256,
        },
      ],
      bundle_id: "seb_sj_browser",
      job_id: "sj_browser",
      manifest: {
        entries: [
          {
            bundle_path: "evidence/simulations/000.json",
            evidence_classification: "simulation",
            source: "ode",
            type: "simulation_result",
          },
          {
            bundle_path: "evidence/analyses/000.json",
            evidence_classification: "analysis",
            source: "ode",
            type: "analysis_result",
          },
          {
            bundle_path: "evidence/default-flows/runs/000.json",
            evidence_classification: "default_flow",
            source: "studio_default_adaptive_precision_v1",
            type: "default_flow_run",
          },
          {
            bundle_path: "evidence/default-flows/attestations/000.json",
            evidence_classification: "default_flow",
            source: "studio_default_adaptive_precision_v1",
            type: "default_flow_attestation",
          },
          { sha256: "f".repeat(64), type: "manifest" },
        ],
      },
      schema_version: "studio.evidence-bundle.v1",
      summary: {
        artifact_path_count: 5,
        entry_count: 5,
        entry_type_counts: {
          analysis_result: 1,
          default_flow_attestation: 1,
          default_flow_run: 1,
          manifest: 1,
          simulation_result: 1,
        },
        evidence_classification_counts: {
          analysis: 1,
          simulation: 1,
        },
        source_job_count: 0,
        source_job_kind_counts: {},
        source_job_owner_counts: {},
      },
    }],
    ["/api/studio/operator/status", operatorStatus],
    ["/api/models", []],
    ["/api/templates", []],
    ["/api/presets", []],
  ]);
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("sc-studio-onboarding-dismissed", "true");
    window.sessionStorage.setItem("sc-neurocore-studio-auth-token", "browser-token");
  });
  await installApiDispatcher(page, defaultApiMocks());
});

test("admin panel renders aggregate operator status", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByText("1/1 ready")).toBeVisible();
  await page.getByRole("button", { name: "Admin" }).first().click();

  await expect(page.getByRole("heading", { name: "Operator" })).toBeVisible();
  await expect(page.getByText("production")).toBeVisible();
  await expect(page.getByText("enforced")).toBeVisible();
  await expect(page.getByText("93 total / 71 protected")).toBeVisible();
  await expect(page.getByText("audited")).toBeVisible();
  await expect(page.getByText("service_account")).toBeVisible();
  await expect(page.getByText("studio.operator.status.v1")).toBeVisible();
  await expect(page.getByRole("heading", { exact: true, name: "Audit" })).toBeVisible();
  await expect(page.getByText("jsonl")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Jobs" })).toBeVisible();
  await expect(page.getByText("compiler, synthesis, training")).toBeVisible();
  await expect(page.getByRole("heading", { name: "Capabilities" })).toBeVisible();
  await expect(page.getByText("All registered capabilities healthy")).toBeVisible();
});

test("admin panel refreshes operator, audit, export, and job status", async ({ page }) => {
  const refreshedAuditStatus = {
    ...auditStatus,
    healthy: false,
    last_error: "AuditPathPermissionDenied",
  };
  const refreshedAuditExport = {
    ...auditExport,
    event_count: 2,
    events: [
      ...auditExport.events,
      {
        action: "studio.synth.run",
        decision: "deny",
        event_hash: "event-hash-2",
        previous_event_hash: "event-hash-1",
        principal_id: "operator-missing-role",
        reason: "missing_admin_role",
        request_id: "req-browser-2",
        route: "/api/synth/run",
        schema_version: "studio.audit.v1",
        timestamp_utc: "2026-06-20T00:01:00Z",
      },
    ],
  };
  const refreshedJobStatus = {
    ...jobStatus,
    active_count: 0,
    completed_count: 8,
    failed_count: 1,
    timed_out_count: 1,
  };
  const refreshedOperatorStatus = {
    ...operatorStatus,
    audit: refreshedAuditStatus,
    deployment_profile: "development",
    jobs: refreshedJobStatus,
    route_policies: {
      ...operatorStatus.route_policies,
      enforced: false,
      protected_audit_action_count: 39,
      protected_routes_audited: false,
    },
    schema_version: "studio.operator.status.v2",
  };
  const api = await installApiDispatcher(
    page,
    new Map<string, ApiMockPayload>([
      ["/api/studio/capabilities", capabilityRegistry],
      [
        "/api/studio/operator/status",
        { sequence: [operatorStatus, refreshedOperatorStatus] },
      ],
      ["/api/studio/audit/status", { sequence: [auditStatus, refreshedAuditStatus] }],
      ["/api/studio/audit/export?limit=100", refreshedAuditExport],
      ["/api/studio/jobs/status", refreshedJobStatus],
      ["/api/studio/jobs", jobList],
      ["/api/models", []],
      ["/api/templates", []],
      ["/api/presets", []],
    ]),
  );

  await page.goto("/");
  await page.getByRole("button", { name: "Admin" }).first().click();

  await page.getByRole("button", { name: "Refresh operator status" }).click();
  await expect(page.getByText("development")).toBeVisible();
  await expect(page.getByText("disabled")).toBeVisible();
  await expect(page.getByText("incomplete")).toBeVisible();
  await expect(page.getByText("studio.operator.status.v2")).toBeVisible();

  await page.getByRole("button", { name: "Refresh audit status" }).click();
  await expect(page.getByText("unhealthy")).toBeVisible();
  await expect(page.getByText("AuditPathPermissionDenied")).toBeVisible();

  await page.getByRole("button", { name: "Export audit events" }).click();
  await expect(page.getByText("studio.synth.run")).toBeVisible();
  await expect(page.getByText("operator-missing-role - missing_admin_role")).toBeVisible();

  await page.getByRole("button", { name: "Refresh job status" }).click();
  await expect(page.getByText("1 failed jobs recorded by the local worker manager")).toBeVisible();

  expect(api.requests("/api/studio/operator/status")).toBeGreaterThanOrEqual(2);
  expect(api.requests("/api/studio/audit/status")).toBeGreaterThanOrEqual(2);
  expect(api.requests("/api/studio/audit/export?limit=100")).toBe(1);
  expect(api.requests("/api/studio/jobs/status")).toBe(1);
});

test("admin audit archive controls create, review, and purge archives", async ({ page }) => {
  const mocks = defaultApiMocks();
  mocks.set("/api/studio/audit/quarantine/archive", auditArchiveResult);
  mocks.set("/api/studio/audit/quarantine/archive/validate", auditArchiveValidation);
  mocks.set("/api/studio/audit/quarantine/archive/restore", auditArchiveRestore);
  mocks.set(
    "/api/studio/audit/quarantine/archive/retention?retain_latest=1",
    { sequence: [auditArchiveRetention, auditArchiveRetentionAfterPurge] },
  );
  mocks.set("/api/studio/audit/quarantine/archive/purge", auditArchivePurge);
  const api = await installApiDispatcher(page, mocks);

  await page.goto("/");
  await page.getByRole("button", { name: "Admin" }).first().click();

  await expect(page.getByRole("heading", { name: "Audit archive" })).toBeVisible();
  await expect(page.getByText("No archive retention inventory loaded")).toBeVisible();

  await page.getByRole("spinbutton", { name: "Audit archive limit" }).fill("75");
  await page.getByRole("button", { name: "Create audit quarantine archive" }).click();
  await expect(page.getByText("saqa_sj_archive")).toBeVisible();
  await expect(page.getByText("chain_broken:1, legacy_row:2")).toBeVisible();

  const archivePayload = {
    archive_id: "saqa_sj_archive",
    events: [{ event_hash: "1".repeat(64), quarantine_reason: "chain_broken" }],
    schema_version: "studio.audit-quarantine-archive.v1",
    summary: auditArchiveSummary,
  };
  const manifestPayload = {
    archive_id: "saqa_sj_archive",
    archive_sha256: "2".repeat(64),
    schema_version: "studio.audit-quarantine-archive.v1",
  };
  await page.getByRole("textbox", { name: "Audit archive JSON" }).fill(JSON.stringify(archivePayload));
  await page
    .getByRole("textbox", { name: "Audit archive manifest JSON" })
    .fill(JSON.stringify(manifestPayload));
  await page.getByRole("button", { name: "Validate audit archive restore payload" }).click();
  await expect(page.getByText("valid", { exact: true })).toBeVisible();
  await expect(page.getByText("manifest_recomputed")).toBeVisible();

  await page.getByRole("button", { name: "Materialize audit archive restore" }).click();
  await expect(page.getByText("sj_restore")).toBeVisible();
  await expect(page.getByText("Restore artifacts")).toBeVisible();

  await page.getByRole("spinbutton", { name: "Audit archive retain latest" }).fill("1");
  await page.getByRole("button", { name: "Review audit archive retention" }).click();
  await expect(page.getByText("saqa_sj_new")).toBeVisible();
  await expect(page.getByText("saqa_sj_old")).toBeVisible();
  await expect(page.getByText("prune_candidate")).toBeVisible();

  await page.getByRole("button", { name: "Purge audit archive prune candidates" }).click();
  await expect(page.getByText("1 purged / 1 retained")).toBeVisible();

  expect(api.bodies("/api/studio/audit/quarantine/archive")).toEqual([{ limit: 75 }]);
  expect(api.bodies("/api/studio/audit/quarantine/archive/validate")).toEqual([
    { archive: archivePayload, manifest: manifestPayload },
  ]);
  expect(api.bodies("/api/studio/audit/quarantine/archive/restore")).toEqual([
    { archive: archivePayload, manifest: manifestPayload },
  ]);
  expect(api.bodies("/api/studio/audit/quarantine/archive/purge")).toEqual([
    { retain_latest: 1 },
  ]);
  expect(api.requests("/api/studio/audit/quarantine/archive/retention?retain_latest=1")).toBe(2);
});

test("admin evidence bundle form submits simulation and analysis result payloads", async ({ page }) => {
  const mocks = defaultApiMocks();
  mocks.set("/api/studio/jobs/sj_browser/artifacts/evidence/simulations/000.json", {
    binaryBody: "{\"kind\":\"simulation\"}\n",
    contentType: "application/json",
  });
  const api = await installApiDispatcher(page, mocks);

  await page.goto("/");
  await page.getByRole("button", { name: "Admin" }).first().click();

  const simulationPayload = {
    dt: 0.1,
    n_steps: 2,
    run_metadata: {
      dt: 0.1,
      evidence_classification: "simulation",
      input_sha256: "1".repeat(64),
      n_steps: 2,
      result_sha256: "2".repeat(64),
      sample_count: 2,
      schema_version: "studio.simulation-run.v1",
      source: "ode",
      spike_count: 0,
      state_variables: ["v"],
    },
    spike_count: 0,
    states: { v: [0, 0.1] },
    time: [0, 0.1],
  };
  const analysisPayload = {
    analysis_metadata: {
      analysis_type: "fi_curve",
      evidence_classification: "analysis",
      input_sha256: "3".repeat(64),
      output_keys: ["currents", "rates"],
      result_sha256: "4".repeat(64),
      schema_version: "studio.analysis-result.v1",
      source: "ode",
    },
    currents: [0, 1],
    rates: [0, 10],
  };
  const defaultFlowRunPayload = {
    action_order: ["auto_tune_adaptive_precision"],
    executed_count: 1,
    execution_time_ms: 1,
    flow_id: "studio_default_adaptive_precision_v1",
    preset_id: "fpga_precision",
    reproducibility_manifest: {
      hash_algorithm: "sha256",
      inputs_fingerprint_sha256: "7".repeat(64),
      run_fingerprint_sha256: "8".repeat(64),
    },
    results: [],
    schema_version: "sc-neurocore.studio.default-flow-run.v1",
  };
  const defaultFlowAttestationPayload = {
    attestation_fingerprint_sha256: "9".repeat(64),
    flow_id: "studio_default_adaptive_precision_v1",
    inputs_fingerprint_sha256: "7".repeat(64),
    plan_fingerprint_sha256: "a".repeat(64),
    preset_id: "fpga_precision",
    run_fingerprint_sha256: "8".repeat(64),
    schema_version: "sc-neurocore.studio.default-flow-attestation.v1",
  };

  await page.getByRole("textbox", { name: "Evidence simulation JSON" }).fill(
    JSON.stringify(simulationPayload),
  );
  await page.getByRole("textbox", { name: "Evidence analysis JSON" }).fill(
    JSON.stringify(analysisPayload),
  );
  await page.getByRole("textbox", { name: "Evidence default-flow run JSON" }).fill(
    JSON.stringify(defaultFlowRunPayload),
  );
  await page.getByRole("textbox", { name: "Evidence default-flow attestation JSON" }).fill(
    JSON.stringify(defaultFlowAttestationPayload),
  );
  await page.getByRole("button", { name: "Create evidence bundle" }).click();

  await expect(page.getByText("seb_sj_browser")).toBeVisible();
  await expect(page.getByText("analysis_result:1")).toBeVisible();
  await expect(page.getByText("simulation:1")).toBeVisible();
  await expect(page.getByText("simulation - evidence/simulations/000.json")).toBeVisible();
  await expect(page.getByText("analysis - evidence/analyses/000.json")).toBeVisible();
  await expect(page.getByText("unclassified - sha ffffffffffff")).toBeVisible();
  await expect(page.getByText("evidence/simulations/000.json", { exact: true })).toBeVisible();
  await expect(page.getByText("256 B - sha cccccccccccc")).toBeVisible();

  const bodies = api.bodies("/api/studio/evidence/bundle");
  expect(bodies).toHaveLength(1);
  expect(bodies[0]).toMatchObject({
    analysis_results: [analysisPayload],
    default_flow_attestations: [defaultFlowAttestationPayload],
    default_flow_runs: [defaultFlowRunPayload],
    include_audit: true,
    simulation_results: [simulationPayload],
  });

  await page
    .getByRole("button", { name: "Download evidence artifact evidence/simulations/000.json" })
    .click();
  const artifactPath = "/api/studio/jobs/sj_browser/artifacts/evidence/simulations/000.json";
  expect(api.requests(artifactPath)).toBe(1);
  expect(api.headers(artifactPath)[0]).toMatchObject({
    authorization: "Bearer browser-token",
  });
});

test("admin job rows can seed evidence bundle job IDs", async ({ page }) => {
  const mocks = defaultApiMocks();
  mocks.set("/api/studio/jobs", artifactJobList);
  const api = await installApiDispatcher(page, mocks);

  await page.goto("/");
  await page.getByRole("button", { name: "Admin" }).first().click();

  await expect(page.getByText("compiler - sj_artifact")).toBeVisible();
  await expect(page.getByText("2 artifacts - 1 evidence")).toBeVisible();
  await expect(page.getByText("reports/result.txt, compiler/compile-evidence.json")).toBeVisible();

  await page.getByRole("button", { name: "Add sj_artifact to evidence bundle" }).click();
  await expect(page.getByRole("textbox", { name: "Evidence job IDs" })).toHaveValue("sj_artifact");

  await page.getByRole("button", { name: "Create evidence bundle" }).click();
  await expect(page.getByText("seb_sj_browser")).toBeVisible();

  const bodies = api.bodies("/api/studio/evidence/bundle");
  expect(bodies).toHaveLength(1);
  expect(bodies[0]).toMatchObject({
    include_audit: true,
    job_ids: ["sj_artifact"],
  });
});

test("project evidence strip exports saved project bundles", async ({ page }) => {
  const mocks = defaultApiMocks();
  mocks.set("/api/project/save", projectSaveResult);
  mocks.set("/api/studio/jobs/sj_browser/artifacts/evidence/simulations/000.json", {
    binaryBody: "{\"kind\":\"simulation\"}\n",
    contentType: "application/json",
  });
  const api = await installApiDispatcher(page, mocks);

  page.once("dialog", async (dialog) => {
    await dialog.accept("saved-network");
  });
  await page.goto("/");

  await page.getByRole("button", { name: "Save" }).first().click();
  await expect(page.getByText("project_workspace")).toBeVisible();
  await expect(page.getByText("state sha aaaaaaaaaaaa")).toBeVisible();
  await expect(page.getByText("project sha bbbbbbbbbbbb")).toBeVisible();

  await page
    .getByRole("button", { name: "Export saved-network project evidence bundle" })
    .click();
  await expect(page.getByText("seb_sj_browser")).toBeVisible();
  await expect(page.getByText("sj_browser", { exact: true })).toBeVisible();
  await expect(page.getByText("evidence/simulations/000.json", { exact: true })).toBeVisible();
  await expect(page.getByText("256 B - sha cccccccccccc")).toBeVisible();

  const bodies = api.bodies("/api/studio/evidence/bundle");
  expect(bodies).toHaveLength(1);
  expect(bodies[0]).toMatchObject({
    command_replay: {
      method: "POST",
      request_sha256: "b".repeat(64),
      route: "/api/project/save",
    },
    include_audit: true,
    project_name: "saved-network",
  });

  await page
    .getByRole("button", { name: "Download project evidence artifact evidence/simulations/000.json" })
    .click();
  const artifactPath = "/api/studio/jobs/sj_browser/artifacts/evidence/simulations/000.json";
  expect(api.requests(artifactPath)).toBe(1);
  expect(api.headers(artifactPath)[0]).toMatchObject({
    authorization: "Bearer browser-token",
  });
});

test("synthesis dashboard renders target provenance matrix from all-target run", async ({ page }) => {
  const verilog = "module test(input clk, output y); assign y = clk; endmodule";
  const api = await installApiDispatcher(
    page,
    new Map<string, ApiMockPayload>([
      [
        "/api/studio/capabilities",
        registry([capabilityRegistryContract, compilerCapability, synthesisCapability]),
      ],
      ["/api/models", []],
      ["/api/templates", []],
      ["/api/presets", []],
      ["/api/synth/tools-status", {
        nextpnr_ice40: { available: false, version: null },
        yosys: { available: true, version: "Yosys 0.test" },
      }],
      ["/api/ir/emit-sv-direct", {
        chars: verilog.length,
        compile_traceability: {
          evidence_classification: "compile",
          input_sha256: "1".repeat(64),
          output: {
            language: "systemverilog",
            module_name: "test",
            rtl_chars: verilog.length,
            rtl_sha256: "2".repeat(64),
          },
          schema_version: "studio.compile-traceability.v1",
          source: "ode",
          source_payload: {
            equations: ["dv/dt = -(v - E_L) / tau_m + I / C"],
            init: { v: -65 },
            params: { C: 1, E_L: -65, tau_m: 10 },
            reset: "v = -65",
            threshold: "v > -50",
          },
          traceability_sha256: "3".repeat(64),
        },
        ir_repr: "%0 = input clk",
        module_name: "test",
        verilog,
      }],
      ["/api/studio/evidence/bundle", {
        artifact_paths: [
          "evidence/replay.json",
          "evidence/manifest.json",
        ],
        artifacts: [
          {
            relative_path: "evidence/replay.json",
            sha256: "c".repeat(64),
            size_bytes: 128,
          },
        ],
        bundle_id: "seb_compile",
        job_id: "sj_compile",
        manifest: {
          entries: [
            { type: "command_replay" },
            { type: "manifest" },
          ],
        },
        schema_version: "studio.evidence-bundle.v1",
        summary: {
          artifact_path_count: 2,
          entry_count: 2,
          entry_type_counts: {
            command_replay: 1,
            manifest: 1,
          },
          evidence_classification_counts: {},
          source_job_count: 0,
          source_job_kind_counts: {},
          source_job_owner_counts: {},
        },
      }],
      ["/api/studio/operator/status", operatorStatus],
      ["/api/studio/jobs", jobList],
      ["/api/synth/multi-target", {
        supported: ["ice40", "gowin"],
        target_provenance_matrix: {
          matrix_sha256: "a".repeat(64),
          schema_version: "studio.synthesis-target-provenance-matrix.v1",
          targets: {
            gowin: {
              capacity: { brams: 41, dsps: 0, ffs: 20736, luts: 20736 },
              device: null,
              evidence_classification: "synthesis",
              pnr_ready: true,
              pnr_tool: null,
              schema_version: "studio.synthesis-target-provenance.v1",
              synthesis_command: "synth_gowin",
              synthesis_ready: true,
              target: "gowin",
              tools: [
                {
                  available: true,
                  executable: "yosys",
                  key: "yosys",
                  role: "synthesis",
                  version: "Yosys 0.test",
                },
              ],
            },
            ice40: {
              capacity: { brams: 30, dsps: 0, ffs: 5280, luts: 5280 },
              device: "up5k",
              evidence_classification: "synthesis",
              pnr_ready: false,
              pnr_tool: "nextpnr-ice40",
              schema_version: "studio.synthesis-target-provenance.v1",
              synthesis_command: "synth_ice40",
              synthesis_ready: true,
              target: "ice40",
              tools: [
                {
                  available: true,
                  executable: "yosys",
                  key: "yosys",
                  role: "synthesis",
                  version: "Yosys 0.test",
                },
                {
                  available: false,
                  executable: "nextpnr-ice40",
                  key: "nextpnr_ice40",
                  role: "place_and_route",
                  version: null,
                },
              ],
            },
          },
        },
        targets: {
          gowin: {
            capacity: { brams: 41, dsps: 0, ffs: 20736, luts: 20736 },
            log_excerpt: "",
            resources: { brams: 0, cells: 1, dsps: 0, ffs: 1, luts: 2, wires: 1 },
            success: true,
            target: "gowin",
            target_provenance: {
              capacity: { brams: 41, dsps: 0, ffs: 20736, luts: 20736 },
              device: null,
              evidence_classification: "synthesis",
              pnr_ready: true,
              pnr_tool: null,
              schema_version: "studio.synthesis-target-provenance.v1",
              synthesis_command: "synth_gowin",
              synthesis_ready: true,
              target: "gowin",
              tools: [],
            },
            utilisation: { brams: 0, dsps: 0, ffs: 0, luts: 0 },
          },
          ice40: {
            capacity: { brams: 30, dsps: 0, ffs: 5280, luts: 5280 },
            log_excerpt: "",
            resources: { brams: 0, cells: 1, dsps: 0, ffs: 1, luts: 2, wires: 1 },
            success: true,
            target: "ice40",
            target_provenance: {
              capacity: { brams: 30, dsps: 0, ffs: 5280, luts: 5280 },
              device: "up5k",
              evidence_classification: "synthesis",
              pnr_ready: false,
              pnr_tool: "nextpnr-ice40",
              schema_version: "studio.synthesis-target-provenance.v1",
              synthesis_command: "synth_ice40",
              synthesis_ready: true,
              target: "ice40",
              tools: [],
            },
            utilisation: { brams: 0, dsps: 0, ffs: 0, luts: 0 },
          },
        },
      }],
    ]),
  );

  await page.goto("/");
  await page.getByRole("button", { name: "ODE", exact: true }).click();
  await page.getByRole("button", { name: "SV", exact: true }).click();
  await expect(page.getByText("SystemVerilog")).toBeVisible();
  await expect(page.getByText("trace 333333333333")).toBeVisible();
  await page.getByRole("button", { name: "Export compile evidence bundle" }).click();
  await expect(page.getByText("bundle seb_compile")).toBeVisible();

  const evidenceBodies = api.bodies("/api/studio/evidence/bundle");
  expect(evidenceBodies).toHaveLength(1);
  expect(evidenceBodies[0]).toMatchObject({
    command_replay: {
      method: "POST",
      request_sha256: "1".repeat(64),
      route: "/api/ir/emit-sv-direct",
    },
    include_audit: true,
    project_name: "compile-test",
  });

  await page.getByRole("button", { name: "FPGA" }).first().click();
  await page.getByRole("button", { name: "All Targets" }).click();

  await expect(page.getByText("Target provenance matrix")).toBeVisible();
  await expect(page.getByText("aaaaaaaaaaaa")).toBeVisible();
  const matrixTable = page.getByRole("table").nth(1);
  await expect(matrixTable.getByRole("cell", { exact: true, name: "ICE40" })).toBeVisible();
  await expect(matrixTable.getByRole("cell", { exact: true, name: "up5k" })).toBeVisible();
  await expect(matrixTable.getByText("missing - nextpnr-ice40 missing")).toBeVisible();
  await expect(matrixTable.getByText("ready - not required")).toBeVisible();
});

test("capability menu exposes unavailable requirements", async ({ page }) => {
  const mocks = defaultApiMocks();
  mocks.set(
    "/api/studio/capabilities",
    registry([
      capabilityRegistryContract,
      simulationCapability,
      analysisUnavailable,
      synthesisUnavailable,
    ]),
  );
  await installApiDispatcher(page, mocks);

  await page.goto("/");
  await page.getByText("2/4 ready").click();

  const capabilityMenu = page.locator(".capability-menu");
  await expect(capabilityMenu.getByText("Analysis Suite")).toBeVisible();
  await expect(capabilityMenu.getByText("analysis: analysis endpoint disabled")).toBeVisible();
  await expect(capabilityMenu.getByText("Synthesis Dashboard")).toBeVisible();
  await expect(capabilityMenu.getByText("yosys: yosys unavailable")).toBeVisible();
});

test("unavailable panel contracts disable toolbar and keyboard activation", async ({ page }) => {
  const mocks = defaultApiMocks();
  mocks.set(
    "/api/studio/capabilities",
    registry([
      capabilityRegistryContract,
      simulationCapability,
      analysisUnavailable,
    ]),
  );
  await installApiDispatcher(page, mocks);

  await page.goto("/");

  await expect(page.getByRole("button", { name: "f-I" }).first()).toBeDisabled();
  await expect(page.getByRole("button", { name: "f-I" }).last()).toBeDisabled();

  await page.keyboard.press("3");

  await expect(page.getByText("Analysis endpoints are unavailable.")).toHaveCount(0);
  await expect(page.locator("canvas")).toBeVisible();
});

test("missing active panel capability fails closed at startup", async ({ page }) => {
  const mocks = defaultApiMocks();
  mocks.set("/api/studio/capabilities", registry([capabilityRegistryContract]));
  const api = await installApiDispatcher(page, mocks);

  await page.goto("/");

  await expect(page.locator(".capability-blocked-title", { hasText: "Trace" })).toBeVisible();
  await expect(page.getByText("Backend capability contract is missing from the registry.")).toBeVisible();
  await page.keyboard.press("Space");
  await page.waitForTimeout(100);
  expect(api.requests("/api/simulate")).toBe(0);
});
