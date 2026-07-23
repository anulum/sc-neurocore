// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

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
