// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — App analysis-job-host browser contract (W12-E/F real surface)
import { expect, test, type Page } from "@playwright/test";

test.setTimeout(60_000);

interface ApiMockSequence {
  sequence: object[];
}

type ApiMockPayload = object | ApiMockSequence;

function capability(overrides: Record<string, unknown>): Record<string, unknown> {
  return {
    capability_id: "studio.capability_registry",
    docs_path: "docs/studio/index.md",
    evidence: ["browser_contract"],
    healthy: true,
    message: "Capability is available.",
    requirements: [{ available: true, detail: "available", name: "studio.platform" }],
    status: "stable",
    summary: "Studio capability.",
    title: "Capability Registry",
    ui_placement: "Admin",
    ...overrides,
  };
}

async function installApiDispatcher(
  page: Page,
  mocks: Map<string, ApiMockPayload>,
): Promise<{ requests: (path: string) => number }> {
  const counts = new Map<string, number>();
  await page.route((url) => url.pathname.startsWith("/api/"), async (route) => {
    const url = new URL(route.request().url());
    const path = url.pathname;
    const count = counts.get(path) ?? 0;
    counts.set(path, count + 1);
    const payload = mocks.get(path);
    if (payload === undefined) {
      await route.fulfill({
        contentType: "application/json",
        json: { detail: `unmocked ${path}` },
        status: 404,
      });
      return;
    }
    const selected = "sequence" in payload
      ? payload.sequence[Math.min(count, payload.sequence.length - 1)]
      : payload;
    await route.fulfill({ contentType: "application/json", json: selected, status: 200 });
  });
  return { requests: (path: string) => counts.get(path) ?? 0 };
}

const auditStatus = {
  configured: true,
  healthy: true,
  last_error: null,
  path_configured: true,
  sink_type: "jsonl",
};

const jobStatus = {
  active_count: 0,
  allowed_kinds: ["analysis", "compiler", "synthesis", "training"],
  completed_count: 0,
  configured: true,
  failed_count: 0,
  process_count: 0,
  resource_profiles: [],
  schema_version: "studio.jobs.status.v1",
  thread_count: 0,
  timed_out_count: 0,
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
    healthy_count: 6,
    stable_count: 6,
    total_count: 6,
    unavailable_count: 0,
  },
  deployment_profile: "production",
  identity: { configured: true, header_principal_allowed: false, mode: "service_account" },
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

const fiResult = {
  analysis_metadata: {
    analysis_type: "fi_curve",
    evidence_classification: "analysis",
    input_sha256: "c".repeat(64),
    output_keys: ["currents", "rates"],
    result_sha256: "d".repeat(64),
    schema_version: "studio.analysis-result.v1",
    source: "ode",
    status: "completed",
  },
  currents: [0, 10, 20],
  rates: [0, 5, 12],
};

const analysisReceipt = {
  analysis: "fi_curve",
  execution_mode: "async_job",
  job: {
    artifacts: [],
    created_at_utc: "2026-07-20T00:00:00Z",
    error: null,
    execution_model: "thread",
    finished_at_utc: null,
    job_id: "sj_analysis_host",
    kind: "analysis",
    owner: "studio",
    request_id: null,
    result: null,
    started_at_utc: null,
    status: "pending",
  },
  job_id: "sj_analysis_host",
  schema_version: "studio.analysis.job.v1",
  status_route: "/api/studio/jobs/sj_analysis_host",
};

const analysisJobPoll = {
  sequence: [
    {
      artifacts: [],
      created_at_utc: "2026-07-20T00:00:00Z",
      error: null,
      execution_model: "thread",
      finished_at_utc: null,
      job_id: "sj_analysis_host",
      kind: "analysis",
      owner: "studio",
      request_id: null,
      result: null,
      started_at_utc: "2026-07-20T00:00:01Z",
      status: "running",
    },
    {
      artifacts: [],
      created_at_utc: "2026-07-20T00:00:00Z",
      error: null,
      execution_model: "thread",
      finished_at_utc: "2026-07-20T00:00:02Z",
      job_id: "sj_analysis_host",
      kind: "analysis",
      owner: "studio",
      request_id: null,
      result: fiResult,
      started_at_utc: "2026-07-20T00:00:01Z",
      status: "completed",
    },
  ],
};

function hostMocks(): Map<string, ApiMockPayload> {
  return new Map<string, ApiMockPayload>([
    ["/api/studio/capabilities", {
      capabilities: [
        capability({
          capability_id: "studio.capability_registry",
          title: "Capability Registry",
          ui_placement: "Admin",
        }),
        capability({
          capability_id: "studio.simulation_workbench",
          title: "Simulation Workbench",
          ui_placement: "Trace",
        }),
        capability({
          capability_id: "studio.analysis_suite",
          title: "Analysis Suite",
          ui_placement: "Analysis",
        }),
        capability({
          capability_id: "studio.compiler_inspector",
          title: "Compiler Inspector",
          ui_placement: "IR",
        }),
        capability({
          capability_id: "studio.synthesis_dashboard",
          title: "Synthesis Dashboard",
          ui_placement: "FPGA",
        }),
        capability({
          capability_id: "studio.training_monitor",
          title: "Training Monitor",
          ui_placement: "Training",
        }),
      ],
    }],
    ["/api/studio/auth/session", {
      authenticated: true,
      principal_id: "svc-admin",
      roles: ["studio.admin"],
    }],
    ["/api/studio/audit/status", auditStatus],
    ["/api/studio/jobs", { jobs: [], schema_version: "studio.jobs.list.v1" }],
    ["/api/studio/operator/status", operatorStatus],
    ["/api/models", []],
    ["/api/templates", []],
    ["/api/presets", []],
    ["/api/analysis/jobs", analysisReceipt],
    ["/api/studio/jobs/sj_analysis_host", analysisJobPoll],
  ]);
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("sc-studio-onboarding-dismissed", "true");
    window.sessionStorage.setItem("sc-neurocore-studio-auth-token", "browser-token");
  });
});

test("App mounts analysis-job-host and completes async fi_curve job", async ({ page }) => {
  const dispatcher = await installApiDispatcher(page, hostMocks());
  await page.goto("/");

  const host = page.getByTestId("analysis-job-host");
  await expect(host).toBeVisible();
  await expect(page.getByTestId("analysis-job-control")).toBeVisible();
  await expect(page.getByTestId("analysis-job-control-selection")).toContainText("f-I");

  // Capability wiring: host title comes from panel state message.
  await expect(host).toHaveAttribute("title", /.+/);

  // Kind follows active tab — open sensitivity tab before submit.
  await page.getByRole("button", { name: "Sens", exact: true }).click();
  await expect(page.getByTestId("analysis-job-control-selection")).toContainText(
    "sensitivity",
    { timeout: 5_000 },
  );

  // Return to f-I tab (label "f-I" on tab strip).
  await page.getByRole("button", { name: "f-I", exact: true }).click();
  await expect(page.getByTestId("analysis-job-control-selection")).toContainText("f-I");

  const submit = page.getByTestId("analysis-job-control-submit");
  await expect(submit).toBeEnabled();
  await submit.click();

  await expect(page.getByTestId("analysis-job-control-phase")).toContainText(
    /pending|running|completed/i,
    { timeout: 10_000 },
  );
  await expect(page.getByTestId("analysis-job-control-completed-summary")).toBeVisible({
    timeout: 15_000,
  });
  await expect(page.getByTestId("analysis-job-meta-type")).toHaveText("fi_curve");
  await expect(page.getByTestId("analysis-job-meta-status")).toHaveText("completed");

  expect(dispatcher.requests("/api/analysis/jobs")).toBeGreaterThanOrEqual(1);
  expect(dispatcher.requests("/api/studio/jobs/sj_analysis_host")).toBeGreaterThanOrEqual(1);
});
