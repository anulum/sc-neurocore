// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio guided operator run browser contract
import { expect, test, type Page } from "@playwright/test";

test.setTimeout(60_000);

interface ApiMockSequence {
  sequence: object[];
}

type ApiMockPayload = object | ApiMockSequence;

interface ApiDispatcher {
  bodies: (path: string) => unknown[];
  requests: (path: string) => number;
}

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
): Promise<ApiDispatcher> {
  const bodies = new Map<string, unknown[]>();
  const counts = new Map<string, number>();
  await page.route((url) => url.pathname.startsWith("/api/"), async (route) => {
    const url = new URL(route.request().url());
    const path = `${url.pathname}${url.search}`;
    const count = counts.get(path) ?? 0;
    counts.set(path, count + 1);
    const postData = route.request().postData();
    if (postData !== null) {
      const recorded = bodies.get(path) ?? [];
      recorded.push(JSON.parse(postData) as unknown);
      bodies.set(path, recorded);
    }
    const payload = mocks.get(path);
    if (payload === undefined) {
      await route.fulfill({
        contentType: "application/json",
        json: { detail: `unmocked ${path}` },
        status: 404,
      });
      return;
    }
    const selectedPayload = "sequence" in payload
      ? payload.sequence[Math.min(count, payload.sequence.length - 1)]
      : payload;
    await route.fulfill({ contentType: "application/json", json: selectedPayload, status: 200 });
  });
  return {
    bodies: (path: string) => bodies.get(path) ?? [],
    requests: (path: string) => counts.get(path) ?? 0,
  };
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
  allowed_kinds: ["compiler", "synthesis", "training"],
  completed_count: 1,
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

const simulationResponse = {
  current_trace: [10, 10, 10],
  dt: 0.1,
  n_steps: 3,
  run_metadata: {
    dt: 0.1,
    evidence_classification: "simulation",
    input_sha256: "1".repeat(64),
    n_steps: 3,
    result_sha256: "2".repeat(64),
    sample_count: 3,
    schema_version: "studio.simulation-run.v1",
    source: "ode",
    spike_count: 1,
    state_variables: ["v"],
    status: "completed",
  },
  spike_count: 1,
  spikes: [0.2],
  states: { v: [-65, -61, -55] },
  stats: { isi_cv: null, isi_histogram: null, isi_mean_ms: null, rate_hz: 10 },
  time: [0, 0.1, 0.2],
};

const fiCurveResponse = {
  analysis_metadata: {
    analysis_type: "fi_curve",
    evidence_classification: "analysis",
    input_sha256: "3".repeat(64),
    output_keys: ["currents", "rates"],
    result_sha256: "4".repeat(64),
    schema_version: "studio.analysis-result.v1",
    source: "ode",
    status: "completed",
  },
  currents: [0, 10, 20],
  rates: [0, 10, 22],
};

const verilog = "module lif(input clk, output spike); assign spike = clk; endmodule";

const compileResponse = {
  chars: verilog.length,
  compile_traceability: {
    evidence_classification: "compile",
    input_sha256: "5".repeat(64),
    output: {
      language: "systemverilog",
      module_name: "lif",
      rtl_chars: verilog.length,
      rtl_sha256: "6".repeat(64),
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
    status: "completed",
    traceability_sha256: "7".repeat(64),
  },
  ir_repr: "%0 = input current",
  module_name: "lif",
  verilog,
};

const synthesisResponse = {
  capacity: { brams: 30, dsps: 0, ffs: 5280, luts: 5280 },
  log_excerpt: "synthesis completed",
  resources: { brams: 0, cells: 2, dsps: 0, ffs: 1, luts: 2, wires: 1 },
  success: true,
  target: "ice40",
  target_provenance: {
    capacity: { brams: 30, dsps: 0, ffs: 5280, luts: 5280 },
    device: "up5k",
    evidence_classification: "synthesis",
    pnr_ready: false,
    pnr_tool: "nextpnr-ice40",
    provenance_grade: "tool_backed",
    schema_version: "studio.synthesis-target-provenance.v1",
    status: "completed",
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
    ],
  },
  utilisation: { brams: 0, dsps: 0, ffs: 0.02, luts: 0.04 },
};

const synthesisJobList = {
  jobs: [
    {
      artifacts: [
        { relative_path: "synthesis/result.json", sha256: "8".repeat(64), size_bytes: 512 },
        { relative_path: "synthesis/evidence.json", sha256: "9".repeat(64), size_bytes: 384 },
      ],
      created_at_utc: "2026-06-28T01:00:00Z",
      error: null,
      execution_model: "process",
      finished_at_utc: "2026-06-28T01:00:02Z",
      job_id: "sj_guided_synthesis",
      kind: "synthesis",
      owner: "studio-synthesis",
      request_id: "req-guided-synth",
      result: null,
      started_at_utc: "2026-06-28T01:00:01Z",
      status: "completed",
    },
  ],
  schema_version: "studio.jobs.list.v1",
};

const evidenceBundle = {
  artifact_paths: ["evidence/jobs/sj_guided_synthesis/artifacts/synthesis/evidence.json"],
  artifacts: [
    {
      relative_path: "evidence/jobs/sj_guided_synthesis/artifacts/synthesis/evidence.json",
      sha256: "a".repeat(64),
      size_bytes: 384,
    },
  ],
  bundle_id: "seb_guided",
  job_id: "sj_guided_bundle",
  manifest: { entries: [{ type: "action_evidence" }] },
  schema_version: "studio.evidence-bundle.v1",
  summary: {
    artifact_path_count: 1,
    entry_count: 1,
    entry_type_counts: { action_evidence: 1 },
    evidence_classification_counts: { synthesis: 1 },
    source_job_count: 1,
    source_job_kind_counts: { synthesis: 1 },
    source_job_owner_counts: { "studio-synthesis": 1 },
  },
};

function guidedMocks(): Map<string, ApiMockPayload> {
  return new Map<string, ApiMockPayload>([
    ["/api/studio/capabilities", {
      capabilities: [
        capability({ capability_id: "studio.capability_registry", title: "Capability Registry", ui_placement: "Admin" }),
        capability({ capability_id: "studio.simulation_workbench", title: "Simulation Workbench", ui_placement: "Trace" }),
        capability({ capability_id: "studio.analysis_suite", title: "Analysis Suite", ui_placement: "Analysis" }),
        capability({ capability_id: "studio.compiler_inspector", title: "Compiler Inspector", ui_placement: "IR" }),
        capability({ capability_id: "studio.synthesis_dashboard", title: "Synthesis Dashboard", ui_placement: "FPGA" }),
        capability({ capability_id: "studio.training_monitor", title: "Training Monitor", ui_placement: "Training" }),
      ],
    }],
    ["/api/studio/auth/session", { authenticated: true, principal_id: "svc-admin", roles: ["studio.admin"] }],
    ["/api/studio/audit/status", auditStatus],
    ["/api/studio/jobs", { sequence: [{ jobs: [], schema_version: "studio.jobs.list.v1" }, synthesisJobList] }],
    ["/api/studio/operator/status", operatorStatus],
    ["/api/models", []],
    ["/api/templates", []],
    ["/api/presets", []],
    ["/api/simulate", simulationResponse],
    ["/api/fi-curve", fiCurveResponse],
    ["/api/ir/emit-sv-direct", compileResponse],
    ["/api/synth/run", synthesisResponse],
    ["/api/studio/evidence/bundle", evidenceBundle],
  ]);
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("sc-studio-onboarding-dismissed", "true");
    window.sessionStorage.setItem("sc-neurocore-studio-auth-token", "browser-token");
  });
});

test("guided operator run executes one ODE workflow through evidence export", async ({ page }) => {
  const api = await installApiDispatcher(page, guidedMocks());

  await page.goto("/");
  await page.getByRole("button", { exact: true, name: "ODE" }).click();

  const runNext = page.getByRole("button", { name: "Run next guided step" });
  await expect(runNext).toContainText("Run simulation");
  await runNext.click();
  await expect(runNext).toContainText("Run f-I analysis");
  await runNext.click();
  await expect(runNext).toContainText("Skip training");
  await runNext.click();
  await expect(runNext).toContainText("Compile RTL");
  await runNext.click();
  await expect(runNext).toContainText("Run synthesis");
  await runNext.click();
  await expect(runNext).toContainText("Export evidence");
  await expect(page.getByText("Evidence ready")).toBeVisible();
  await runNext.click();

  await expect(page.getByText(/Last export:/)).toBeVisible();
  await expect(page.getByText("Workflow complete", { exact: true }).first()).toBeVisible();

  expect(api.requests("/api/simulate")).toBe(1);
  expect(api.requests("/api/fi-curve")).toBe(1);
  expect(api.requests("/api/ir/emit-sv-direct")).toBe(1);
  expect(api.requests("/api/synth/run")).toBe(1);
  expect(api.requests("/api/studio/evidence/bundle")).toBe(0);
});
