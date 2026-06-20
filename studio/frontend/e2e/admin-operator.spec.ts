import { expect, test, type Page } from "@playwright/test";

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

const jobStatus = {
  active_count: 1,
  allowed_kinds: ["compiler", "synthesis", "training"],
  completed_count: 7,
  configured: true,
  failed_count: 0,
  resource_profiles: [
    {
      default_timeout_seconds: 120,
      execution_models: ["thread", "process"],
      kind: "compiler",
      max_artifact_bytes: 16777216,
    },
  ],
  schema_version: "studio.jobs.status.v1",
  timed_out_count: 0,
};

const jobList = {
  jobs: [],
  schema_version: "studio.jobs.list.v1",
};

const operatorStatus = {
  audit: auditStatus,
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
  sequence: object[];
}

type ApiMockPayload = object | ApiMockSequence;

interface ApiDispatcher {
  bodies: (path: string) => unknown[];
  requests: (path: string) => number;
}

async function installApiDispatcher(
  page: Page,
  mocks: Map<string, ApiMockPayload>,
): Promise<ApiDispatcher> {
  await page.unrouteAll({ behavior: "ignoreErrors" });
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
    await route.fulfill({
      contentType: "application/json",
      json: selectedPayload,
      status: 200,
    });
  });
  return {
    bodies: (path: string) => bodies.get(path) ?? [],
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
          { type: "simulation_result" },
          { type: "analysis_result" },
          { type: "default_flow_run" },
          { type: "default_flow_attestation" },
          { type: "manifest" },
        ],
      },
      schema_version: "studio.evidence-bundle.v1",
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
  await expect(page.getByRole("heading", { name: "Audit" })).toBeVisible();
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

test("admin evidence bundle form submits simulation and analysis result payloads", async ({ page }) => {
  const api = await installApiDispatcher(page, defaultApiMocks());

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

  const bodies = api.bodies("/api/studio/evidence/bundle");
  expect(bodies).toHaveLength(1);
  expect(bodies[0]).toMatchObject({
    analysis_results: [analysisPayload],
    default_flow_attestations: [defaultFlowAttestationPayload],
    default_flow_runs: [defaultFlowRunPayload],
    include_audit: true,
    simulation_results: [simulationPayload],
  });
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
