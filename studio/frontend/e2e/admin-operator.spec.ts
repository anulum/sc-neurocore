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
  schema_version: "studio.jobs.status.v1",
  timed_out_count: 0,
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
  route_policies: { enforced: true },
  schema_version: "studio.operator.status.v1",
};

async function fulfillJson(page: Page, path: string, payload: object): Promise<void> {
  await page.route((url) => `${url.pathname}${url.search}` === path, async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: payload,
      status: 200,
    });
  });
}

async function fulfillJsonSequence(
  page: Page,
  path: string,
  payloads: object[],
): Promise<() => number> {
  let requestCount = 0;
  await page.route((url) => `${url.pathname}${url.search}` === path, async (route) => {
    const index = Math.min(requestCount, payloads.length - 1);
    requestCount += 1;
    await route.fulfill({
      contentType: "application/json",
      json: payloads[index],
      status: 200,
    });
  });
  return () => requestCount;
}

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("sc-studio-onboarding-dismissed", "true");
  });
  await page.route((url) => url.pathname.startsWith("/api/"), async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: { detail: "unmocked Studio browser test route" },
      status: 404,
    });
  });
  await fulfillJson(page, "/api/studio/capabilities", capabilityRegistry);
  await fulfillJson(page, "/api/studio/audit/status", auditStatus);
  await fulfillJson(page, "/api/studio/audit/export?limit=100", auditExport);
  await fulfillJson(page, "/api/studio/jobs/status", jobStatus);
  await fulfillJson(page, "/api/studio/operator/status", operatorStatus);
  await fulfillJson(page, "/api/models", []);
  await fulfillJson(page, "/api/templates", []);
  await fulfillJson(page, "/api/presets", []);
});

test("admin panel renders aggregate operator status", async ({ page }) => {
  await page.goto("/");

  await expect(page.getByText("1/1 ready")).toBeVisible();
  await page.getByRole("button", { name: "Admin" }).first().click();

  await expect(page.getByRole("heading", { name: "Operator" })).toBeVisible();
  await expect(page.getByText("production")).toBeVisible();
  await expect(page.getByText("enforced")).toBeVisible();
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
    route_policies: { enforced: false },
    schema_version: "studio.operator.status.v2",
  };
  const operatorRequests = await fulfillJsonSequence(page, "/api/studio/operator/status", [
    operatorStatus,
    refreshedOperatorStatus,
  ]);
  const auditStatusRequests = await fulfillJsonSequence(page, "/api/studio/audit/status", [
    auditStatus,
    refreshedAuditStatus,
  ]);
  const auditExportRequests = await fulfillJsonSequence(page, "/api/studio/audit/export?limit=100", [
    refreshedAuditExport,
  ]);
  const jobStatusRequests = await fulfillJsonSequence(page, "/api/studio/jobs/status", [
    refreshedJobStatus,
  ]);

  await page.goto("/");
  await page.getByRole("button", { name: "Admin" }).first().click();

  await page.getByRole("button", { name: "Refresh operator status" }).click();
  await expect(page.getByText("development")).toBeVisible();
  await expect(page.getByText("disabled")).toBeVisible();
  await expect(page.getByText("studio.operator.status.v2")).toBeVisible();

  await page.getByRole("button", { name: "Refresh audit status" }).click();
  await expect(page.getByText("unhealthy")).toBeVisible();
  await expect(page.getByText("AuditPathPermissionDenied")).toBeVisible();

  await page.getByRole("button", { name: "Export audit events" }).click();
  await expect(page.getByText("studio.synth.run")).toBeVisible();
  await expect(page.getByText("operator-missing-role - missing_admin_role")).toBeVisible();

  await page.getByRole("button", { name: "Refresh job status" }).click();
  await expect(page.getByText("1 failed jobs recorded by the local worker manager")).toBeVisible();

  expect(operatorRequests()).toBeGreaterThanOrEqual(2);
  expect(auditStatusRequests()).toBeGreaterThanOrEqual(2);
  expect(auditExportRequests()).toBe(1);
  expect(jobStatusRequests()).toBe(1);
});

test("capability menu exposes unavailable requirements", async ({ page }) => {
  await fulfillJson(page, "/api/studio/capabilities", registry([
    capabilityRegistryContract,
    simulationCapability,
    analysisUnavailable,
    synthesisUnavailable,
  ]));

  await page.goto("/");
  await page.getByText("2/4 ready").click();

  const capabilityMenu = page.locator(".capability-menu");
  await expect(capabilityMenu.getByText("Analysis Suite")).toBeVisible();
  await expect(capabilityMenu.getByText("analysis: analysis endpoint disabled")).toBeVisible();
  await expect(capabilityMenu.getByText("Synthesis Dashboard")).toBeVisible();
  await expect(capabilityMenu.getByText("yosys: yosys unavailable")).toBeVisible();
});

test("unavailable panel contracts disable toolbar and keyboard activation", async ({ page }) => {
  await fulfillJson(page, "/api/studio/capabilities", registry([
    capabilityRegistryContract,
    simulationCapability,
    analysisUnavailable,
  ]));

  await page.goto("/");

  await expect(page.getByRole("button", { name: "f-I" }).first()).toBeDisabled();
  await expect(page.getByRole("button", { name: "f-I" }).last()).toBeDisabled();

  await page.keyboard.press("3");

  await expect(page.getByText("Analysis endpoints are unavailable.")).toHaveCount(0);
  await expect(page.locator("canvas")).toBeVisible();
});

test("missing active panel capability fails closed at startup", async ({ page }) => {
  let simulateRequests = 0;
  await fulfillJson(page, "/api/studio/capabilities", registry([capabilityRegistryContract]));
  await page.route((url) => url.pathname === "/api/simulate", async (route) => {
    simulateRequests += 1;
    await route.fulfill({
      contentType: "application/json",
      json: { detail: "simulation should not run while the panel is unavailable" },
      status: 500,
    });
  });

  await page.goto("/");

  await expect(page.locator(".capability-blocked-title", { hasText: "Trace" })).toBeVisible();
  await expect(page.getByText("Backend capability contract is missing from the registry.")).toBeVisible();
  await page.keyboard.press("Space");
  await page.waitForTimeout(100);
  expect(simulateRequests).toBe(0);
});
