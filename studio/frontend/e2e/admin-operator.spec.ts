import { expect, test, type Page } from "@playwright/test";

const capabilityRegistry = {
  capabilities: [
    {
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
    },
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
