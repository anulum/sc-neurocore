// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { expect, test } from "@playwright/test";

import {
  auditExport,
  auditStatus,
  capabilityRegistry,
  defaultApiMocks,
  installApiDispatcher,
  jobList,
  jobStatus,
  operatorStatus,
  type ApiMockPayload,
} from "./adminOperatorHarness";

test.setTimeout(60_000);


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
  const operatorSection = page.locator("section.admin-section").filter({
    has: page.getByRole("heading", { name: "Operator" }),
  });
  await expect(operatorSection.getByText("production", { exact: true }).first()).toBeVisible();
  await expect(operatorSection.getByText("enforced", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("93 total / 71 protected")).toBeVisible();
  await expect(page.getByText("audited")).toBeVisible();
  await expect(operatorSection.getByText("service_account", { exact: true }).first()).toBeVisible();
  await expect(operatorSection.getByText("Ready for configured profile")).toBeVisible();
  await expect(operatorSection.getByText("0 blockers / 0 warnings")).toBeVisible();
  await expect(page.getByText("studio.operator.status.v1")).toBeVisible();
  await expect(page.getByRole("heading", { exact: true, name: "Audit" })).toBeVisible();
  const auditSection = page.locator("section.admin-section").filter({
    has: page.getByRole("heading", { exact: true, name: "Audit" }),
  });
  await expect(auditSection.getByText("jsonl", { exact: true })).toBeVisible();
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
  const operatorSection = page.locator("section.admin-section").filter({
    has: page.getByRole("heading", { name: "Operator" }),
  });
  await expect(operatorSection.getByText("development", { exact: true }).first()).toBeVisible();
  await expect(operatorSection.getByText("disabled", { exact: true }).first()).toBeVisible();
  await expect(operatorSection.getByText("incomplete", { exact: true }).first()).toBeVisible();
  await expect(operatorSection.getByText("Readiness blocked")).toBeVisible();
  await expect(page.getByText("studio.operator.status.v2")).toBeVisible();

  await page.getByRole("button", { name: "Refresh audit status" }).click();
  const auditSection = page.locator("section.admin-section").filter({
    has: page.getByRole("heading", { exact: true, name: "Audit" }),
  });
  await expect(auditSection.getByText("unhealthy", { exact: true }).first()).toBeVisible();
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
