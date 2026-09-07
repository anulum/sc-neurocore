// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { expect, test } from "@playwright/test";

import {
  defaultApiMocks,
  installApiDispatcher,
  projectSaveResult,
} from "./adminOperatorHarness";

test.setTimeout(60_000);


test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("sc-studio-onboarding-dismissed", "true");
    window.sessionStorage.setItem("sc-neurocore-studio-auth-token", "browser-token");
  });
  await installApiDispatcher(page, defaultApiMocks());
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

  await page.getByLabel("Save project", { exact: true }).click();
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
  // The click starts the download and does not wait for it; polling asserts
  // the request was made rather than that it had already been made.
  await expect.poll(() => api.requests(artifactPath)).toBe(1);
  expect(api.headers(artifactPath)[0]).toMatchObject({
    authorization: "Bearer browser-token",
  });
});

test("project evidence strip ignores admin bundle artifacts", async ({ page }) => {
  const mocks = defaultApiMocks();
  mocks.set("/api/project/save", projectSaveResult);
  mocks.set("/api/studio/evidence/bundle", {
    sequence: [
      {
        artifact_paths: ["evidence/admin/audit.json"],
        artifacts: [
          {
            relative_path: "evidence/admin/audit.json",
            sha256: "9".repeat(64),
            size_bytes: 64,
          },
        ],
        bundle_id: "seb_admin",
        job_id: "sj_admin",
        manifest: { entries: [{ bundle_path: "evidence/admin/audit.json", type: "audit" }] },
        schema_version: "studio.evidence-bundle.v1",
        summary: {
          artifact_path_count: 1,
          entry_count: 1,
          entry_type_counts: { audit: 1 },
          evidence_classification_counts: { audit: 1 },
          source_job_count: 0,
          source_job_kind_counts: {},
          source_job_owner_counts: {},
        },
      },
      {
        artifact_paths: ["evidence/projects/saved-network.json"],
        artifacts: [
          {
            relative_path: "evidence/projects/saved-network.json",
            sha256: "8".repeat(64),
            size_bytes: 128,
          },
        ],
        bundle_id: "seb_project",
        job_id: "sj_project",
        manifest: {
          entries: [
            {
              bundle_path: "evidence/projects/saved-network.json",
              evidence_classification: "project_workspace",
              type: "project",
            },
          ],
        },
        schema_version: "studio.evidence-bundle.v1",
        summary: {
          artifact_path_count: 1,
          entry_count: 1,
          entry_type_counts: { project: 1 },
          evidence_classification_counts: { project_workspace: 1 },
          source_job_count: 0,
          source_job_kind_counts: {},
          source_job_owner_counts: {},
        },
      },
    ],
  });
  const api = await installApiDispatcher(page, mocks);

  page.once("dialog", async (dialog) => {
    await dialog.accept("saved-network");
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Admin" }).first().click();
  await page.getByRole("button", { name: "Create evidence bundle" }).click();
  await expect(page.getByText("seb_admin")).toBeVisible();
  await expect(page.getByText("evidence/admin/audit.json", { exact: true })).toHaveCount(2);

  await page.getByLabel("Save project", { exact: true }).click();
  await expect(page.getByText("project_workspace")).toBeVisible();
  await expect(page.getByText("evidence/admin/audit.json", { exact: true })).toHaveCount(2);
  await expect(page.getByText("evidence/projects/saved-network.json", { exact: true })).toHaveCount(0);

  await page
    .getByRole("button", { name: "Export saved-network project evidence bundle" })
    .click();
  await expect(page.getByText("seb_project")).toBeVisible();
  await expect(page.getByText("evidence/projects/saved-network.json", { exact: true })).toBeVisible();

  expect(api.bodies("/api/studio/evidence/bundle")).toHaveLength(2);
});
