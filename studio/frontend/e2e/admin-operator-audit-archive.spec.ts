// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { expect, test } from "@playwright/test";

import {
  auditArchivePurge,
  auditArchiveRestore,
  auditArchiveResult,
  auditArchiveRetention,
  auditArchiveRetentionAfterPurge,
  auditArchiveSummary,
  auditArchiveValidation,
  defaultApiMocks,
  installApiDispatcher,
} from "./adminOperatorHarness";

test.setTimeout(60_000);


test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("sc-studio-onboarding-dismissed", "true");
    window.sessionStorage.setItem("sc-neurocore-studio-auth-token", "browser-token");
  });
  await installApiDispatcher(page, defaultApiMocks());
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
