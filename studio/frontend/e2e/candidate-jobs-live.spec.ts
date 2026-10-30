// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Candidate job recovery through a real browser and backend

import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { expect, test, type Page } from "@playwright/test";
import { parseCandidateJob, parseCandidateJobReceipt } from "../src/candidateJobs";
import fixture from "./fixtures/SlowAdaptationAdEx.candidate.json" with { type: "json" };

test.describe.configure({ mode: "serial" });

/** Open the actual candidate workbench without the first-run overlay. */
async function openCandidate(page: Page): Promise<void> {
  await page.addInitScript(() => { localStorage.setItem("sc-studio-onboarding-dismissed", "true"); });
  await page.goto("./");
  await page.getByRole("tab", { name: "Candidate model" }).click();
}

test("a completed job keeps its input identity across draft changes and browser reload", async ({ page }) => {
  await openCandidate(page);
  const draft = page.getByLabel("Candidate package (JSON)", { exact: true });
  await draft.fill(JSON.stringify(fixture));
  await page.getByLabel("Steps", { exact: true }).fill("20");
  const admitted = page.waitForResponse((response) => new URL(response.url()).pathname === "/api/candidates/simulate");
  await page.getByRole("button", { name: "Simulate", exact: true }).click();
  const response = await admitted;
  expect(response.status()).toBe(202);
  const receipt = parseCandidateJobReceipt(await response.json(), "simulate");
  const selected = page.getByRole("status", { name: "Selected candidate job" });
  await expect(page.locator("#candidate-outcome")).toContainText("spikes in 20 steps");
  await draft.fill('{"name":"ChangedDraft"}');
  await expect(selected).toContainText(fixture.name);
  await expect(selected).toContainText(receipt.job.admission.candidate_sha256);
  await page.reload();
  await page.getByRole("tab", { name: "Candidate model" }).click();
  await page.getByRole("list", { name: "Stored candidate jobs" }).getByRole("button", {
    name: new RegExp(`${receipt.job.job_id}$`),
  }).click();
  await expect(page.locator("#candidate-outcome")).toContainText("spikes in 20 steps");
  const downloaded = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download request.json", exact: true }).click();
  const download = await downloaded;
  const path = await download.path();
  const bytes = readFileSync(path);
  expect(createHash("sha256").update(bytes).digest("hex")).toBe(receipt.job.admission.request_sha256);
  const request: unknown = JSON.parse(bytes.toString("utf8"));
  expect(request).toEqual({ candidate: fixture, current: 0, steps: 20 });
});

test("a long reference job is cancelled through the browser and retained after reload", async ({ page }) => {
  await openCandidate(page);
  const candidate = structuredClone(fixture);
  const reference = candidate.reference_tests[0];
  if (reference === undefined) throw new Error("The real candidate fixture has no reference test.");
  candidate.reference_tests = Array.from({ length: 16 }, (_, index) => ({
    ...reference, name: `cancel-reference-${String(index)}`, steps: 100_000,
  }));
  await page.getByLabel("Candidate package (JSON)", { exact: true }).fill(JSON.stringify(candidate));
  const admitted = page.waitForResponse((response) => new URL(response.url()).pathname === "/api/candidates/review-packet");
  await page.getByRole("button", { name: "Run reference tests", exact: true }).click();
  const response = await admitted;
  expect(response.status()).toBe(202);
  const receipt = parseCandidateJobReceipt(await response.json(), "review");
  await page.getByRole("button", { name: "Cancel candidate job", exact: true }).click();
  await expect(page.getByRole("status", { name: "Selected candidate job" })).toContainText("review · cancelled");
  const recordResponse = await page.request.get(receipt.status_route);
  expect(recordResponse.ok()).toBe(true);
  const record = parseCandidateJob(await recordResponse.json(), receipt.job.job_id);
  expect(record.status).toBe("cancelled");
  expect(record.result).toBeNull();
  await page.reload();
  await page.getByRole("tab", { name: "Candidate model" }).click();
  const retained = page.getByRole("list", { name: "Stored candidate jobs" }).getByRole("button", {
    name: new RegExp(`${receipt.job.job_id}$`),
  });
  await expect(retained).toContainText("cancelled");
  await retained.click();
  await expect(page.getByRole("button", { name: "Export review packet", exact: true })).toBeDisabled();
});
