// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Preregistered training criterion against the live Studio

import { expect, test, type Page } from "@playwright/test";

test.describe.configure({ mode: "serial" });

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("sc-studio-onboarding-dismissed", "true");
  });
});

/**
 * Declare a criterion in the form, train on the live server and wait for the job to finish.
 *
 * @param page - The browser page.
 * @param metric - The criterion's metric option value.
 * @param threshold - The criterion's bound as typed.
 * @returns The job the server started.
 */
async function trainWithCriterion(page: Page, metric: "val_accuracy" | "val_loss", threshold: string): Promise<string> {
  await page.goto("./");
  await page.getByRole("tab", { name: "Training" }).click();
  await page.getByLabel("Epochs", { exact: true }).fill("1");
  await page.getByLabel("Batch Size", { exact: true }).fill("32");
  await page.getByLabel("Timesteps", { exact: true }).fill("4");
  await page.getByLabel("Judge this run against a criterion stored before it starts").check();
  await page.getByLabel("Criterion metric", { exact: true }).selectOption(metric);
  await page.getByLabel("Criterion threshold", { exact: true }).fill(threshold);
  await page.getByLabel("Criterion rationale", { exact: true }).fill(`declared before the run: ${metric} ${threshold}`);
  const train = page.getByRole("button", { name: "Train", exact: true }).last();
  await expect(train).toBeEnabled();
  const started = page.waitForRequest((request) => new URL(request.url()).pathname === "/api/training/start");
  await train.click();
  const body = (await started).postDataJSON() as { preregistration?: unknown };
  expect(body.preregistration).toEqual({
    metric, threshold: Number(threshold), rationale: `declared before the run: ${metric} ${threshold}`,
  });
  const response = await (await started).response();
  const start = await response?.json() as { job_id: string };
  await expect.poll(async () => {
    const status = await page.request.get(`/api/training/status/${start.job_id}`);
    return (await status.json() as { status: string }).status;
  }, { timeout: 120_000 }).toBe("completed");
  return start.job_id;
}

test("a met criterion is reported as met after the live run", async ({ page }) => {
  const jobId = await trainWithCriterion(page, "val_accuracy", "0");
  const verdict = page.getByRole("status", { name: "Preregistered verdict" });
  await expect(verdict).toContainText("Criterion met: validation accuracy ≥ 0, observed");
  await expect(verdict).toContainText("Criterion stored before the run as");
  // The run list is read once the job record is sealed; read on the terminal
  // event it listed the finished run as still running until Refresh.
  await expect(page.getByLabel("Retained run", { exact: true }).locator(`option[value="${jobId}"]`))
    .toHaveText(new RegExp(`^${jobId} · completed`));
});

test("a missed criterion is reported as missed and survives selecting the retained run", async ({ page }) => {
  const jobId = await trainWithCriterion(page, "val_loss", "0");
  const verdict = page.getByRole("status", { name: "Preregistered verdict" });
  await expect(verdict).toContainText("Criterion missed: validation loss ≤ 0, observed");
  await page.reload();
  await page.getByRole("tab", { name: "Training" }).click();
  await page.getByRole("button", { name: "Refresh runs", exact: true }).click();
  await page.getByLabel("Retained run", { exact: true }).selectOption(jobId);
  await expect(page.getByRole("status", { name: "Preregistered verdict" }))
    .toContainText("Criterion missed: validation loss ≤ 0, observed");
});
