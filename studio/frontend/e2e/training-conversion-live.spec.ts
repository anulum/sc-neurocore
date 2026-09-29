// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — QCFS conversion training route against the live Studio

import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("sc-studio-onboarding-dismissed", "true");
  });
});

test("a conversion run is judged on its converted network and says so", async ({ page }) => {
  await page.goto("./");
  await page.getByRole("tab", { name: "Training" }).click();
  await page.getByLabel("Model", { exact: true }).selectOption("qcfs_conversion");
  await expect(page.getByLabel("Surrogate", { exact: true })).toHaveCount(0);
  await page.getByLabel("Epochs", { exact: true }).fill("1");
  await page.getByLabel("Batch Size", { exact: true }).fill("32");
  await page.getByLabel("Timesteps", { exact: true }).fill("4");
  await page.getByLabel("Target", { exact: true }).selectOption("ecp5");
  await page.getByLabel("Judge this run against a criterion stored before it starts").check();
  await page.getByLabel("Criterion metric", { exact: true }).selectOption("conversion_accuracy_drop");
  await page.getByLabel("Criterion threshold", { exact: true }).fill("1");
  const train = page.getByRole("button", { name: "Train", exact: true }).last();
  await expect(train).toBeEnabled();
  const started = page.waitForRequest((request) => new URL(request.url()).pathname === "/api/training/start");
  await train.click();
  const body = (await started).postDataJSON() as Record<string, unknown>;
  expect(body.model_kind).toBe("qcfs_conversion");
  expect(body.target_profile).toBe("ecp5");
  expect(body).not.toHaveProperty("surrogate");
  expect(body).not.toHaveProperty("learn_beta");
  expect(body).not.toHaveProperty("learn_threshold");
  const start = await (await (await started).response())?.json() as { job_id: string };
  await expect.poll(async () => {
    const status = await page.request.get(`/api/training/status/${start.job_id}`);
    return (await status.json() as { status: string }).status;
  }, { timeout: 120_000 }).toBe("completed");
  await expect(page.getByRole("status", { name: "Conversion result" }))
    .toContainText("With coefficients rounded for ecp5:");
  await expect(page.getByRole("status", { name: "Preregistered verdict" }))
    .toContainText("Criterion met: conversion accuracy drop ≤ 1, observed");
  const report = await page.request.get(
    `/api/studio/jobs/${start.job_id}/artifacts/training/conversion_report.json`,
  );
  expect(report.ok()).toBe(true);
  const sealed = await report.json() as { schema_version: string; timesteps: number };
  expect(sealed.schema_version).toBe("sc-neurocore.conversion-loss-report.v1");
  expect(sealed.timesteps).toBe(4);
  const target = await page.request.get(
    `/api/studio/jobs/${start.job_id}/artifacts/training/target_report.json`,
  );
  expect(target.ok()).toBe(true);
  const calibration = await target.json() as { profile: { name: string }; schema_version: string };
  expect(calibration.schema_version).toBe("sc-neurocore.conversion-target-report.v1");
  expect(calibration.profile.name).toBe("ecp5");
});
