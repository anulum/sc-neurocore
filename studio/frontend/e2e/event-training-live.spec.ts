// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — live event training input and workspace acceptance

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "@playwright/test";
import type { ResolvedTrainingConfig } from "../src/api/types";

test("event input reaches real training, checkpoint export and workspace reopening", async ({ page }) => {
  const root = process.env.SC_NEUROCORE_STUDIO_EVENT_TEST_ROOT;
  if (root === undefined) throw new Error("SC_NEUROCORE_STUDIO_EVENT_TEST_ROOT is required");
  const declaration = readFileSync(join(root, "event_data.json"), "utf-8");
  await page.addInitScript(() => {
    window.localStorage.setItem("sc-studio-onboarding-dismissed", "true");
  });
  await page.goto("./");
  await page.getByRole("button", { name: "Train", exact: true }).first().click();
  await page.getByLabel("Dataset", { exact: true }).selectOption("nmnist");
  const train = page.getByRole("button", { name: "Train", exact: true }).last();
  await expect(train).toBeDisabled();
  await page.getByLabel("Event input JSON", { exact: true }).fill("{bad");
  await page.getByRole("button", { name: "Apply event input", exact: true }).click();
  await expect(page.getByRole("alert")).toBeVisible();
  await expect(train).toBeDisabled();
  await page.getByLabel("Event input file", { exact: true }).setInputFiles(join(root, "event_data.json"));
  await expect(page.getByLabel("Event input JSON", { exact: true })).toHaveValue(declaration);
  await page.getByRole("button", { name: "Apply event input", exact: true }).click();
  await expect(page.getByLabel("Timesteps", { exact: true })).toHaveValue("4");
  await page.getByLabel("Epochs", { exact: true }).fill("1");
  await page.getByLabel("Batch Size", { exact: true }).fill("3");
  await page.getByLabel("Seed", { exact: true }).fill("7");
  await page.getByLabel("Gradient norm limit", { exact: true }).fill("0");
  await expect(train).toBeEnabled();
  const started = page.waitForResponse((response) => new URL(response.url()).pathname === "/api/training/start");
  await train.click();
  const response = await started;
  expect(response.ok()).toBe(true);
  const start = await response.json() as { job_id: string };
  await expect.poll(async () => {
    const status = await page.request.get(`/api/training/status/${start.job_id}`);
    const value = await status.json() as { status: string };
    return value.status;
  }, { timeout: 120_000 }).toBe("completed");
  const checkpoint = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export checkpoint", exact: true }).click();
  const checkpointPath = await (await checkpoint).path();
  const pack = JSON.parse(readFileSync(checkpointPath, "utf-8")) as { config: ResolvedTrainingConfig };
  expect(pack.config.event_data).toEqual(JSON.parse(declaration));
  expect(pack.config.seed).toBe(7);
  expect(pack.config.max_grad_norm).toBe(0);
  const warmStart = page.getByRole("button", { name: "Attach (warm-start)", exact: true });
  await expect(warmStart).toBeEnabled();
  await page.getByLabel("Event input JSON", { exact: true }).fill("{bad");
  await expect(warmStart).toBeDisabled();
  await page.getByLabel("Event input JSON", { exact: true }).fill(declaration);
  await page.getByRole("button", { name: "Apply event input", exact: true }).click();
  await expect(warmStart).toBeEnabled();
  const name = `event-training-${Date.now()}`;
  page.once("dialog", (dialog) => { void dialog.accept(name); });
  const saved = page.waitForResponse((r) => new URL(r.url()).pathname === "/api/project/save");
  await page.getByLabel("Save project", { exact: true }).click();
  expect((await saved).ok()).toBe(true);
  await page.reload();
  await page.getByRole("button", { name: "Refresh projects", exact: true }).click();
  await page.getByRole("button", { name: new RegExp(`^Open project ${name}`) }).click();
  await page.getByRole("button", { name: "Train", exact: true }).first().click();
  await expect(page.getByLabel("Dataset", { exact: true })).toHaveValue("nmnist");
  await expect(page.getByLabel("Seed", { exact: true })).toHaveValue("7");
  await expect(page.getByLabel("Gradient norm limit", { exact: true })).toHaveValue("0");
  await expect.poll(async () => {
    const value = await page.getByLabel("Event input JSON", { exact: true }).inputValue();
    return JSON.parse(value) as unknown;
  }).toEqual(pack.config.event_data);
  await page.getByLabel("Epochs", { exact: true }).fill("2");
  const resume = page.getByRole("button", { name: "Resume from checkpoint", exact: true });
  await expect(resume).toBeEnabled();
  const changed = JSON.parse(declaration) as {
    encoder: Record<string, unknown>; digests: { encoder: string };
  };
  changed.encoder.dt_ms = 2.004;
  const canonicalEncoder = Object.fromEntries(Object.keys(changed.encoder).sort().map((key) => [key, changed.encoder[key]]));
  changed.digests.encoder = "sha256:" + createHash("sha256").update(JSON.stringify(canonicalEncoder)).digest("hex");
  await page.getByLabel("Event input JSON", { exact: true }).fill(JSON.stringify(changed));
  await page.getByRole("button", { name: "Apply event input", exact: true }).click();
  const beforeRefusal = await (await page.request.get("/api/training/jobs")).json() as unknown;
  const refused = page.waitForResponse((r) => new URL(r.url()).pathname === "/api/studio/training/weight-restore/attach");
  await resume.click();
  const refusal = await refused;
  expect(refusal.status()).toBe(422);
  expect(await refusal.text()).toContain("exact resume requires the source manifest, split and encoder unchanged");
  expect(await (await page.request.get("/api/training/jobs")).json()).toEqual(beforeRefusal);
  await expect(resume).toBeEnabled();
  await page.getByLabel("Event input JSON", { exact: true }).fill(declaration);
  await page.getByRole("button", { name: "Apply event input", exact: true }).click();
  const resumed = page.waitForResponse((r) => new URL(r.url()).pathname === "/api/studio/training/weight-restore/attach");
  await resume.click();
  const resumeResponse = await resumed;
  expect(resumeResponse.ok()).toBe(true);
  const attachment = await resumeResponse.json() as { job_id: string; mode: string; source_job_id: string };
  expect(attachment.mode).toBe("exact_resume");
  expect(attachment.source_job_id).toBe(start.job_id);
  await expect(page.getByText(attachment.job_id, { exact: true }).first()).toBeVisible();
  await expect.poll(async () => {
    const status = await page.request.get(`/api/training/status/${attachment.job_id}`);
    return (await status.json() as { status: string }).status;
  }, { timeout: 120_000 }).toBe("completed");
  const continued = await page.request.get(`/api/training/checkpoint/${attachment.job_id}`);
  const continuation = await continued.json() as { config: ResolvedTrainingConfig };
  expect(continuation.config.event_data).toEqual(pack.config.event_data);
  expect(continuation.config.epochs).toBe(2);
  await expect(page.getByText("exact_resume", { exact: true })).toBeVisible();
  const uninterruptedResponse = await page.request.post("/api/training/start", {
    data: { ...pack.config, epochs: 2 },
  });
  expect(uninterruptedResponse.ok()).toBe(true);
  const uninterrupted = await uninterruptedResponse.json() as { job_id: string };
  await expect.poll(async () => {
    const status = await page.request.get(`/api/training/status/${uninterrupted.job_id}`);
    return (await status.json() as { status: string }).status;
  }, { timeout: 120_000 }).toBe("completed");
  const files: string[] = [];
  for (const [label, job] of [["resumed", attachment.job_id], ["uninterrupted", uninterrupted.job_id]]) {
    const artifact = await page.request.get(`/api/studio/jobs/${job}/artifacts/training/model_state.pt`);
    expect(artifact.ok()).toBe(true);
    const body = await artifact.body();
    expect(createHash("sha256").update(body).digest("hex")).toBe(artifact.headers()["x-studio-artifact-sha256"]);
    const path = join(root, `${label}.pt`);
    writeFileSync(path, body);
    files.push(path);
  }
  const compared = execFileSync("python", ["e2e/event_training_state.py", ...files], { encoding: "utf-8", timeout: 60_000 });
  expect(compared).toContain("Complete saved training states match");

});
