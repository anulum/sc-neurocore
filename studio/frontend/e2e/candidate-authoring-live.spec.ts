// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — live candidate-model authoring, against a real backend

/**
 * A reader who did not write the candidate imports it, edits it, validates,
 * diffs, simulates and reviews it, exports it and its review packet, and finds
 * it again in the saved workspace -- all against the real Studio backend.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import { expect, test, type Download, type Page } from "@playwright/test";

const FIXTURE = fileURLToPath(new URL("./fixtures/SlowAdaptationAdEx.candidate.json", import.meta.url));

test.describe.configure({ mode: "serial" });

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("sc-studio-onboarding-dismissed", "true");
  });
});

/** Open the Studio on the candidate panel. */
async function openCandidate(page: Page): Promise<void> {
  await page.goto("./");
  await page.getByRole("button", { name: "Candidate", exact: true }).first().click();
  await expect(page.getByRole("region", { name: "Candidate model" })).toBeVisible();
}

/** Read a download's text. */
async function downloadedText(download: Download): Promise<string> {
  const path = await download.path();
  return readFileSync(path, "utf8");
}

test("a candidate is imported, edited, checked, reviewed, exported and kept with the workspace", async ({
  page,
}) => {
  await openCandidate(page);
  const draft = page.getByLabel("Candidate package (JSON)", { exact: true });
  const outcome = page.locator("#candidate-outcome");

  await page.getByLabel("Import candidate package file").setInputFiles(FIXTURE);
  await expect(draft).toHaveValue(readFileSync(FIXTURE, "utf8"));

  await page.getByRole("button", { name: "Validate", exact: true }).click();
  await expect(outcome).toContainText("Valid. Digest");

  // Edit the adaptation time constant the candidate proposes.
  const edited = (await draft.inputValue()).replace('"tau_w": 300.0', '"tau_w": 450.0');
  await draft.fill(edited);
  await page.getByRole("button", { name: "Diff against parent", exact: true }).click();
  const diff = page.getByRole("table", { name: "What the candidate changes against AdExNeuron" });
  await expect(diff).toBeVisible();
  await expect(diff.getByRole("row").filter({ hasText: "tau_w" })).toContainText("100 → 450");

  await page.getByLabel("Current", { exact: true }).fill("800");
  await page.getByLabel("Steps", { exact: true }).fill("2000");
  await page.getByRole("button", { name: "Simulate", exact: true }).click();
  await expect(outcome).toContainText(/\d+ spikes in 2000 steps \(euler, dt 0\.1 ms at 800 pA\)/);

  await page.getByRole("button", { name: "Run reference tests", exact: true }).click();
  await expect(outcome).toContainText("Reference tests passed.");
  await expect(outcome).toContainText("catalogue promotion");

  const packetDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export review packet", exact: true }).click();
  const packetFile = await packetDownload;
  expect(packetFile.suggestedFilename()).toBe("SlowAdaptationAdEx.review.json");
  const packet = JSON.parse(await downloadedText(packetFile)) as {
    candidate: unknown;
    reference_tests_passed: boolean;
  };
  expect(packet.reference_tests_passed).toBe(true);
  expect(packet.candidate).toEqual(JSON.parse(edited));

  // The exported candidate is the draft itself, byte for byte.
  const candidateDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Export candidate", exact: true }).click();
  const candidateFile = await candidateDownload;
  expect(candidateFile.suggestedFilename()).toBe("SlowAdaptationAdEx.candidate.json");
  expect(await downloadedText(candidateFile)).toBe(edited);

  // A fault is reported at its field.
  const broken = JSON.parse(edited) as { units: { state: Record<string, string> } };
  broken.units.state.v = "banana";
  await draft.fill(JSON.stringify(broken, null, 2));
  await page.getByRole("button", { name: "Validate", exact: true }).click();
  await expect(page.getByRole("list", { name: "Candidate problems" })).toContainText(
    "/units/state/v 'banana' is not a unit",
  );
  await draft.fill(edited);

  // The draft is the workspace's: it is saved with it and comes back with it.
  const name = `candidate-${Date.now()}`;
  page.once("dialog", (dialog) => { void dialog.accept(name); });
  const saved = page.waitForResponse(
    (response) => new URL(response.url()).pathname === "/api/project/save" && response.ok(),
  );
  await page.getByLabel("Save project", { exact: true }).click();
  await saved;

  await page.reload();
  await openCandidate(page);
  await expect(page.getByLabel("Candidate package (JSON)", { exact: true })).toHaveValue("");
  await page.getByRole("button", { name: "Refresh projects", exact: true }).click();
  const loaded = page.waitForResponse(
    (response) => new URL(response.url()).pathname === `/api/project/load/${name}` && response.ok(),
  );
  await page.getByRole("button", { name: new RegExp(`^Open project ${name}`) }).click();
  await loaded;
  await expect(page.getByLabel("Candidate package (JSON)", { exact: true })).toHaveValue(edited);
});
