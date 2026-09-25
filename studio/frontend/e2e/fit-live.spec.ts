// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — live parameter fitting, against a real backend

/**
 * A candidate's model is fitted to recordings imported as files, one held out;
 * the result states what the data constrain and replays to the same digest.
 * Then the ratio-only pair is fitted and reported as unconstrained.
 */

import { fileURLToPath } from "node:url";

import { expect, test, type Page } from "@playwright/test";

const fixture = (name: string) => fileURLToPath(new URL(`./fixtures/${name}`, import.meta.url));

test.describe.configure({ mode: "serial" });

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("sc-studio-onboarding-dismissed", "true");
  });
});

/** Import the candidate, open the fit panel and load the cohort, one held out. */
async function prepare(page: Page): Promise<void> {
  await page.goto("./");
  await page.getByRole("button", { name: "Candidate", exact: true }).first().click();
  await page.getByLabel("Import candidate package file").setInputFiles(fixture("LeakyIntegratorToFit.candidate.json"));
  await page.getByRole("button", { name: "Fit", exact: true }).first().click();
  await page.getByLabel("Candidate draft").check();
  await page.getByLabel("Recordings (CSV: current,observed per step)").setInputFiles([
    fixture("step-plus-5.csv"),
    fixture("step-minus-4.csv"),
    fixture("step-plus-8.csv"),
  ]);
  await page.getByLabel("Set of step-plus-8.csv").selectOption("holdout");
}

/**
 * Describe one parameter row of the form.
 *
 * @param page - The page.
 * @param index - The row, from 1.
 * @param domain - Name, bounds and scale.
 */
async function parameter(page: Page, index: number, domain: [string, string, string, boolean]): Promise<void> {
  const [name, low, high, log] = domain;
  await page.getByLabel(`Parameter ${index} name`).fill(name);
  await page.getByLabel(`Parameter ${index} low bound`).fill(low);
  await page.getByLabel(`Parameter ${index} high bound`).fill(high);
  await page.getByLabel(`Parameter ${index} on a log scale`).setChecked(log);
}

test("a candidate's model is fitted, validated on held-out data and replayed", async ({ page }) => {
  await prepare(page);
  await parameter(page, 1, ["v_rest", "-80", "-50", false]);
  await page.getByRole("button", { name: "Add parameter", exact: true }).click();
  await parameter(page, 2, ["tau_m", "1", "50", true]);
  await page.getByLabel("Fixed parameters (name=value, one per line)").fill("R=1\nC=1");
  await page.getByLabel("Generations", { exact: true }).fill("15");
  await page.getByLabel("Population", { exact: true }).fill("8");
  await page.getByRole("button", { name: "Run fit", exact: true }).click();

  const fitted = page.getByRole("table", { name: "Fitted parameters" });
  await expect(fitted).toBeVisible({ timeout: 120_000 });
  const vRest = Number(await fitted.getByRole("row").filter({ hasText: "v_rest" }).locator("td").first().innerText());
  const tau = Number(await fitted.getByRole("row").filter({ hasText: "tau_m" }).locator("td").first().innerText());
  expect(Math.abs(vRest + 65)).toBeLessThan(0.5);
  expect(Math.abs(tau - 10)).toBeLessThan(0.5);
  await expect(fitted).not.toContainText("not stated");
  await expect(page.getByRole("list", { name: "Hold-out error" })).toContainText(/step-plus-8\.csv: RMSE 0\.\d+/);

  await page.getByRole("button", { name: "Replay", exact: true }).click();
  await expect(page.locator("#fit-outcome")).toContainText("Replay reproduced the result", { timeout: 120_000 });
});

test("resistance and capacitance, seen only as a ratio, are reported unconstrained", async ({ page }) => {
  await prepare(page);
  await parameter(page, 1, ["R", "0.1", "10", true]);
  await page.getByRole("button", { name: "Add parameter", exact: true }).click();
  await parameter(page, 2, ["C", "0.1", "10", true]);
  await page.getByLabel("Fixed parameters (name=value, one per line)").fill("v_rest=-65\ntau_m=10");
  await page.getByLabel("Generations", { exact: true }).fill("10");
  await page.getByLabel("Population", { exact: true }).fill("6");
  await page.getByRole("button", { name: "Run fit", exact: true }).click();

  const outcome = page.locator("#fit-outcome");
  await expect(outcome).toContainText("The data do not constrain the combination", { timeout: 120_000 });
  await expect(page.getByRole("table", { name: "Fitted parameters" })).toContainText("not stated");
});
