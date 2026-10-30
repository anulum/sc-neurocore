// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — live workbench accessibility: names, the plot in words, zoom

/**
 * What the simulation workbench offers a reader who does not see the canvas
 * or uses a keyboard, checked in a real browser against a real backend.
 */

import { readFileSync } from "node:fs";
import { expect, test, type Locator, type Page } from "@playwright/test";
import { parseStudioJson } from "../src/studioJson";

test.describe.configure({ mode: "serial" });

/** Open an already-dismissed tour and wait for its first simulation to arrive. */
async function openWorkbench(page: Page): Promise<void> {
  await page.addInitScript(() => {
    window.localStorage.setItem("sc-studio-onboarding-dismissed", "true");
  });
  const simulated = page.waitForResponse(
    (response) => new URL(response.url()).pathname === "/api/models/simulate" && response.ok(),
  );
  await page.goto("./");
  await simulated;
}

/**
 * Reach a toolbar action through the browser's actual tab order.
 *
 * @param page - The running workbench.
 * @param target - The accessible control the researcher is looking for.
 */
async function tabToControl(page: Page, target: Locator): Promise<void> {
  for (let count = 0; count < 100; count += 1) {
    if (await target.evaluate((element) => element === document.activeElement)) return;
    await page.keyboard.press("Tab");
  }
  await expect(target).toBeFocused();
}

test("the first-run tour keeps keys inside until Escape, then a keyboard export matches the real run", async ({ page }) => {
  const simulated = page.waitForResponse(
    (response) => new URL(response.url()).pathname === "/api/models/simulate" && response.ok(),
  );
  await page.goto("./");
  const response = await simulated;
  const expected: unknown = parseStudioJson(await response.text());
  const tour = page.getByRole("dialog", { name: "Welcome to SC-NeuroCore Studio", exact: true });
  await expect(tour).toBeVisible();
  const skip = tour.getByRole("button", { name: "Skip tour", exact: true });
  await expect(skip).toBeFocused();
  await page.keyboard.press("Shift+Tab");
  await expect(tour.getByRole("button", { name: "Next", exact: true })).toBeFocused();
  await page.keyboard.press("Tab");
  await expect(skip).toBeFocused();
  await page.keyboard.press("3");
  await expect(page.getByRole("tab", { name: "Trace", exact: true })).toHaveAttribute("aria-selected", "true");
  await expect(tour).toBeVisible();
  await page.setViewportSize({ width: 640, height: 400 });
  const bounds = await tour.boundingBox();
  if (bounds === null) throw new Error("The visible tour has no browser geometry");
  expect(bounds.x).toBeGreaterThanOrEqual(0);
  expect(bounds.x + bounds.width).toBeLessThanOrEqual(640);
  await page.keyboard.press("Escape");
  await expect(tour).not.toBeVisible();
  const exportButton = page.getByRole("button", { name: "JSON", exact: true });
  await tabToControl(page, exportButton);
  const downloaded = page.waitForEvent("download");
  await page.keyboard.press("Enter");
  const file = await downloaded;
  expect(await file.failure()).toBeNull();
  const path = await file.path();
  const actual: unknown = parseStudioJson(readFileSync(path, "utf8"));
  if (expected === null || typeof expected !== "object" || Array.isArray(expected)) throw new Error("Simulation returned no result object");
  expect(actual).toMatchObject(expected as Record<string, unknown>);
  await page.reload();
  await expect(page.getByRole("img", { name: /^Trace of / })).toBeVisible();
  await expect(page.getByRole("dialog", { name: "Welcome to SC-NeuroCore Studio", exact: true })).toHaveCount(0);
});

test("every slider is named and says its value", async ({ page }) => {
  await openWorkbench(page);
  const sliders = page.getByRole("slider");
  const count = await sliders.count();
  expect(count).toBeGreaterThan(2);
  for (let index = 0; index < count; index += 1) {
    const slider = sliders.nth(index);
    expect(await slider.getAttribute("aria-label")).toMatch(/\S/);
    expect(await slider.getAttribute("aria-valuetext")).toMatch(/\S/);
  }
  await expect(page.getByRole("slider", { name: "T (ms)", exact: true })).toBeVisible();
});

test("the plot names itself with the run it draws, and offers it as a table", async ({ page }) => {
  await openWorkbench(page);
  const plot = page.getByRole("img", { name: /^Trace of / });
  await expect(plot).toBeVisible();
  await expect(plot).toHaveAttribute("aria-label", /\d+ steps of [\d.]+ ms\): \d+ spikes?\./);

  const toggle = page.getByRole("button", { name: "Data table", exact: true });
  await toggle.focus();
  await page.keyboard.press("Enter");
  await expect(toggle).toHaveAttribute("aria-pressed", "true");
  const table = page.getByRole("table", { name: /^Trace data: / });
  await expect(table).toBeVisible();
  await expect(table.getByRole("rowheader").first()).toBeVisible();
  expect(await table.getByRole("columnheader").allInnerTexts()).toEqual([
    "Variable",
    "Samples shown",
    "Minimum",
    "Maximum",
    "Final",
  ]);
});

test("an analysis view names itself with its own numbers and tables them, not the trace", async ({ page }) => {
  await openWorkbench(page);
  await page.getByRole("tab", { name: "f-I curve", exact: true }).click();
  await page.getByTestId("analysis-job-control-submit").click();
  // Every non-trace view used to be named "... Its values are in the CSV and
  // JSON exports", which hold only the trace.
  const plot = page.getByRole("img", { name: /^f-I curve over I from / });
  await expect(plot).toBeVisible({ timeout: 120_000 });
  await expect(plot).toHaveAttribute("aria-label", /\d+ points: .*Currents are in the model's own units\.$/);

  await page.getByRole("button", { name: "Data table", exact: true }).click();
  const table = page.getByRole("table", { name: "f-I curve: firing rate at each constant current" });
  await expect(table).toBeVisible();
  expect(await table.getByRole("columnheader").allInnerTexts()).toEqual(["Current", "Rate (Hz)"]);
  expect(await table.getByRole("rowheader").count()).toBeGreaterThan(1);
});

test("at twice the zoom the page does not scroll sideways", async ({ page }) => {
  // 200 % zoom on a 1280-pixel window leaves 640 CSS pixels.
  await page.setViewportSize({ width: 640, height: 400 });
  await openWorkbench(page);
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});

test("no model name in the library runs under the badges beside it", async ({ page }) => {
  await openWorkbench(page);
  await expect(page.locator("button.model-row-select").first()).toBeVisible();
  // Long names (ClosedFormContinuous, CompositionalBinding) used to run on
  // under the tier badges; each name now ends in an ellipsis before them.
  const overlapping = await page.evaluate(() => {
    const found: string[] = [];
    for (const button of document.querySelectorAll<HTMLElement>("button.model-row-select")) {
      const name = button.firstElementChild;
      const badges = button.nextElementSibling;
      if (!name || !badges) continue;
      const a = name.getBoundingClientRect();
      const b = badges.getBoundingClientRect();
      if (a.width > 0 && b.width > 0 && a.right > b.left + 0.5) found.push(name.textContent);
    }
    return found;
  });
  expect(overlapping).toEqual([]);
});
