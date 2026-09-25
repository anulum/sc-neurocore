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

import { expect, test, type Page } from "@playwright/test";

test.describe.configure({ mode: "serial" });

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("sc-studio-onboarding-dismissed", "true");
  });
});

/** Open the Studio and wait for its first simulation to arrive. */
async function openWorkbench(page: Page): Promise<void> {
  const simulated = page.waitForResponse(
    (response) => new URL(response.url()).pathname === "/api/models/simulate" && response.ok(),
  );
  await page.goto("./");
  await simulated;
}

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

test("at twice the zoom the page does not scroll sideways", async ({ page }) => {
  // 200 % zoom on a 1280-pixel window leaves 640 CSS pixels.
  await page.setViewportSize({ width: 640, height: 400 });
  await openWorkbench(page);
  const overflow = await page.evaluate(
    () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
  );
  expect(overflow).toBeLessThanOrEqual(0);
});
