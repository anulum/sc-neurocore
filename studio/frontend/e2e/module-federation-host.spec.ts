// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — built Module Federation host browser contract

import { expect, test } from "@playwright/test";

test("the built remote loads its production panel through a real federation host", async ({ page }) => {
  const pageErrors: string[] = [];
  const remoteResponses: { path: string; status: number }[] = [];

  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("response", (response) => {
    const url = new URL(response.url());
    if (url.port === "5184") {
      remoteResponses.push({ path: url.pathname, status: response.status() });
    }
  });
  let librariesRestored = false;
  await page.route("**/api/**", async (route) => {
    const path = new URL(route.request().url()).pathname;
    if (librariesRestored && (path === "/api/models" || path === "/api/presets")) {
      await route.fulfill({ contentType: "application/json", json: [], status: 200 });
      return;
    }
    await route.fulfill({
      contentType: "application/json",
      json: { detail: "The federation contract host has no Studio backend." },
      status: 503,
    });
  });

  await page.goto("/");

  await expect(page.locator("#root")).toHaveAttribute("data-federation-status", "loaded");
  await expect(page.getByRole("heading", { exact: true, name: "SC-NeuroCore Studio" })).toBeVisible();
  await expect(page.getByLabel("Save project", { exact: true })).toBeVisible();
  await expect(page.getByLabel("Save session", { exact: true })).toBeVisible();
  await expect(page.getByText("Hosted view · no backend connected")).toHaveCount(0);
  await page.getByRole("button", { name: "Skip tour", exact: true }).click();
  const modelRetry = page.getByRole("button", { name: "Retry model library", exact: true });
  const presetRetry = page.getByRole("button", { name: "Retry experiment library", exact: true });
  await expect(modelRetry).toBeVisible();
  await expect(presetRetry).toBeVisible();
  librariesRestored = true;
  await modelRetry.click();
  await presetRetry.click();
  await expect(modelRetry).toHaveCount(0);
  await expect(presetRetry).toHaveCount(0);
  expect(pageErrors).toEqual([]);
  expect(remoteResponses).toContainEqual({
    path: "/studios/sc-neurocore/remoteEntry.js",
    status: 200,
  });
  expect(remoteResponses.some(({ path }) => path.includes("SnnStudioPanel"))).toBe(true);
  expect(remoteResponses.every(({ status }) => status < 400)).toBe(true);
});
