// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — live review comments on a saved revision, against a real backend

/**
 * Comments are made on the saved revision that is open, replies thread under
 * what they answer, and both come back when the project is opened again.
 */

import { expect, test } from "@playwright/test";

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("sc-studio-onboarding-dismissed", "true");
  });
});

test("a saved revision is reviewed, replied to, and the thread survives reopening", async ({ page }) => {
  await page.goto("./");
  await page.getByRole("button", { name: "Review", exact: true }).first().click();
  await expect(page.getByRole("region", { name: "Review" })).toContainText("Save or open a project");

  const name = `review-${Date.now()}`;
  page.once("dialog", (dialog) => { void dialog.accept(name); });
  const saved = page.waitForResponse(
    (response) => new URL(response.url()).pathname === "/api/project/save" && response.ok(),
  );
  await page.getByLabel("Save project", { exact: true }).click();
  await saved;

  const review = page.getByRole("region", { name: "Review" });
  await expect(review.getByRole("heading")).toHaveText(`Review of ${name}, revision 1`);
  await page.getByLabel("Comment on this revision", { exact: true }).fill("Why this timestep?");
  await page.getByRole("button", { name: "Add comment", exact: true }).click();
  const comments = review.getByRole("list", { name: "Comments" });
  await expect(comments).toContainText("Why this timestep?");
  await expect(comments).toContainText("local");

  await page.getByRole("button", { name: /^Reply to local: Why this timestep\?/ }).click();
  await page.getByLabel("Reply to local", { exact: true }).fill("It resolves the spike upstroke.");
  await page.getByRole("button", { name: "Add reply", exact: true }).click();
  // The reply is nested under the comment it answers.
  await expect(comments.locator("li ul li")).toContainText("It resolves the spike upstroke.");

  await page.reload();
  await page.getByRole("button", { name: "Refresh projects", exact: true }).click();
  await page.getByRole("button", { name: new RegExp(`^Open project ${name}`) }).click();
  await page.getByRole("button", { name: "Review", exact: true }).first().click();
  await expect(page.getByRole("list", { name: "Comments" })).toContainText("It resolves the spike upstroke.");
});
