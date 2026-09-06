// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { expect, test } from "@playwright/test";

import {
  analysisUnavailable,
  capabilityRegistryContract,
  defaultApiMocks,
  installApiDispatcher,
  registry,
  simulationCapability,
  synthesisUnavailable,
} from "./adminOperatorHarness";

test.setTimeout(60_000);


test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("sc-studio-onboarding-dismissed", "true");
    window.sessionStorage.setItem("sc-neurocore-studio-auth-token", "browser-token");
  });
  await installApiDispatcher(page, defaultApiMocks());
});

test("capability menu exposes unavailable requirements", async ({ page }) => {
  const mocks = defaultApiMocks();
  mocks.set(
    "/api/studio/capabilities",
    registry([
      capabilityRegistryContract,
      simulationCapability,
      analysisUnavailable,
      synthesisUnavailable,
    ]),
  );
  await installApiDispatcher(page, mocks);

  await page.goto("/");
  await page.getByText("2/4 ready").click();

  const capabilityMenu = page.locator(".capability-menu");
  await expect(capabilityMenu.getByText("Analysis Suite")).toBeVisible();
  await expect(capabilityMenu.getByText("analysis: analysis endpoint disabled")).toBeVisible();
  await expect(capabilityMenu.getByText("Synthesis Dashboard")).toBeVisible();
  await expect(capabilityMenu.getByText("yosys: yosys unavailable")).toBeVisible();
});

test("unavailable panel contracts disable toolbar and keyboard activation", async ({ page }) => {
  const mocks = defaultApiMocks();
  mocks.set(
    "/api/studio/capabilities",
    registry([
      capabilityRegistryContract,
      simulationCapability,
      analysisUnavailable,
    ]),
  );
  await installApiDispatcher(page, mocks);

  await page.goto("/");

  await expect(page.getByRole("button", { name: "f-I" }).first()).toBeDisabled();
  await expect(page.getByRole("button", { name: "f-I" }).last()).toBeDisabled();

  await page.keyboard.press("3");

  await expect(page.getByText("Analysis endpoints are unavailable.")).toHaveCount(0);
  await expect(page.locator("canvas")).toBeVisible();
});

test("missing active panel capability fails closed at startup", async ({ page }) => {
  const mocks = defaultApiMocks();
  mocks.set("/api/studio/capabilities", registry([capabilityRegistryContract]));
  const api = await installApiDispatcher(page, mocks);

  await page.goto("/");

  await expect(page.locator(".capability-blocked-title", { hasText: "Trace" })).toBeVisible();
  await expect(page.getByText("Backend capability contract is missing from the registry.")).toBeVisible();
  await page.keyboard.press("Space");
  await page.waitForTimeout(100);
  expect(api.requests("/api/simulate")).toBe(0);
});
