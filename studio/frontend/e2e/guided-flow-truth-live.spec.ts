// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Live guided-workflow truth: races, invalidation and failure

/**
 * What the guided workflow claims, checked in a real browser against a real
 * Studio server.
 *
 * Each case drives the built bundle against the real backend, so completion
 * comes from real simulation and analysis responses. Where a case needs a
 * response to arrive late, or a request to fail, it intervenes only at the
 * network boundary: the real response is held back and then delivered
 * unchanged, or the connection is refused. No response body is invented.
 */

import { expect, test, type Page, type Route } from "@playwright/test";

/** The two catalogue models the cases switch between. */
const FIRST_MODEL = "PerfectIntegratorNeuron";
const SECOND_MODEL = "QuadraticIFNeuron";

/** The route every catalogue-model simulation is posted to. */
const MODEL_SIMULATE = "/api/models/simulate";

test.describe.configure({ mode: "serial" });

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("sc-studio-onboarding-dismissed", "true");
  });
});

/**
 * The guided workflow's row for one step.
 *
 * @param page - The page.
 * @param step - The step key.
 * @returns Its list item.
 */
function guidedStep(page: Page, step: string) {
  return page.getByLabel("Guided flow", { exact: true }).locator(`li[data-step="${step}"]`);
}

/**
 * Open the Studio and wait for the full live catalogue.
 *
 * @param page - The page to drive.
 */
async function openStudio(page: Page): Promise<void> {
  await page.goto("./");
  await expect(page.getByPlaceholder("Search models...")).toBeVisible();
  await expect(page.getByText(/^\d+\/\d+ models$/)).toBeVisible();
}

/**
 * Search for a catalogue model and select it.
 *
 * @param page - The page to drive.
 * @param modelName - The model's catalogue name.
 */
async function selectModel(page: Page, modelName: string): Promise<void> {
  await page.getByPlaceholder("Search models...").fill(modelName);
  const contract = page.getByTestId(`model-contract-${modelName}`);
  await expect(contract).toBeVisible();
  await contract.locator("..").click();
}

/**
 * Intercept model simulations, handing each one to `handle` with its model.
 *
 * @param page - The page.
 * @param handle - Decides what to do with the route, given the model the
 *   request simulates (`null` when the body names none).
 */
async function interceptModelSimulations(
  page: Page,
  handle: (route: Route, modelName: string | null) => Promise<void>,
): Promise<void> {
  await page.route((url) => url.pathname === MODEL_SIMULATE, async (route) => {
    const body: unknown = route.request().postDataJSON();
    const modelName = typeof body === "object" && body !== null && "model_name" in body
      && typeof body.model_name === "string"
      ? body.model_name
      : null;
    await handle(route, modelName);
  });
}

test("a late simulation of the previous model cannot complete the model now selected", async ({ page }) => {
  let releaseFirst: () => void = () => undefined;
  const firstHeld = new Promise<void>((resolve) => { releaseFirst = resolve; });
  let firstDelivered: () => void = () => undefined;
  const delivered = new Promise<void>((resolve) => { firstDelivered = resolve; });
  let firstArrived: () => void = () => undefined;
  const arrived = new Promise<void>((resolve) => { firstArrived = resolve; });
  let held = false;
  await interceptModelSimulations(page, async (route, modelName) => {
    if (modelName !== FIRST_MODEL || held) {
      await route.continue();
      return;
    }
    held = true;
    // Take the real answer, then hold it until the reader has moved on.
    const response = await route.fetch();
    firstArrived();
    await firstHeld;
    await route.fulfill({ response });
    firstDelivered();
  });

  await openStudio(page);
  await selectModel(page, FIRST_MODEL);
  await arrived;
  await selectModel(page, SECOND_MODEL);
  await expect(page.getByTestId("model-integration-method")).toBeVisible();

  releaseFirst();
  await delivered;
  // The first model's run arrived after the second was selected: it is not
  // evidence of the second model, so simulation must still be to do.
  await expect(page.getByRole("button", { name: "Run next guided step" }))
    .toContainText("Run simulation");
  await expect(guidedStep(page, "simulate")).toHaveAttribute("data-status", "current");

  await page.getByRole("button", { name: "Run next guided step" }).click();
  await expect(guidedStep(page, "simulate")).toHaveAttribute("data-status", "completed");
});

test("editing a parameter withdraws the analysis while the new run completes", async ({ page }) => {
  await openStudio(page);
  await selectModel(page, FIRST_MODEL);
  await expect(guidedStep(page, "simulate")).toHaveAttribute("data-status", "completed");

  const runNext = page.getByRole("button", { name: "Run next guided step" });
  await expect(runNext).toContainText("Run f-I analysis");
  await runNext.click();
  await expect(guidedStep(page, "analyse")).toHaveAttribute("data-status", "completed");

  // Changing the run length changes the experiment: the old analysis is of
  // another experiment, and the automatic re-simulation is of this one.
  await page.getByTestId("slider-T (ms)").fill("300");

  await expect(guidedStep(page, "analyse")).not.toHaveAttribute("data-status", "completed");
  await expect(guidedStep(page, "simulate")).toHaveAttribute("data-status", "completed");
  await expect(guidedStep(page, "analyse")).toHaveAttribute("data-status", "current");
  await expect(runNext).toContainText("Run f-I analysis");
});

test("a failed simulation is shown as failed, offered as a retry, and cleared by a retry that succeeds", async ({ page }) => {
  // Refuse every simulation until the reader retries, so no automatic re-run
  // can succeed behind the failure the case is about.
  let refuse = true;
  await interceptModelSimulations(page, async (route, modelName) => {
    if (refuse && modelName === FIRST_MODEL) {
      await route.abort("connectionrefused");
      return;
    }
    await route.continue();
  });

  await openStudio(page);
  await selectModel(page, FIRST_MODEL);

  const simulate = guidedStep(page, "simulate");
  await expect(simulate).toHaveAttribute("data-status", "failed");
  await expect(simulate).toHaveAttribute("aria-current", "step");
  await expect(simulate).toContainText("failed:");
  const runNext = page.getByRole("button", { name: "Run next guided step" });
  await expect(runNext).toContainText("Retry: Run simulation");

  refuse = false;
  await runNext.click();
  await expect(simulate).toHaveAttribute("data-status", "completed");
  await expect(simulate).not.toContainText("failed:");
  await expect(runNext).toContainText("Run f-I analysis");
});
