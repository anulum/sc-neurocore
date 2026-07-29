// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — live representative-class catalogue-to-silicon browser contract

import { expect, test, type Page } from "@playwright/test";

interface ClassRepresentative {
  className: string;
  integrator: string;
  modelName: string;
  validation: string;
}

const CLASS_REPRESENTATIVES: readonly ClassRepresentative[] = [
  { className: "linear IF", integrator: "euler", modelName: "PerfectIntegratorNeuron", validation: "parity" },
  { className: "polynomial IF", integrator: "euler", modelName: "QuadraticIFNeuron", validation: "parity" },
  { className: "conductance / transcendental", integrator: "rk4", modelName: "HodgkinHuxleyNeuron", validation: "parity" },
  { className: "relaxation oscillator", integrator: "rk4", modelName: "FitzHughNagumoNeuron", validation: "parity" },
  { className: "chaotic", integrator: "rk4", modelName: "HindmarshRoseNeuron", validation: "parity" },
  { className: "discrete map", integrator: "map", modelName: "RulkovMapNeuron", validation: "trajectory" },
  { className: "stochastic", integrator: "poisson_interval", modelName: "PoissonNeuron", validation: "statistical" },
  { className: "multi-compartment", integrator: "euler", modelName: "PinskyRinzelNeuron", validation: "none" },
  { className: "published-discrete", integrator: "euler", modelName: "BalancedResonateAndFireNeuron", validation: "none" },
] as const;

const TERMINAL_REPRESENTATIVES = [
  { className: "linear IF", modelName: "PerfectIntegratorNeuron" },
  { className: "polynomial IF", modelName: "QuadraticIFNeuron" },
] as const;

test.describe.configure({ mode: "serial" });

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("sc-studio-onboarding-dismissed", "true");
  });
});

async function openLiveStudio(page: Page): Promise<Set<string>> {
  const completedApiRoutes = new Set<string>();
  page.on("response", (response) => {
    const url = new URL(response.url());
    if (url.pathname.startsWith("/api/") && response.ok()) {
      completedApiRoutes.add(url.pathname);
    }
  });
  await page.goto("./");
  await expect(page.getByText("160/160 models", { exact: true })).toBeVisible();
  await expect(page.getByText("capability check failed")).toHaveCount(0);
  return completedApiRoutes;
}

async function selectCatalogueModel(page: Page, modelName: string): Promise<void> {
  await page.getByPlaceholder("Search models...").fill(modelName);
  const contract = page.getByTestId(`model-contract-${modelName}`);
  await expect(contract).toBeVisible();
  const detailResponse = page.waitForResponse((response) => {
    const url = new URL(response.url());
    return url.pathname === `/api/models/${modelName}` && response.ok();
  });
  await contract.locator("..").click();
  await detailResponse;
  await expect(page.getByTestId("model-integration-method")).toBeVisible();
}

test("the live catalogue surfaces one honest representative of every scientific class", async ({ page }) => {
  const completedApiRoutes = await openLiveStudio(page);

  for (const representative of CLASS_REPRESENTATIVES) {
    await test.step(representative.className, async () => {
      await selectCatalogueModel(page, representative.modelName);
      await expect(page.getByTestId("model-integration-method")).toContainText(
        representative.integrator,
      );
      await expect(page.getByTestId("model-validation-metric")).toContainText(
        representative.validation,
      );
    });
  }

  expect(completedApiRoutes).toContain("/api/models");
  expect(completedApiRoutes).toContain("/api/models/facets");
  for (const { modelName } of CLASS_REPRESENTATIVES) {
    expect(completedApiRoutes).toContain(`/api/models/${modelName}`);
  }
});

for (const representative of TERMINAL_REPRESENTATIVES) {
  test(`${representative.className} reaches routed terminal evidence through live browser endpoints`, async ({ page }) => {
    const completedApiRoutes = await openLiveStudio(page);
    await selectCatalogueModel(page, representative.modelName);

    const runNext = page.getByRole("button", { name: "Run next guided step" });
    await expect(runNext).toContainText("Run f-I analysis");
    await runNext.click();
    await expect(runNext).toContainText("Skip training");
    await runNext.click();
    await expect(runNext).toContainText("Compile RTL");
    await runNext.click();
    await expect(runNext).toContainText("Run RTL co-sim");
    await runNext.click();
    await expect(runNext).toContainText("Run synthesis");

    const hardwareAction = page.locator('[data-card="compile"]').getByRole("button");
    await expect(hardwareAction).toHaveText("Open synthesis");
    await hardwareAction.click();
    const synthesisHeader = page.getByText("FPGA Synthesis").locator("..");
    await synthesisHeader.getByRole("combobox").selectOption("ecp5");
    await synthesisHeader.getByRole("button", { name: "Synthesise + Route" }).click();
    await expect(runNext).toContainText("Export evidence");

    const evidenceCard = page.locator('[data-card="export"]');
    const evidenceAction = evidenceCard.getByRole("button");
    await expect(evidenceAction).toHaveText("Export synthesis bundle");
    await evidenceAction.click();
    await expect(runNext).toContainText("Workflow complete");
    await evidenceAction.click();

    const terminalSummary = page.getByText("Selected RTL synthesis/PnR terminal").locator("..");
    await expect(terminalSummary).toBeVisible();
    await expect(terminalSummary).toContainText(`Model: ${representative.modelName} /`);
    await expect(terminalSummary.getByText("Status: completed", { exact: true })).toBeVisible();
    await expect(terminalSummary.getByText(/^Netlist: [0-9a-f]{12}$/)).toBeVisible();
    await expect(terminalSummary.getByText(/^Routed design: [0-9a-f]{12}$/)).toBeVisible();

    for (const route of [
      "/api/models/simulate",
      "/api/analysis/jobs",
      "/api/models/compile",
      "/api/models/cosim",
      "/api/synth/terminal",
      "/api/studio/evidence/bundle",
    ]) {
      expect(completedApiRoutes).toContain(route);
    }
  });
}
