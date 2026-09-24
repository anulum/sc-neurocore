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
  modelName: string;
}

/** The detail fields the browser must render exactly as the live API declares them. */
interface ModelContract {
  integration_method: string;
  validation_metric: string;
}

const CLASS_REPRESENTATIVES: readonly ClassRepresentative[] = [
  { className: "linear IF", modelName: "PerfectIntegratorNeuron" },
  { className: "polynomial IF", modelName: "QuadraticIFNeuron" },
  { className: "conductance / transcendental", modelName: "HodgkinHuxleyNeuron" },
  { className: "relaxation oscillator", modelName: "FitzHughNagumoNeuron" },
  { className: "chaotic", modelName: "HindmarshRoseNeuron" },
  { className: "discrete map", modelName: "RulkovMapNeuron" },
  { className: "stochastic", modelName: "PoissonNeuron" },
  { className: "multi-compartment", modelName: "PinskyRinzelNeuron" },
  { className: "published-discrete", modelName: "BalancedResonateAndFireNeuron" },
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

/**
 * Open the Studio against the live server and record which routes answered.
 *
 * The set is returned rather than asserted here, so a case can say which
 * routes it required rather than requiring the same ones for every case.
 *
 * @param page - The page to drive.
 * @returns The `/api/` paths that answered successfully, filled as they do.
 */
async function openLiveStudio(page: Page): Promise<Set<string>> {
  const completedApiRoutes = new Set<string>();
  page.on("response", (response) => {
    const url = new URL(response.url());
    if (url.pathname.startsWith("/api/") && response.ok()) {
      completedApiRoutes.add(url.pathname);
    }
  });
  // The whole catalogue must load; its size is whatever the live server
  // serves, so the expectation cannot go stale as models are enrolled.
  const catalogue = page.waitForResponse(
    (response) => new URL(response.url()).pathname === "/api/models" && response.ok(),
  );
  await page.goto("./");
  const models: unknown = await (await catalogue).json();
  expect(Array.isArray(models)).toBe(true);
  const total = (models as unknown[]).length;
  expect(total).toBeGreaterThan(0);
  await expect(page.getByText(`${total}/${total} models`, { exact: true })).toBeVisible();
  await expect(page.getByText("capability check failed")).toHaveCount(0);
  return completedApiRoutes;
}

/**
 * Search for a catalogue model and open it, waiting for its contract to show.
 *
 * @param page - The page to drive.
 * @param modelName - The model's catalogue name.
 * @returns The contract fields the live API answered for that model.
 */
async function selectCatalogueModel(page: Page, modelName: string): Promise<ModelContract> {
  await page.getByPlaceholder("Search models...").fill(modelName);
  const contract = page.getByTestId(`model-contract-${modelName}`);
  await expect(contract).toBeVisible();
  const detailResponse = page.waitForResponse((response) => {
    const url = new URL(response.url());
    return url.pathname === `/api/models/${modelName}` && response.ok();
  });
  await contract.locator("..").click();
  const detail = (await (await detailResponse).json()) as ModelContract;
  expect(typeof detail.integration_method).toBe("string");
  expect(typeof detail.validation_metric).toBe("string");
  await expect(page.getByTestId("model-integration-method")).toBeVisible();
  return detail;
}

test("the live catalogue surfaces one honest representative of every scientific class", async ({ page }) => {
  const completedApiRoutes = await openLiveStudio(page);

  for (const representative of CLASS_REPRESENTATIVES) {
    await test.step(representative.className, async () => {
      // The browser must show the served contract, not a copy that ages.
      const served = await selectCatalogueModel(page, representative.modelName);
      await expect(page.getByTestId("model-integration-method")).toContainText(
        served.integration_method,
      );
      await expect(page.getByTestId("model-validation-metric")).toContainText(
        served.validation_metric,
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

test("proven-readiness filters and a model link reach the live catalogue", async ({ page }) => {
  await openLiveStudio(page);

  // The floor is the server's decision, on the verified tiers only.
  const proven = page.waitForResponse((response) => {
    const url = new URL(response.url());
    return url.pathname === "/api/models/query"
      && url.searchParams.get("min_verified_science") === "5"
      && response.ok();
  });
  await page.getByRole("button", { name: "S5 proven", exact: true }).click();
  const answer = (await (await proven).json()) as { matched: number; total: number; models: string[] };
  expect(answer.matched).toBeLessThan(answer.total);
  await expect(page.getByRole("button", { name: "S5 proven", exact: true }))
    .toHaveAttribute("aria-pressed", "true");
  await expect(page.getByText(`${answer.matched}/${answer.total} models`, { exact: true })).toBeVisible();
  for (const name of answer.models) {
    await expect(page.getByTestId(`model-contract-${name}`)).toBeVisible();
  }

  // A model link opens that model after a fresh load, and names one it cannot open.
  const linked = "PerfectIntegratorNeuron";
  const detail = page.waitForResponse(
    (response) => new URL(response.url()).pathname === `/api/models/${linked}` && response.ok(),
  );
  await page.goto("about:blank");
  await page.goto(`./#model=${linked}`);
  await detail;
  await expect(page.getByTestId("model-integration-method")).toBeVisible();

  await page.goto("about:blank");
  await page.goto("./#model=NoSuchNeuron");
  await expect(page.getByText('This link opens "NoSuchNeuron", which this catalogue does not hold.', {
    exact: false,
  })).toBeVisible();
});
