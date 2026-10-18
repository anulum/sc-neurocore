// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — live representative-class catalogue-to-silicon browser contract

import { expect, test, type Page, type Response } from "@playwright/test";
import { execFileSync } from "node:child_process";

/** A scientific class and its real catalogue representative. */
interface ClassRepresentative {
  className: string;
  modelName: string;
}

/** The detail fields the browser must render exactly as the live API declares them. */
interface ModelContract {
  integration_method: string;
  validation_metric: string;
}

import type { SimulateResponse, ReplayPack } from "../src/api/client";

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
    const synthesisHeader = page.getByRole("heading", { name: "FPGA synthesis", exact: true }).locator("..");
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
  // The model panel states which silicon operations this installation can run.
  const capabilities = page.getByTestId("model-capabilities");
  await expect(capabilities).toContainText("compile");
  await expect(capabilities).toContainText("formal");

  await page.goto("about:blank");
  await page.goto("./#model=NoSuchNeuron");
  await expect(page.getByText('This link opens "NoSuchNeuron", which this catalogue does not hold.', {
    exact: false,
  })).toBeVisible();
});

/**
 * Match an actual model run to the requested control values.
 *
 * @param response - The live API response.
 * @param model - The chosen model.
 * @param seed - Explicit seed or null for the model default.
 * @param dt - Timestep in milliseconds.
 * @returns Whether this is the requested run, including refused responses.
 */
function modelInputResponse(response: Response, model: string, seed: number | null, dt: number): boolean {
  if (new URL(response.url()).pathname !== "/api/models/simulate") return false;
  const body = response.request().postDataJSON() as Record<string, unknown>;
  return body.model_name === model && (body.seed ?? null) === seed && body.dt === dt;
}

test("Poisson constructor controls run, replay, export and preserve observed state", async ({ page }, testInfo) => {
  await openLiveStudio(page);
  const initial = page.waitForResponse((r) => modelInputResponse(r, "PoissonNeuron", null, 1));
  await selectCatalogueModel(page, "PoissonNeuron");
  const first = await initial;
  expect(first.ok(), await first.text()).toBe(true);
  expect(first.request().postDataJSON()).toMatchObject({ params: { rate_hz: 100 } });
  const firstBody = first.request().postDataJSON() as { params: Record<string, number> };
  expect(firstBody.params).not.toHaveProperty("rng_state");
  expect(firstBody.params).not.toHaveProperty("seed");
  expect(firstBody.params).not.toHaveProperty("dt_ms");
  await expect(page.getByTestId("read-only-state-rng_state")).toBeVisible();
  await expect(page.getByTestId("slider-rng_state₀")).toHaveCount(0);
  await expect(page.getByTestId("slider-seed")).toHaveCount(0);
  await expect(page.getByTestId("slider-dt_ms")).toHaveCount(0);

  await page.getByTestId("slider-I").fill("100");
  await page.getByTestId("slider-dt").fill("0.5");
  const seeded = page.waitForResponse((r) => modelInputResponse(r, "PoissonNeuron", 123, 0.5));
  await page.getByRole("textbox", { name: "seed", exact: true }).fill("123");
  const response = await seeded;
  expect(response.ok(), await response.text()).toBe(true);
  const original = await response.json() as SimulateResponse;
  expect(original.dt).toBe(0.5);
  expect(original.raw?.states).toHaveProperty("rng_state");
  await expect(page.getByRole("button", { name: "Run", exact: true })).toBeEnabled();
  const repeated = page.waitForResponse((r) => modelInputResponse(r, "PoissonNeuron", 123, 0.5));
  await page.getByRole("button", { name: "Run", exact: true }).click();
  const replay = await (await repeated).json() as SimulateResponse;
  expect(replay.experiment).toEqual(original.experiment);
  expect(replay.raw).toEqual(original.raw);
  await page.getByRole("button", { name: "Data table", exact: true }).click();
  await expect(page.getByRole("table", { name: /^Trace data:/ })).toBeVisible();
  const download = page.waitForEvent("download");
  const exported = page.waitForResponse((r) => new URL(r.url()).pathname === "/api/export/replay-pack");
  await page.getByTestId("export-replay-pack").click();
  const packed = await exported;
  expect(packed.ok(), await packed.text()).toBe(true);
  const pack = await packed.json() as ReplayPack;
  expect(pack.request).toMatchObject({ seed: 123, dt: 0.5 });
  const packPath = testInfo.outputPath("poisson-replay.json");
  await (await download).saveAs(packPath);
  const environment = { ...process.env };
  delete environment.PYTHONPATH;
  const output = execFileSync("python", ["-m", "sc_neurocore.studio.replay_pack", packPath, "--json"], {
    cwd: testInfo.outputDir, env: environment, encoding: "utf-8", timeout: 120_000,
  });
  const replayed = JSON.parse(output) as { verdict: string; differences: unknown[] };
  expect(replayed).toMatchObject({ verdict: "match", differences: [] });
  const submitted: unknown = response.request().postDataJSON();
  await testInfo.attach("poisson-experiment", {
    body: JSON.stringify({ request: submitted, experiment: original.experiment, raw: original.raw }),
    contentType: "application/json",
  });

  const fractional = page.waitForResponse((r) => modelInputResponse(r, "PoissonNeuron", 1.5, 0.5));
  await page.getByRole("textbox", { name: "seed", exact: true }).fill("1.5");
  expect((await fractional).status()).toBe(422);
  await expect(page.getByRole("textbox", { name: "seed", exact: true })).toHaveValue("1.5");

  const validSeed = page.waitForResponse((r) => modelInputResponse(r, "PoissonNeuron", 123, 0.5));
  await page.getByRole("textbox", { name: "seed", exact: true }).fill("123");
  expect((await validSeed).ok()).toBe(true);
  const fresh = page.waitForResponse((r) => {
    if (!modelInputResponse(r, "PoissonNeuron", 123, 0.5)) return false;
    const body = r.request().postDataJSON() as Record<string, unknown>;
    return body.trial === "fresh";
  });
  await page.getByRole("combobox", { name: "trial mode", exact: true }).selectOption("fresh");
  const freshResponse = await fresh;
  expect(freshResponse.ok(), await freshResponse.text()).toBe(true);
  const drawn = await freshResponse.json() as SimulateResponse;
  expect(drawn.experiment?.randomness).toMatchObject({ seed_source: "drawn", effective_trial: "fresh" });

  const reset = page.waitForResponse((r) => modelInputResponse(r, "PoissonNeuron", null, 1));
  await page.getByRole("button", { name: "Reset", exact: true }).click();
  expect((await reset).ok()).toBe(true);
  await expect(page.getByRole("textbox", { name: "seed", exact: true })).toHaveValue("");
});

test("legacy parameter seeds migrate through the real seed control", async ({ page, request }) => {
  await openLiveStudio(page);
  await selectCatalogueModel(page, "PoissonNeuron");
  const name = `legacy-seed-${Date.now()}`;
  const saved = await request.post("/api/project/save", {
    data: { name, state: {
      sourceMode: "model", selectedModelName: "PoissonNeuron", modelParams: { seed: 42, rate_hz: 100 },
      dt: 1, duration: 100, current: 100, protocol: "constant",
    } },
  });
  expect(saved.ok(), await saved.text()).toBe(true);
  await page.getByRole("button", { name: "Refresh projects", exact: true }).click();
  const loaded = page.waitForResponse((r) => modelInputResponse(r, "PoissonNeuron", null, 1));
  await page.getByRole("button", { name: `Open project ${name} revision 1`, exact: true }).click();
  expect((await loaded).ok()).toBe(true);
  await expect(page.getByRole("textbox", { name: "seed", exact: true })).toHaveValue("42");
  for (const seed of [123, null]) {
    const changed = page.waitForResponse((r) => modelInputResponse(r, "PoissonNeuron", seed, 1));
    await page.getByRole("textbox", { name: "seed", exact: true }).fill(seed === null ? "" : String(seed));
    const response = await changed;
    expect(response.ok(), await response.text()).toBe(true);
    const body = response.request().postDataJSON() as { params: Record<string, number> };
    expect(body.params).not.toHaveProperty("seed");
  }
});

test("integer and writable initial-state controls preserve constructor types", async ({ page }) => {
  await openLiveStudio(page);
  await selectCatalogueModel(page, "IntegerQIFNeuron");
  const integer = page.getByTestId("slider-v_threshold");
  await expect(integer).toHaveAttribute("step", "1");
  const run = page.waitForResponse((r) => {
    if (!modelInputResponse(r, "IntegerQIFNeuron", null, 1)) return false;
    const body = r.request().postDataJSON() as { params: Record<string, number> };
    return body.params.v_threshold === 240;
  });
  await integer.fill("240");
  expect((await run).ok()).toBe(true);

  await selectCatalogueModel(page, "AdExNeuron");
  const writable = page.getByTestId("slider-v₀");
  await expect(writable).toBeVisible();
  const defaultState = Number(await writable.inputValue());
  const step = Number(await writable.getAttribute("step"));
  const initialState = page.waitForResponse((r) => {
    if (new URL(r.url()).pathname !== "/api/models/simulate") return false;
    const body = r.request().postDataJSON() as { model_name: string; params: Record<string, number> };
    return body.model_name === "AdExNeuron" && body.params.v !== defaultState;
  });
  await writable.fill(String(defaultState + 10 * step));
  const response = await initialState;
  expect(response.ok(), await response.text()).toBe(true);
  const result = await response.json() as SimulateResponse;
  const submitted = response.request().postDataJSON() as { params: Record<string, number> };
  expect(submitted.params.v).not.toBe(defaultState);
  expect(result.initial_state?.v).toBe(submitted.params.v);
});

test("legacy large seed reaches the real progress socket exactly and preserves the model domain refusal", async ({ page, request }) => {
  await openLiveStudio(page);
  await selectCatalogueModel(page, "PoissonNeuron");
  const name = `legacy-large-seed-${Date.now()}`;
  const saved = await request.post("/api/project/save", {
    data: `{"name":"${name}","state":{"sourceMode":"model","selectedModelName":"PoissonNeuron","modelParams":{"seed":9223372036854775807,"rate_hz":100},"dt":1,"duration":20,"current":100,"protocol":"constant"}}`,
    headers: { "Content-Type": "application/json" },
  });
  expect(saved.ok(), await saved.text()).toBe(true);
  await page.getByRole("button", { name: "Refresh projects", exact: true }).click();
  const refused = page.waitForResponse((r) => new URL(r.url()).pathname === "/api/models/simulate"
    && (r.request().postData() ?? "").includes('"seed":9223372036854775807'));
  await page.getByRole("button", { name: `Open project ${name} revision 1`, exact: true }).click();
  expect((await refused).status()).toBe(422);
  await expect(page.getByRole("textbox", { name: "seed", exact: true })).toHaveValue("9223372036854775807");

  const characterize = page.getByRole("banner").getByRole("button", { name: "Characterize", exact: true });
  const socket = page.waitForEvent("websocket", { predicate: (ws) => ws.url().endsWith("/ws/progress") });
  const sent = socket.then((ws) => ws.waitForEvent("framesent"));
  const received = socket.then((ws) => ws.waitForEvent("framereceived", {
    predicate: ({ payload }) => String(payload).includes('"type":"error"') || String(payload).includes('"type": "error"'),
  }));
  const [sentFrame, receivedFrame] = await Promise.all([
    sent, received, characterize.click(),
  ]);
  expect(String(sentFrame.payload)).toContain('"seed":9223372036854775807');
  const failure = JSON.parse(String(receivedFrame.payload)) as { type: string; msg: string };
  expect(failure.type).toBe("error");
  expect(failure.msg).toBe("constructor: model constructor rejected the supplied parameters");
  await expect(page.getByRole("alert").filter({ hasText: failure.msg })).toBeVisible();
  await expect(characterize).toBeEnabled();
});
