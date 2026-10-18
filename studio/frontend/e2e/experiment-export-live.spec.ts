// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — live browser-to-server-to-replay export contract

import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";

import { expect, test, type Page, type Response } from "@playwright/test";

import type { CodegenResponse, ReplayPack, SimulateResponse } from "../src/api/types";

/** The installed acceptance replays with the same interpreter that serves the wheel. */
const REPLAY_PYTHON = process.env.SC_NEUROCORE_STUDIO_INSTALLED_PYTHON ?? "python";

/**
 * What `python -m sc_neurocore.studio.replay_pack --json` prints.
 *
 * The runner's contract, not the browser's, so it is stated here rather than
 * in `src/api/types.ts`: nothing the Studio ships reads this document.
 */
interface ReplayOutcome {
  /** `"match"` when the replay reproduced the sealed expectation. */
  verdict: string;
  /** Every field that did not reproduce, empty on a match. */
  differences: readonly unknown[];
  /** The identity the pack was sealed under, echoed back. */
  experiment_identity_sha256: string;
}

const MODEL_NAME = "AdExNeuron";
// The model's own default timestep is 0.1 ms and its default drive is
// constant. Asking the UI for neither is the case a syntax-only export got
// silently wrong. (Hodgkin-Huxley was the model here until the Studio learned
// that its step is a fixed 1 ms macro step, timed only at its own 0.01 ms.)
const REQUESTED_DT = "0.05";
const REQUESTED_PROTOCOL = "step";

/** A reproducible diffusion experiment accepted by the public equation API. */
const DIFFUSION_WORKSPACE = {
  sourceMode: "ode",
  equations: ["dv/dt = -v + I + 0.1 * xi"],
  threshold: "v > 100",
  reset: "v = 0",
  odeParams: {},
  odeInit: { v: 0 },
  dt: 0.1,
  duration: 20,
  current: 10,
  protocol: "sine",
  frequencyHz: 37,
  seed: 424242,
  trial: "replay",
};

test.describe.configure({ mode: "serial" });

test.beforeEach(async ({ page }) => {
  await page.addInitScript(() => {
    window.localStorage.setItem("sc-studio-onboarding-dismissed", "true");
  });
});

/** Open the Studio on the model this receipt is about, with its detail loaded. */
async function openStudioWithTheModel(page: Page): Promise<void> {
  await page.goto("./");
  await page.getByPlaceholder("Search models...").fill(MODEL_NAME);
  const contract = page.getByTestId(`model-contract-${MODEL_NAME}`);
  await expect(contract).toBeVisible();
  const detail = page.waitForResponse(
    (response) =>
      new URL(response.url()).pathname === `/api/models/${MODEL_NAME}` && response.ok(),
  );
  await contract.locator("..").click();
  await detail;
  await expect(page.getByTestId("slider-tau_w")).toBeVisible();
  await expect(page.getByTestId("slider-g_a")).not.toBeVisible();
}

/**
 * Ask for a timestep and protocol that are not the model's own defaults.
 *
 * A syntax-only export agreed with the defaults by accident; asking for
 * neither is the case that caught it.
 */
async function chooseNondefaultExperiment(page: Page): Promise<void> {
  await page.getByTestId("protocol-select").selectOption(REQUESTED_PROTOCOL);
  await page.getByTestId("slider-dt").fill(REQUESTED_DT);
  await expect(page.getByTestId("protocol-select")).toHaveValue(REQUESTED_PROTOCOL);
}

/**
 * Bind a response to the submitted experiment, excluding intervening auto-runs.
 *
 * @param response - A real server response observed by the browser.
 * @param path - The model or equation simulation route.
 * @param frequency - Expected sine frequency in hertz.
 * @param seed - Expected explicit seed, or absence of one.
 * @param trial - Expected replay or fresh selection.
 * @returns Whether this response belongs to the requested experiment.
 */
function experimentResponse(response: Response, path: string, frequency: number,
  seed: number | null, trial: "replay" | "fresh"): boolean {
  if (new URL(response.url()).pathname !== path || !response.ok()) return false;
  const body = response.request().postDataJSON() as Record<string, unknown>;
  return body.frequency_hz === frequency && (body.seed ?? null) === seed && body.trial === trial;
}

for (const source of ["model", "ode"] as const) {
  test(`a saved ${source} project preserves the experiment through reopen and replay export`, async ({ page, request }, testInfo) => {
    const name = `experiment-${source}-${Date.now()}`;
    const simulationPath = source === "model" ? "/api/models/simulate" : "/api/simulate";
    const seed = source === "model" ? null : DIFFUSION_WORKSPACE.seed;
    if (source === "model") {
      await openStudioWithTheModel(page);
      await page.getByTestId("protocol-select").selectOption("sine");
      await page.getByTestId("slider-f (Hz)").fill("37");
    } else {
      const provisioned = await request.post("/api/project/save", {
        data: { name: `${name}-input`, state: DIFFUSION_WORKSPACE },
      });
      expect(provisioned.ok()).toBe(true);
      await page.goto("./");
      await page.getByRole("button", { name: "Refresh projects", exact: true }).click();
      await page.getByRole("button", { name: `Open project ${name}-input revision 1`, exact: true }).click();
    }
    await expect(page.getByTestId("slider-f (Hz)")).toHaveValue("37");
    await expect(page.getByRole("textbox", { name: "seed", exact: true })).toHaveValue(seed === null ? "" : String(seed));
    await expect(page.getByRole("button", { name: "Run", exact: true })).toBeEnabled();
    const originalRun = page.waitForResponse((response) =>
      experimentResponse(response, simulationPath, 37, seed, "replay"));
    await page.getByRole("button", { name: "Run", exact: true }).click();
    const original = await (await originalRun).json() as SimulateResponse;

    const saved = page.waitForResponse((response) =>
      new URL(response.url()).pathname === "/api/project/save" && response.ok());
    await page.getByLabel("Save project", { exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "Save project" });
    await dialog.getByLabel("Project name").fill(name);
    await dialog.getByRole("button", { name: "Save", exact: true }).click();
    const savedRequest: unknown = (await saved).request().postDataJSON();
    expect(savedRequest).toMatchObject({ state: { frequencyHz: 37, seed, trial: "replay" } });
    const revision = await request.get(`/api/project/load/${name}`);
    expect(revision.ok()).toBe(true);
    const stored: unknown = await revision.json();
    expect(stored).toMatchObject({ state: { frequencyHz: 37, seed, trial: "replay" } });

    const interveningRun = page.waitForResponse((response) =>
      experimentResponse(response, simulationPath, 11, seed === null ? null : 7, "fresh"));
    await page.getByTestId("slider-f (Hz)").fill("11");
    if (seed !== null) await page.getByRole("textbox", { name: "seed", exact: true }).fill("7");
    await page.getByRole("combobox", { name: "trial mode", exact: true }).selectOption("fresh");
    await interveningRun;
    await expect(page.getByRole("button", { name: "Run", exact: true })).toBeEnabled();
    const restoredRun = page.waitForResponse((response) =>
      experimentResponse(response, simulationPath, 37, seed, "replay"));
    await page.getByRole("button", { name: `Open project ${name} revision 1`, exact: true }).click();
    const reopened = await (await restoredRun).json() as SimulateResponse;
    await expect(page.getByTestId("slider-f (Hz)")).toHaveValue("37");
    await expect(page.getByRole("textbox", { name: "seed", exact: true })).toHaveValue(seed === null ? "" : String(seed));
    await expect(page.getByRole("combobox", { name: "trial mode", exact: true })).toHaveValue("replay");
    expect(reopened.experiment).toEqual(original.experiment);
    expect(reopened.raw).toEqual(original.raw);
    await page.getByRole("button", { name: "Data table", exact: true }).click();
    await expect(page.getByRole("table", { name: /^Trace data:/ })).toBeVisible();

    const download = page.waitForEvent("download");
    const packed = page.waitForResponse((response) =>
      new URL(response.url()).pathname === "/api/export/replay-pack" && response.ok());
    await page.getByTestId("export-replay-pack").click();
    const pack = await (await packed).json() as ReplayPack;
    expect(pack.request).toMatchObject({ frequency_hz: 37, trial: "replay" });
    if (seed === null) expect(pack.request).not.toHaveProperty("seed");
    else expect(pack.request).toHaveProperty("seed", seed);
    const packPath = testInfo.outputPath("reopened-project.json");
    await (await download).saveAs(packPath);
    const environment = { ...process.env };
    delete environment.PYTHONPATH;
    const output = execFileSync(REPLAY_PYTHON, ["-m", "sc_neurocore.studio.replay_pack", packPath, "--json"], {
      cwd: testInfo.outputDir, env: environment, encoding: "utf-8", timeout: 120_000,
    });
    const replayed = JSON.parse(output) as ReplayOutcome;
    expect(replayed).toMatchObject({ verdict: "match", differences: [],
      experiment_identity_sha256: pack.experiment_identity_sha256 });
  });
}

test("fresh and legacy project settings reopen explicitly and malformed revisions leave the editor unchanged", async ({ page, request }) => {
  const name = `experiment-compatibility-${Date.now()}`;
  const provisioned = await request.post("/api/project/save", {
    data: { name, state: { ...DIFFUSION_WORKSPACE, trial: "fresh" } },
  });
  expect(provisioned.ok()).toBe(true);
  const initialRun = page.waitForResponse((response) =>
    new URL(response.url()).pathname === "/api/models/simulate" && response.ok());
  await page.goto("./");
  await initialRun;
  await page.getByRole("button", { name: "Refresh projects", exact: true }).click();
  const freshRun = page.waitForResponse((response) =>
    new URL(response.url()).pathname === "/api/simulate" && response.ok());
  await page.getByRole("button", { name: `Open project ${name} revision 1`, exact: true }).click();
  const fresh: unknown = await (await freshRun).json();
  expect(fresh).toMatchObject({ experiment: { randomness: { trial: "fresh", seed_source: "drawn" } } });
  await expect(page.getByRole("combobox", { name: "trial mode", exact: true })).toHaveValue("fresh");
  const saved = page.waitForResponse((response) =>
    new URL(response.url()).pathname === "/api/project/save" && response.ok());
  await page.getByLabel("Save project", { exact: true }).click();
  const dialog = page.getByRole("dialog", { name: "Save project" });
  await dialog.getByLabel("Project name").fill(name);
  await dialog.getByRole("button", { name: "Save", exact: true }).click();
  const freshSaved: unknown = (await saved).request().postDataJSON();
  expect(freshSaved).toMatchObject({ expected_revision: 1,
    state: { frequencyHz: 37, seed: 424242, trial: "fresh" } });

  const legacy: Record<string, unknown> = { ...DIFFUSION_WORKSPACE };
  delete legacy.frequencyHz;
  delete legacy.seed;
  delete legacy.trial;
  expect((await request.post("/api/project/save", { data: { name: `${name}-legacy`, state: legacy } })).ok()).toBe(true);
  await page.getByRole("button", { name: "Refresh projects", exact: true }).click();
  await page.getByRole("button", { name: `Open project ${name}-legacy revision 1`, exact: true }).click();
  await expect(page.getByTestId("slider-f (Hz)")).toHaveValue("10");
  await expect(page.getByRole("textbox", { name: "seed", exact: true })).toHaveValue("");
  await expect(page.getByRole("combobox", { name: "trial mode", exact: true })).toHaveValue("replay");
  await expect(page.getByRole("button", { name: "Run", exact: true })).toBeEnabled();

  const invalidSettings = [
    { patch: { frequencyHz: 0 }, message: "Saved simulation frequency must be finite and positive" },
    { patch: { seed: -1 }, message: "Saved simulation seed must be a non-negative integer below 2^63" },
    { patch: { trial: "independent" }, message: "Saved simulation trial must be replay or fresh" },
  ];
  for (const [index, { patch, message }] of invalidSettings.entries()) {
    const invalidName = `${name}-invalid-${index}`;
    expect((await request.post("/api/project/save", {
      data: { name: invalidName, state: { ...DIFFUSION_WORKSPACE, current: -25, ...patch } },
    })).ok()).toBe(true);
    await page.getByRole("button", { name: "Refresh projects", exact: true }).click();
    const invalidLoaded = page.waitForResponse((response) =>
      new URL(response.url()).pathname === `/api/project/load/${invalidName}` && response.ok());
    await page.getByRole("button", { name: `Open project ${invalidName} revision 1`, exact: true }).click();
    await invalidLoaded;
    await expect(page.getByRole("alert").filter({ hasText: message })).toBeVisible();
    await expect(page.getByTestId("slider-f (Hz)")).toHaveValue("10");
    await expect(page.getByTestId("slider-I")).toHaveValue("10");
    await expect(page.getByRole("textbox", { name: "seed", exact: true })).toHaveValue("");
    await expect(page.getByRole("combobox", { name: "trial mode", exact: true })).toHaveValue("replay");
  }
});

test("the browser exports a script that states the experiment the server resolved", async ({
  page,
}, testInfo) => {
  await openStudioWithTheModel(page);
  await chooseNondefaultExperiment(page);

  const codegen = page.waitForResponse(
    (response) => new URL(response.url()).pathname === "/api/codegen" && response.ok(),
  );
  await page.getByTestId("run-codegen").click();
  const payload = (await (await codegen).json()) as CodegenResponse;

  expect(payload.request).toMatchObject({
    name: MODEL_NAME,
    dt: Number(REQUESTED_DT),
    protocol: REQUESTED_PROTOCOL,
    trial: "replay",
  });
  const script = await page.getByTestId("codegen-script").innerText();
  expect(script).toContain(payload.experiment_sha256);
  const scriptPath = testInfo.outputPath("experiment.py");
  writeFileSync(scriptPath, script, "utf-8");
  const environment = { ...process.env };
  delete environment.PYTHONPATH;
  // Execute exactly what the code panel exposes, including digest admission.
  // The interpreter may use an editable install; wheel isolation is verified
  // separately by test_studio_distribution.py.
  const output = execFileSync(REPLAY_PYTHON, ["-c", `
import contextlib, io, json, runpy, sys
from sc_neurocore.studio.replay_pack import replay_expectation
with contextlib.redirect_stdout(io.StringIO()):
    exported = runpy.run_path(sys.argv[1], run_name="__main__")
print(json.dumps({"request": exported["REQUEST"],
                  "expectation": replay_expectation(exported["result"])}))
`, scriptPath], {
    cwd: testInfo.outputDir, env: environment, encoding: "utf-8", timeout: 120_000,
  });
  const executed: unknown = JSON.parse(output);
  const reference = page.waitForResponse(
    (response) => new URL(response.url()).pathname === "/api/export/replay-pack" && response.ok(),
  );
  await page.getByTestId("export-replay-pack").click();
  const pack = (await (await reference).json()) as ReplayPack;
  expect(executed).toEqual({ request: payload.request, expectation: pack.expectation });
  // The assumptions the previous export hard-coded must not be back.
  expect(script).not.toContain("step(current=");
  expect(script).not.toContain("neuron.v");
  await expect(page.getByTestId("codegen-experiment")).toContainText(
    payload.experiment_sha256.slice(0, 16),
  );
});

test("a downloaded pack replays in a separate interpreter and reports tampering", async ({ page }, testInfo) => {
  await openStudioWithTheModel(page);
  await chooseNondefaultExperiment(page);

  const download = page.waitForEvent("download");
  const packResponse = page.waitForResponse(
    (response) =>
      new URL(response.url()).pathname === "/api/export/replay-pack",
    { timeout: 60_000 },
  );
  await page.getByTestId("export-replay-pack").click();
  const response = await packResponse;
  const failureBody = response.ok() ? "" : await response.text();
  expect(response.status(), failureBody).toBe(200);
  const served = (await response.json()) as ReplayPack;
  const artefact = await download;

  expect(artefact.suggestedFilename()).toContain(
    served.experiment_identity_sha256.slice(0, 12),
  );

  const workspace = testInfo.outputDir;
  const packPath = testInfo.outputPath("pack.json");
  await artefact.saveAs(packPath);

  // What the browser wrote to disk is what the server sealed.
  const saved = JSON.parse(readFileSync(packPath, "utf-8")) as ReplayPack;
  expect(saved).toEqual(served);
  expect(saved.schema_version).toBe("studio.replay-pack.v2");
  expect(saved.request).toMatchObject({
    name: MODEL_NAME,
    dt: Number(REQUESTED_DT),
    protocol: REQUESTED_PROTOCOL,
  });

  // A separate interpreter, outside the repository, with no PYTHONPATH: the
  // pack carries the replay inputs. The installed configuration selects its
  // verified wheel interpreter; the source configurations use PATH.
  const environment = { ...process.env };
  delete environment.PYTHONPATH;
  const replayed = execFileSync(
    REPLAY_PYTHON,
    ["-m", "sc_neurocore.studio.replay_pack", packPath, "--json"],
    { cwd: workspace, env: environment, encoding: "utf-8", timeout: 900_000 },
  );
  const outcome = JSON.parse(replayed) as ReplayOutcome;

  expect(outcome.verdict).toBe("match");
  expect(outcome.differences).toEqual([]);
  expect(outcome.experiment_identity_sha256).toBe(served.experiment_identity_sha256);

  // A pack whose expectation no longer describes the run is reported, not
  // rounded away.
  // `expectation` travels opaquely to the Python runner, so its fields are
  // `unknown` here. Reading the count through a check rather than a cast keeps
  // the tamper honest: if the pack stops carrying a numeric spike count, this
  // fails loudly instead of writing `NaN` into the file and asserting on it.
  const spikeCount = saved.expectation.spike_count;
  if (typeof spikeCount !== "number") {
    throw new TypeError(`the pack's expectation carries no numeric spike_count: ${typeof spikeCount}`);
  }
  const tampered: ReplayPack = {
    ...saved,
    expectation: { ...saved.expectation, spike_count: spikeCount + 3 },
  };
  const tamperedPath = testInfo.outputPath("tampered.json");
  writeFileSync(tamperedPath, JSON.stringify(tampered), "utf-8");
  let status = 0;
  let output: string;
  try {
    output = execFileSync(
      REPLAY_PYTHON,
      ["-m", "sc_neurocore.studio.replay_pack", tamperedPath],
      { cwd: workspace, env: environment, encoding: "utf-8", timeout: 900_000 },
    );
  } catch (failure) {
    const error = failure as { status: number; stdout: string };
    status = error.status;
    output = error.stdout;
  }
  expect(status).toBe(1);
  expect(output).toContain("verdict: mismatch");
});

/**
 * Match a run by its original numeric seed token, without parsing through Number.
 *
 * @param response - The live simulation response.
 * @param token - Exact decimal seed digits expected on the wire.
 * @returns Whether the real simulation request carries those original digits.
 */
function exactSeedResponse(response: Response, token: string): boolean {
  return ["/api/simulate", "/api/models/simulate"].includes(new URL(response.url()).pathname)
    && (response.request().postData() ?? "").includes(`"seed":${token}`);
}

for (const source of ["ode", "model"] as const) {
  test(`a full-domain seed survives serialized project custody, editing, local sessions and independent replay (${source})`, async ({ page, request }, testInfo) => {
    const token = "9223372036854775807";
    const alternate = "9007199254740993";
    const name = `exact-seed-${source}-${Date.now()}`;
    const raw = JSON.stringify({ name: `${name}-input`, state: {
      ...DIFFUSION_WORKSPACE,
      ...(source === "model" ? {
        sourceMode: "model", selectedModelName: "SCStochasticRateAdaptationNeuron",
        dt: 0.5, duration: 200, current: 10, protocol: "constant", modelParams: { tau_a: 37 },
      } : { modelParams: { seed: "__legacy_seed__" } }),
      seed: "__exact_seed__",
    } }).replace('"__exact_seed__"', token).replace('"__legacy_seed__"', alternate);
    const provisioned = await request.post("/api/project/save", {
      data: raw, headers: { "Content-Type": "application/json" },
    });
    expect(provisioned.ok(), await provisioned.text()).toBe(true);
    await page.goto("./");
    await page.getByRole("button", { name: "Refresh projects", exact: true }).click();
    const initial = page.waitForResponse((r) => exactSeedResponse(r, token));
    await page.getByRole("button", { name: `Open project ${name}-input revision 1`, exact: true }).click();
    const first = await initial;
    expect(first.ok(), await first.text()).toBe(true);
    const original = await first.json() as SimulateResponse;
    if (source === "model") {
      expect(await first.text()).toMatch(new RegExp('"parameters":\\{[^}]*"seed":' + token + '(?:[,}])'));
      expect(original.dt).toBe(0.5);
      await expect(page.getByTestId("slider-dt")).toHaveValue("0.5");
      await expect(page.getByTestId("slider-tau_a")).toHaveValue("37");
      const parameter = page.getByTestId("slider-tau_a");
      const changedParameter = page.waitForResponse((r) => exactSeedResponse(r, token)
        && (r.request().postData() ?? "").includes('"tau_a":38.25'));
      await parameter.fill("38.25");
      const parameterResponse = await changedParameter;
      expect(parameterResponse.ok(), await parameterResponse.text()).toBe(true);
      expect((await parameterResponse.json() as SimulateResponse).raw).not.toEqual(original.raw);
      const restoredParameter = page.waitForResponse((r) => exactSeedResponse(r, token)
        && (r.request().postData() ?? "").includes('"tau_a":37'));
      await parameter.fill("37");
      expect((await (await restoredParameter).json() as SimulateResponse).raw).toEqual(original.raw);
    }
    const seed = page.getByRole("textbox", { name: "seed", exact: true });
    await expect(seed).toHaveValue(token);

    const changed = page.waitForResponse((r) => exactSeedResponse(r, alternate));
    await seed.fill(alternate);
    const changedResponse = await changed;
    expect(changedResponse.ok(), await changedResponse.text()).toBe(true);
    expect((await changedResponse.json() as SimulateResponse).raw).not.toEqual(original.raw);
    const restored = page.waitForResponse((r) => exactSeedResponse(r, token));
    await seed.fill(token);
    expect((await (await restored).json() as SimulateResponse).raw).toEqual(original.raw);

    await page.getByLabel("Save project", { exact: true }).click();
    const saveDialog = page.getByRole("dialog", { name: "Save project" });
    await saveDialog.getByLabel("Project name").fill(name);
    const saved = page.waitForResponse((r) => new URL(r.url()).pathname === "/api/project/save");
    await saveDialog.getByRole("button", { name: "Save", exact: true }).click();
    const savedResponse = await saved;
    expect(savedResponse.ok(), await savedResponse.text()).toBe(true);
    expect(savedResponse.request().postData()).toContain(`"seed":${token}`);
    if (source === "ode") expect(savedResponse.request().postData()).toContain(`"seed":${alternate}`);
    const stored = await request.get(`/api/project/load/${name}`);
    const storedText = await stored.text();
    expect(storedText).toMatch(/"seed"\s*:\s*9223372036854775807/);
    if (source === "ode") expect(storedText).toMatch(/"seed"\s*:\s*9007199254740993/);

    await page.getByLabel("Save session", { exact: true }).click();
    const sessionDialog = page.getByRole("dialog", { name: "Save session" });
    await sessionDialog.getByLabel("Session name").fill(name);
    await sessionDialog.getByRole("button", { name: "Save", exact: true }).click();
    const cached = await page.evaluate(() => localStorage.getItem("sc-studio-sessions"));
    expect(cached).toContain(`"seed":${token}`);
    if (source === "ode") expect(cached).toContain(`"seed":${alternate}`);
    const other = page.waitForResponse((r) => exactSeedResponse(r, alternate));
    await seed.fill(alternate);
    await other;
    const reopened = page.waitForResponse((r) => exactSeedResponse(r, token));
    await page.getByRole("button", { name: `Open session ${name}`, exact: true }).click();
    expect((await (await reopened).json() as SimulateResponse).raw).toEqual(original.raw);
    await expect(seed).toHaveValue(token);

    const packed = page.waitForResponse((r) => new URL(r.url()).pathname === "/api/export/replay-pack");
    const download = page.waitForEvent("download");
    await page.getByTestId("export-replay-pack").click();
    const packResponse = await packed;
    expect(packResponse.ok(), await packResponse.text()).toBe(true);
    const path = testInfo.outputPath("exact-seed-replay.json");
    await (await download).saveAs(path);
    expect(readFileSync(path, "utf8")).toMatch(/"seed"\s*:\s*9223372036854775807/);
    const environment = { ...process.env };
    delete environment.PYTHONPATH;
    const result = JSON.parse(execFileSync(REPLAY_PYTHON, ["-m", "sc_neurocore.studio.replay_pack", path, "--json"], {
      cwd: testInfo.outputDir, env: environment, encoding: "utf8", timeout: 120_000,
    })) as ReplayOutcome;
    expect(result).toMatchObject({ verdict: "match", differences: [] });

    await page.reload();
    const afterReload = page.waitForResponse((r) => exactSeedResponse(r, token));
    await page.getByRole("button", { name: `Open session ${name}`, exact: true }).click();
    expect((await (await afterReload).json() as SimulateResponse).raw).toEqual(original.raw);
    await expect(page.getByRole("textbox", { name: "seed", exact: true })).toHaveValue(token);

    // An explicit malformed local seed must not replace the currently replayed experiment.
    await page.evaluate(() => {
      const cached = localStorage.getItem("sc-studio-sessions");
      if (!cached) throw new Error("The saved session cache is missing.");
      localStorage.setItem("sc-studio-sessions",
        cached.slice(0, -1) + ',{"name":"invalid-exact-seed","state":{"seed":1.5}}]');
    });
    await page.reload();
    const validAgain = page.waitForResponse((r) => exactSeedResponse(r, token));
    await page.getByRole("button", { name: `Open session ${name}`, exact: true }).click();
    await validAgain;
    await page.getByRole("button", { name: "Open session invalid-exact-seed", exact: true }).click();
    await expect(page.getByRole("alert").filter({ hasText: "Saved simulation seed must be a non-negative integer below 2^63" })).toBeVisible();
    await expect(page.getByRole("textbox", { name: "seed", exact: true })).toHaveValue(token);

    if (source === "model") {
      const detailPath = "/api/models/SCStochasticRateAdaptationNeuron";
      // Delay a real metadata request; forward it unchanged or inject a transport failure.
      for (const outcome of ["forward", "abort"] as const) {
        let releaseRead: () => void = () => { throw new Error("Metadata hold was not installed"); };
        let observeRead: () => void = () => { throw new Error("Metadata observer was not installed"); };
        const held = new Promise<void>((resolve) => { releaseRead = resolve; });
        const started = new Promise<void>((resolve) => { observeRead = resolve; });
        await page.route(`**${detailPath}`, async (route) => {
          observeRead();
          await held;
          if (outcome === "forward") await route.continue();
          else await route.abort("failed");
        });
        await page.getByRole("button", { name: `Open session ${name}`, exact: true }).click();
        await started;
        await page.getByPlaceholder("Search models...").fill("AdExNeuron");
        const selected = page.waitForResponse((r) => new URL(r.url()).pathname === "/api/models/AdExNeuron");
        await page.getByTestId("model-contract-AdExNeuron").locator("..").click();
        await selected;
        await expect(page.getByTestId("slider-tau_w")).toBeVisible();
        const settled = outcome === "forward"
          ? page.waitForResponse((r) => new URL(r.url()).pathname === detailPath)
          : page.waitForEvent("requestfailed", { predicate: (r) => new URL(r.url()).pathname === detailPath });
        releaseRead();
        await settled;
        await expect(page.getByTestId("slider-tau_w")).toBeVisible();
        await expect(page.getByTestId("slider-tau_a")).toHaveCount(0);
        await expect(page.getByRole("alert")).toHaveCount(0);
        await page.unroute(`**${detailPath}`);
        const recovered = page.waitForResponse((r) => exactSeedResponse(r, token));
        await page.getByRole("button", { name: `Open session ${name}`, exact: true }).click();
        expect((await (await recovered).json() as SimulateResponse).raw).toEqual(original.raw);
        await expect(page.getByTestId("slider-tau_a")).toBeVisible();
      }

      const missing = `${name}-missing-model`;
      const savedMissing = await request.post("/api/project/save", { data: {
        name: missing, state: { sourceMode: "model", selectedModelName: "UnknownModelForSavedWorkspace", seed: 42 },
      } });
      expect(savedMissing.ok(), await savedMissing.text()).toBe(true);
      await page.getByRole("button", { name: "Refresh projects", exact: true }).click();
      const refusedDetail = page.waitForResponse((r) => new URL(r.url()).pathname === "/api/models/UnknownModelForSavedWorkspace");
      await page.getByRole("button", { name: `Open project ${missing} revision 1`, exact: true }).click();
      expect((await refusedDetail).status()).toBe(404);
      await expect(page.getByRole("alert").filter({ hasText: /404|not found|unknown model/i })).toBeVisible();
      await expect(page.getByRole("textbox", { name: "seed", exact: true })).toHaveValue("42");
      await expect(page.getByTestId("export-replay-pack")).toBeDisabled();
    }
  });
}

test("browser session writes report real quota failures and recover without losing the stored list", async ({ page, request }, testInfo) => {
  const project = `session-quota-${Date.now()}`;
  const baseline = "durable";
  const unsaved = `unsaved-${"x".repeat(60)}`;
  const recovered = "recovered";
  const provisioned = await request.post("/api/project/save", {
    data: { name: project, state: DIFFUSION_WORKSPACE },
  });
  expect(provisioned.ok()).toBe(true);
  const serverBefore = await (await request.get(`/api/project/load/${project}`)).json() as unknown;
  await page.coverage.startJSCoverage({ resetOnNavigation: false });
  await page.goto("./");
  await page.getByRole("button", { name: "Refresh projects", exact: true }).click();
  await page.getByRole("button", { name: `Open project ${project} revision 1`, exact: true }).click();
  await expect(page.getByTestId("slider-f (Hz)")).toHaveValue("37");

  /**
   * Save the current experiment through the actual session dialog.
   *
   * @param name - The local session name.
   */
  async function saveSession(name: string): Promise<void> {
    await page.getByRole("button", { name: "Save session", exact: true }).click();
    const dialog = page.getByRole("dialog", { name: "Save session" });
    await dialog.getByLabel("Session name").fill(name);
    await dialog.getByRole("button", { name: "Save", exact: true }).click();
    await expect(dialog).not.toBeVisible();
  }

  await saveSession(baseline);
  await expect(page.getByRole("button", { name: `Open session ${baseline}`, exact: true })).toBeVisible();
  const storedBefore = await page.evaluate(() => localStorage.getItem("sc-studio-sessions"));
  expect(storedBefore).not.toBeNull();
  const quota = await page.evaluate(() => {
    let low = 0;
    let high = 8 * 1024 * 1024;
    let refused = "";
    while (low < high) {
      const size = Math.ceil((low + high) / 2);
      try {
        localStorage.setItem("sc-studio-quota-test", "x".repeat(size));
        low = size;
      } catch (error) {
        if (!(error instanceof DOMException) || error.name !== "QuotaExceededError") throw error;
        refused = error.name;
        high = size - 1;
      }
    }
    return { refused, fillerCharacters: low, stored: localStorage.getItem("sc-studio-sessions") };
  });
  expect(quota.refused).toBe("QuotaExceededError");
  expect(quota.fillerCharacters).toBeGreaterThan(1024 * 1024);
  expect(quota.stored).toBe(storedBefore);
  await page.getByTestId("slider-f (Hz)").fill("41");
  await saveSession(unsaved);
  await expect(page.getByRole("button", { name: `Open session ${unsaved}`, exact: true })).toBeVisible();
  const afterFailedSave = await page.evaluate(() => localStorage.getItem("sc-studio-sessions"));
  expect(afterFailedSave).toBe(storedBefore);
  await testInfo.attach("real-browser-quota-before-feedback", {
    body: JSON.stringify({ quota, afterFailedSave, unsaved }), contentType: "application/json",
  });
  const notice = page.getByTestId("session-storage-status");
  await expect(notice).toHaveAttribute("role", "alert", { timeout: 5000 });
  await expect(notice).toContainText("local storage is full");
  await expect(notice).toContainText("this page only");
  await expect(notice).toContainText("Reloading may restore the last saved list");

  await page.getByRole("button", { name: `Delete session ${baseline}`, exact: true }).click();
  await expect(page.getByRole("button", { name: `Open session ${baseline}`, exact: true })).not.toBeVisible();
  await expect(notice).toHaveAttribute("role", "alert");
  expect(await page.evaluate(() => localStorage.getItem("sc-studio-sessions"))).toBe(storedBefore);
  await page.getByRole("button", { name: `Open session ${unsaved}`, exact: true }).click();
  await expect(page.getByTestId("slider-f (Hz)")).toHaveValue("41");
  await expect(page.getByRole("button", { name: "Run", exact: true })).toBeEnabled();
  const memoryRun = page.waitForResponse((response) =>
    experimentResponse(response, "/api/simulate", 41, 424242, "replay"));
  await page.getByRole("button", { name: "Run", exact: true }).click();
  const original = await (await memoryRun).json() as SimulateResponse;

  await page.reload();
  await expect(page.getByRole("button", { name: `Open session ${baseline}`, exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: `Open session ${unsaved}`, exact: true })).not.toBeVisible();
  await page.evaluate(() => { localStorage.removeItem("sc-studio-quota-test"); });
  await page.getByRole("button", { name: `Open session ${baseline}`, exact: true }).click();
  await expect(page.getByTestId("slider-f (Hz)")).toHaveValue("37");
  await page.getByTestId("slider-f (Hz)").fill("41");
  await saveSession(recovered);
  await expect(page.getByTestId("session-storage-status")).toHaveAttribute("role", "status");
  await expect(page.getByTestId("session-storage-status")).toContainText("saved in this browser");
  await page.getByRole("button", { name: `Delete session ${baseline}`, exact: true }).click();
  await expect(page.getByTestId("session-storage-status")).toHaveAttribute("role", "status");
  await page.reload();
  await expect(page.getByRole("button", { name: `Open session ${baseline}`, exact: true })).not.toBeVisible();
  await expect(page.getByRole("button", { name: `Open session ${recovered}`, exact: true })).toBeVisible();
  await page.getByRole("button", { name: `Open session ${recovered}`, exact: true }).click();
  await expect(page.getByTestId("slider-f (Hz)")).toHaveValue("41");
  await expect(page.getByRole("button", { name: "Run", exact: true })).toBeEnabled();
  const recoveredRun = page.waitForResponse((response) =>
    experimentResponse(response, "/api/simulate", 41, 424242, "replay"));
  await page.getByRole("button", { name: "Run", exact: true }).click();
  expect((await (await recoveredRun).json() as SimulateResponse).raw).toEqual(original.raw);
  const serverAfter = await (await request.get(`/api/project/load/${project}`)).json() as unknown;
  expect(serverAfter).toEqual(serverBefore);
  const coveragePath = testInfo.outputPath("session-browser-v8.json");
  writeFileSync(coveragePath, JSON.stringify(await page.coverage.stopJSCoverage()));
  await testInfo.attach("session-browser-v8", { path: coveragePath, contentType: "application/json" });
});
