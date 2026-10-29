// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Browser equation authoring through real native compiler workers

import { readFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { expect, test, type Page, type Response } from "@playwright/test";
import type { IRBuildResponse, EquationIRResult } from "../src/api/equationIrTypes";
import type { StudioEvidenceBundleResponse } from "../src/api/types";
import { defaultEquationIrConfiguration, type EquationIrSourceInput } from "../src/equationIrConfig";

/** Recognise one real native compiler response without replacing its payload. */
function operation(path: string): (response: Response) => boolean {
  return (response) => new URL(response.url()).pathname === `/api/ir/${path}`;
}

/**
 * Reopen authored source through the public project store and visible UI.
 *
 * @param page - Real browser connected to the production Studio application.
 * @param name - Test-owned project namespace.
 * @param source - Explicit authored source overrides preserved in the saved project.
 */
async function openEquationProject(page: Page, name: string, source: Partial<EquationIrSourceInput> = {}): Promise<void> {
  await page.addInitScript(() => {
    localStorage.setItem("sc-studio-onboarding-dismissed", "true");
  });
  const saved = await page.request.post("/api/project/save", {
    headers: { "Sec-Fetch-Site": "none" },
    data: { name, state: { sourceMode: "ode", equations: ["dv/dt = I"],
      threshold: "", reset: "", odeParams: {}, odeInit: { v: 7 },
      dt: 0.5, duration: 2, current: 1, protocol: "constant", frequencyHz: 10,
      seed: null, trial: "replay", ...source } },
  });
  expect(saved.ok(), await saved.text()).toBe(true);
  await page.goto("./");
  await page.getByRole("button", { name: "Refresh projects", exact: true }).click();
  const simulated = page.waitForResponse((response) =>
    new URL(response.url()).pathname === "/api/simulate");
  await page.getByRole("button", { name: `Open project ${name} revision 1`, exact: true }).click();
  const simulation = await simulated;
  expect(simulation.ok(), await simulation.text()).toBe(true);
  await page.getByRole("tab", { name: "IR", exact: true }).click();
  await page.getByText("Equation compiler settings", { exact: true }).click();
}

/**
 * Build through the real Inspector action and retain the actual HTTP receipt.
 *
 * @param page - Browser showing the equation Inspector.
 * @returns Source-bound native graph produced by its isolated worker.
 */
async function build(page: Page): Promise<IRBuildResponse> {
  const response = page.waitForResponse(operation("build"));
  await page.locator('[aria-label="Native equation compiler actions"]')
    .getByRole("button", { name: "Build IR", exact: true }).click();
  const result = await response;
  expect(result.ok(), await result.text()).toBe(true);
  const body = await result.json() as IRBuildResponse;
  expect(body).toMatchObject({ operation: "build", valid: true, errors: [],
    source_present: true, source_replay: "not_requested" });
  expect(body.studio_job_receipt.job_id).toMatch(/^sj_/);
  expect(body.n_ops).toBeGreaterThan(8);
  expect(body.ir_text).not.toContain("lif_step");
  await expect(page.getByText("Graph structure valid", { exact: true })).toBeVisible();
  return body;
}

test("authored native compiler inputs survive graph replay, RTL export and project/session custody", async ({ page }, info) => {
  const name = `native-equation-${Date.now()}`;
  await openEquationProject(page, name);
  await page.getByLabel("Total scalar bits", { exact: true }).fill("32");
  await page.getByLabel("Fractional bits", { exact: true }).fill("16");
  await page.getByLabel("Integrator", { exact: true }).selectOption("rk4");
  await page.getByLabel("Compiler random key", { exact: true }).fill("18446744073709551615");
  await page.getByLabel("Compiler random stream", { exact: true }).fill("4294967295");
  const first = await build(page);
  expect(first.output_widths).toContainEqual({ name: "v_out", width: 32 });
  await expect(page.getByText("Source replay not run", { exact: true })).toBeVisible();
  for (const [label, route] of [["Check graph structure", "verify"], ["Replay source", "replay"], ["Emit SV", "emit-sv"]] as const) {
    const completed = page.waitForResponse(operation(route));
    await page.getByRole("button", { name: label, exact: true }).click();
    const response = await completed;
    expect(response.ok(), await response.text()).toBe(true);
    const result = await response.json() as EquationIRResult;
    expect(result.source_sha256).toBe(first.source_sha256);
    expect(result.source_replay).toBe(route === "verify" ? "not_requested" : "passed");
  }
  await expect(page.getByLabel("Generated SystemVerilog", { exact: true }))
    .toContainText("module sc_equation_neuron");
  // The toolbar is a second public entry point for the authored native path.
  // Its real worker receipt must bind the same source and graph as the Inspector.
  const toolbarResponse = page.waitForResponse(operation("emit-sv"));
  await page.getByRole("button", { name: "Emit SystemVerilog", exact: true }).click();
  const emittedResponse = await toolbarResponse;
  expect(emittedResponse.ok(), await emittedResponse.text()).toBe(true);
  expect(emittedResponse.request().postDataJSON()).toEqual({ ir_text: first.ir_text });
  const emitted = await emittedResponse.json() as EquationIRResult;
  expect(emitted).toMatchObject({ operation: "emit-sv", valid: true, errors: [],
    source_present: true, source_replay: "passed", ir_text: first.ir_text,
    source_sha256: first.source_sha256, source_graph_sha256: first.source_graph_sha256 });
  expect(emitted.studio_job_receipt.job_id).toMatch(/^sj_/);
  expect(emitted.studio_job_receipt.job_id).not.toBe(first.studio_job_receipt.job_id);
  expect(emitted.systemverilog).toContain("module sc_equation_neuron");
  await expect(page.getByText(`job ${emitted.studio_job_receipt.job_id}`, { exact: true })).toBeVisible();
  const rtlDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download SystemVerilog", exact: true }).click();
  const rtlPath = info.outputPath("authored-native-systemverilog.sv");
  await (await rtlDownload).saveAs(rtlPath);
  expect(readFileSync(rtlPath, "utf-8")).toBe(emitted.systemverilog);

  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download compiler request", exact: true }).click();
  const requestPath = info.outputPath("authored-compiler-request.json");
  await (await download).saveAs(requestPath);
  const downloaded: unknown = JSON.parse(readFileSync(requestPath, "utf-8"));
  expect(downloaded).toMatchObject({ schema_version: "studio.equation-ir-request.v1",
    model: { initial_state: [{ name: "v", value: 7, unit: null }], method: "rk4",
      rng_seed: "18446744073709551615", random_stream: 4294967295 },
    profile: { data_width: 32, fraction: 16 } });

  const exported = page.waitForResponse((response) =>
    new URL(response.url()).pathname === "/api/studio/evidence/bundle");
  await page.getByRole("button", { name: "Export compile evidence bundle", exact: true }).click();
  const exportResponse = await exported;
  expect(exportResponse.ok(), await exportResponse.text()).toBe(true);
  const bundle = await exportResponse.json() as StudioEvidenceBundleResponse;
  expect(exportResponse.request().postDataJSON()).toMatchObject({
    command_replay: { method: "POST", route: "/api/ir/replay", body: { ir_text: first.ir_text } },
    job_ids: [emitted.studio_job_receipt.job_id],
  });
  expect(bundle.summary.source_job_count).toBe(1);
  await expect(page.getByText(`bundle ${bundle.bundle_id}`, { exact: true })).toBeVisible();
  const manifestDownload = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download compile evidence artifact evidence/manifest.json", exact: true }).click();
  const manifestPath = info.outputPath("native-compile-evidence-manifest.json");
  await (await manifestDownload).saveAs(manifestPath);
  const manifestBytes = readFileSync(manifestPath);
  const manifest: unknown = JSON.parse(manifestBytes.toString("utf-8"));
  const { manifest_artifact: artifact, ...sealedManifest } = bundle.manifest;
  expect(manifest).toEqual(sealedManifest);
  expect(artifact).toMatchObject({ bundle_path: "evidence/manifest.json", type: "manifest",
    sha256: createHash("sha256").update(manifestBytes).digest("hex"), size_bytes: manifestBytes.length });

  await page.getByLabel("Save project", { exact: true }).click();
  const projectDialog = page.getByRole("dialog", { name: "Save project" });
  await projectDialog.getByLabel("Project name").fill(name);
  const saved = page.waitForResponse((response) => new URL(response.url()).pathname === "/api/project/save");
  await projectDialog.getByRole("button", { name: "Save", exact: true }).click();
  const savedResponse = await saved;
  expect(savedResponse.ok(), await savedResponse.text()).toBe(true);
  const state: unknown = savedResponse.request().postDataJSON();
  expect(state).toMatchObject({ state: { equationIrConfig: { model: { rng_seed: "18446744073709551615" } } } });
  await page.getByLabel("Save session", { exact: true }).click();
  const sessionDialog = page.getByRole("dialog", { name: "Save session" });
  await sessionDialog.getByLabel("Session name").fill(name);
  await sessionDialog.getByRole("button", { name: "Save", exact: true }).click();
  await page.getByLabel("Compiler random key", { exact: true }).fill("123");
  await expect(page.getByText("Source replay passed", { exact: true })).toHaveCount(0);
  await page.getByRole("button", { name: `Open session ${name}`, exact: true }).click();
  await expect(page.getByLabel("Compiler random key", { exact: true })).toHaveValue("18446744073709551615");
  await expect(page.getByLabel("Integrator", { exact: true })).toHaveValue("rk4");
  await expect(page.getByLabel("Total scalar bits", { exact: true })).toHaveValue("32");
  const reopened = await build(page);
  expect(reopened.source_sha256).toBe(first.source_sha256);
  expect(reopened.source_graph_sha256).toBe(first.source_graph_sha256);
});

test("authored native compiler inputs survive graph replay with partial initial and held state", async ({ page }, info) => {
  const configuration = defaultEquationIrConfiguration();
  configuration.model.method = "map";
  configuration.model.rng_seed = "18446744073709551615";
  configuration.model.random_stream = "4294967295";
  configuration.profile.function_lowering = "v2";
  const name = `native-held-equation-${Date.now()}`;
  await openEquationProject(page, name, {
    equations: ["dv/dt = I**1.5+clip(I,.25,1)**clip(w,-1,1)"],
    odeInit: { w: 0.5 }, dt: 1, equationIrConfig: configuration,
  });
  await expect(page.getByRole("slider", { name: "w₀", exact: true })).toHaveValue("0.5");
  const first = await build(page);
  expect(first.output_widths).toEqual([
    { name: "v_out", width: 16 }, { name: "w_out", width: 16 },
    { name: "spike_out", width: 1 },
  ]);
  for (const [label, route] of [["Replay source", "replay"], ["Emit SV", "emit-sv"]] as const) {
    const completed = page.waitForResponse(operation(route));
    await page.getByRole("button", { name: label, exact: true }).click();
    const response = await completed;
    expect(response.ok(), await response.text()).toBe(true);
    const result = await response.json() as EquationIRResult;
    expect(result.ir_text).toBe(first.ir_text);
    expect(result.source_replay).toBe("passed");
    if (route === "emit-sv") expect(result.systemverilog).toContain("w_out");
  }
  const download = page.waitForEvent("download");
  await page.getByRole("button", { name: "Download compiler request", exact: true }).click();
  const requestPath = info.outputPath("held-state-compiler-request.json");
  await (await download).saveAs(requestPath);
  const request: unknown = JSON.parse(readFileSync(requestPath, "utf-8"));
  expect(request).toMatchObject({ model: {
    equations: [{ name: "v", expression: "I**1.5+clip(I,.25,1)**clip(w,-1,1)" }],
    initial_state: [{ name: "w", value: 0.5, unit: null }], method: "map",
    rng_seed: "18446744073709551615", random_stream: 4294967295,
  }, profile: { function_lowering: "v2" } });
  await page.reload();
  await page.getByRole("button", { name: "Refresh projects", exact: true }).click();
  const simulated = page.waitForResponse((response) => new URL(response.url()).pathname === "/api/simulate");
  await page.getByRole("button", { name: `Open project ${name} revision 1`, exact: true }).click();
  expect((await simulated).ok()).toBe(true);
  await page.getByRole("tab", { name: "IR", exact: true }).click();
  await page.getByText("Equation compiler settings", { exact: true }).click();
  const reopened = await build(page);
  expect(reopened.source_sha256).toBe(first.source_sha256);
  expect(reopened.source_graph_sha256).toBe(first.source_graph_sha256);
});

test("native compiler rejects stale responses and displays actual unsupported-profile refusals", async ({ page }) => {
  await openEquationProject(page, `native-owner-${Date.now()}`);
  let release = (): void => undefined;
  const released = new Promise<void>((resolve) => { release = resolve; });
  let resolveBuilt = (_result: IRBuildResponse): void => undefined;
  let rejectBuilt = (_error: unknown): void => undefined;
  const produced = new Promise<IRBuildResponse>((resolve, reject) => { resolveBuilt = resolve; rejectBuilt = reject; });
  await page.route("**/api/ir/build", async (route) => {
    try {
      const response = await route.fetch();
      expect(response.ok(), await response.text()).toBe(true);
      resolveBuilt(await response.json() as IRBuildResponse);
      await released;
      await route.fulfill({ response });
    } catch (error: unknown) { rejectBuilt(error); throw error; }
  }, { times: 1 });
  const received = page.waitForResponse(operation("build"));
  await page.locator('[aria-label="Native equation compiler actions"]')
    .getByRole("button", { name: "Build IR", exact: true }).click();
  try {
    const obsolete = await produced;
    expect(obsolete.source_present).toBe(true);
    await page.getByLabel("Compiler random key", { exact: true }).fill("123");
    release();
    await received;
    await expect(page.getByText("Graph structure valid", { exact: true })).toHaveCount(0);
    await expect(page.getByRole("button", { name: "Replay source", exact: true })).toBeDisabled();
  } finally { release(); }
  await build(page);
  await page.getByLabel("Total scalar bits", { exact: true }).fill("72");
  const rejected = page.waitForResponse(operation("build"));
  await page.locator('[aria-label="Native equation compiler actions"]')
    .getByRole("button", { name: "Build IR", exact: true }).click();
  const refusal = await rejected;
  expect(refusal.status()).toBe(422);
  expect(await refusal.json()).toMatchObject({ detail: { stage: "lowering", code: "unsupported_request" } });
  await expect(page.getByRole("alert").filter({ hasText: "The compiler does not support this request." })).toBeVisible();
  await expect(page.getByText("Graph structure valid", { exact: true })).toHaveCount(0);
});
