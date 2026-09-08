// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Direct analysis request ownership through the public store

import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { studioGuidedFlowInputs } from "../studioGuidedFlowInputs";
import { studioSimulationConfig } from "../studioSimulationConfig";
import { studioSimulationConfigInput } from "../studioSimulationConfigInput";
import { studioExperimentKey } from "../studioExperimentKey";
import { useStudioStore } from "./studio";

const initial = useStudioStore.getState();
const cases = [
  ["runPrecision", "precResult"], ["runCompare", "compareResult"],
  ["runNullclines", "nullclineResult"], ["runFreqResponse", "freqResult"],
] as const;
type Action = typeof cases[number][0];

beforeEach(() => {
  useStudioStore.setState({ sourceMode: "ode", equations: ["dv/dt = -v + w + I", "dw/dt = -w"], odeInit: { v: 0, w: 1 } });
});
afterEach(() => {
  useStudioStore.setState(initial, true);
  vi.unstubAllGlobals();
});

/**
 * Invoke the public action with a complete comparison configuration if needed.
 *
 * @param action - Direct analysis to run.
 * @returns Its completion promise.
 */
function run(action: Action): Promise<void> {
  const state = useStudioStore.getState();
  return action === "runCompare"
    ? state.runCompare(studioSimulationConfig(studioSimulationConfigInput(state)))
    : state[action]();
}

/** Read the production guided-flow projection. */
function complete(): boolean {
  return studioGuidedFlowInputs(useStudioStore.getState(), {
    trainingSkipped: false, evidenceExportSatisfied: false,
  }).analysisComplete;
}

/**
 * Build response payloads with the fields their plot consumers expect.
 *
 * @param action - Analysis response type.
 * @returns The controlled server response, not scientific evidence.
 */
function payload(action: Action): object {
  const sim = {
    time: [0, 1], dt: 1, n_steps: 2, states: { v: [0, 1], w: [1, 0] },
    current_trace: [1, 1], spikes: [], spike_count: 0,
    stats: { rate_hz: 0, isi_mean_ms: null, isi_cv: null, isi_histogram: null },
  };
  const analysis_metadata = {
    schema_version: "studio.analysis-result.v1", analysis_type: {
      runPrecision: "precision", runCompare: "compare",
      runNullclines: "nullclines", runFreqResponse: "frequency_response",
    }[action],
    source: "ode", status: "completed", evidence_classification: "analysis",
    input_sha256: "a".repeat(64), result_sha256: "b".repeat(64), output_keys: [],
  };
  switch (action) {
    case "runPrecision": return { analysis_metadata, float_result: sim, fixed_result: sim,
      error: { variable: "v", max_error: 0, mean_error: 0, rms_error: 0, trace: [0, 0] }, quantized_params: {} };
    case "runCompare": return { analysis_metadata, a: sim, b: sim };
    case "runNullclines": return { analysis_metadata, var_names: ["v", "w"],
      nullcline_0: { variable: "v", points: [[0, 1]] }, nullcline_1: { variable: "w", points: [[0, 0]] } };
    case "runFreqResponse": return { analysis_metadata, frequencies_hz: [1, 2], rates: [0, 1], amplitude: 1 };
  }
}

it.each(["current", "stale", "empty", "nonfinite", "large"] as const)("validates nullcline range source: %s", async (condition) => {
  const samples = condition === "large" ? Array.from({ length: 200000 }, (_, index) => index)
    : condition === "empty" ? [] : condition === "nonfinite" ? [Number.NaN] : [100, 200];
  vi.stubGlobal("fetch", vi.fn<typeof globalThis.fetch>().mockResolvedValue(new Response(JSON.stringify({
    time: samples.map((_, index) => index), dt: 1, n_steps: samples.length,
    states: { v: samples, w: samples.map((_, index) => 3 + index / Math.max(1, samples.length - 1)) },
    current_trace: samples.map(() => 1), spikes: [], spike_count: 0, stats: { rate_hz: 0 },
  }))));
  await useStudioStore.getState().runSimulation();
  const key = studioExperimentKey(studioSimulationConfigInput(useStudioStore.getState()));
  expect(useStudioStore.getState().resultExperimentKey).toBe(key);
  if (condition === "stale") useStudioStore.getState().setDuration(200);
  const fetch = vi.fn<typeof globalThis.fetch>().mockResolvedValue(new Response(JSON.stringify(payload("runNullclines"))));
  vi.stubGlobal("fetch", fetch);
  await useStudioStore.getState().runNullclines();
  if (condition === "empty" || condition === "nonfinite") {
    expect(fetch).not.toHaveBeenCalled();
    expect(useStudioStore.getState().error).toContain(condition === "empty" ? "no samples" : "non-finite samples");
    expect(useStudioStore.getState().isSimulating).toBe(false);
    return;
  }
  const body = fetch.mock.calls[0]?.[1]?.body;
  if (typeof body !== "string") throw new Error("Nullcline request absent");
  const request: unknown = JSON.parse(body);
  expect(request).toMatchObject({ ranges: condition === "stale" ? { v: [-80, 40], w: [-2, 2] }
    : { v: condition === "large" ? [-10, 200009] : [90, 210], w: [2.5, 4.5] } });
});

/**
 * Hold the HTTP boundary until the case chooses success or failure.
 *
 * @param action - Payload to release.
 * @returns A function resolving or rejecting the outstanding fetch.
 */
function deferred(action: Action): (fail?: boolean) => void {
  let finish = (_fail?: boolean): void => { throw new Error("no request"); };
  vi.stubGlobal("fetch", vi.fn<typeof globalThis.fetch>(() => new Promise<Response>((resolve, reject) => {
    finish = (fail = false) => {
      if (fail) reject(new Error("analysis offline"));
      else resolve(new Response(JSON.stringify(payload(action)), { headers: { "Content-Type": "application/json" } }));
    };
  })));
  return (fail) => { finish(fail); };
}

it.each(cases)("%s withdraws success through rerun failure and restores it on recovery", async (action, field) => {
  let finish = deferred(action);
  let pending = run(action);
  finish();
  await pending;
  expect(complete()).toBe(true);
  const old = useStudioStore.getState()[field];
  finish = deferred(action);
  pending = run(action);
  const pendingComplete = complete();
  finish(true);
  await pending;
  expect(pendingComplete).toBe(false);
  expect(complete()).toBe(false);
  expect(useStudioStore.getState()[field]).toBe(old);
  finish = deferred(action);
  pending = run(action);
  finish();
  await pending;
  expect(complete()).toBe(true);
  expect(useStudioStore.getState().error).toBeNull();
});

it.each(cases)("%s drops stale failures after a source change", async (action) => {
  const finish = deferred(action);
  const pending = run(action);
  useStudioStore.getState().setSourceMode("model");
  finish(true);
  await pending;
  expect(useStudioStore.getState().error).toBeNull();
  expect(useStudioStore.getState().isSimulating).toBe(false);
  expect(complete()).toBe(false);
});

it.each(cases)("%s drops stale successes after a source change", async (action, field) => {
  const finish = deferred(action);
  const pending = run(action);
  useStudioStore.getState().setSourceMode("model");
  finish();
  await pending;
  expect(useStudioStore.getState()[field]).toBeNull();
  expect(useStudioStore.getState().isSimulating).toBe(false);
  expect(complete()).toBe(false);
});

it.each(cases)("%s refuses non-finite input without sending a request", async (action) => {
  const fetch = vi.fn<typeof globalThis.fetch>();
  vi.stubGlobal("fetch", fetch);
  useStudioStore.setState({ dt: Number.NaN });
  await run(action);
  expect(fetch).not.toHaveBeenCalled();
  expect(useStudioStore.getState().error).toContain("NaN");
  expect(useStudioStore.getState().analysisExperimentKey).toBeNull();
  expect(useStudioStore.getState().isSimulating).toBe(false);
});

it.each(cases)("%s cannot start over another direct analysis", async (action) => {
  const finish = deferred("runFreqResponse");
  const pending = run("runFreqResponse");
  const before = useStudioStore.getState();
  await run(action);
  expect(useStudioStore.getState()).toBe(before);
  finish();
  await pending;
  expect(complete()).toBe(true);
});

it.each([false, true])("reports invalid current input when the response fails=%s", async (fail) => {
  const finish = deferred("runPrecision");
  const pending = run("runPrecision");
  useStudioStore.setState({ dt: Number.NaN });
  finish(fail);
  await pending;
  expect(useStudioStore.getState().error).toContain("NaN");
  expect(useStudioStore.getState().isSimulating).toBe(false);
  expect(useStudioStore.getState().analysisExperimentKey).toBeNull();
});

it.each(["runPrecision", "runNullclines"] as const)("%s reports unsupported source without requesting", async (action) => {
  const fetch = vi.fn<typeof globalThis.fetch>();
  vi.stubGlobal("fetch", fetch);
  useStudioStore.getState().setSourceMode("model");
  await run(action);
  expect(fetch).not.toHaveBeenCalled();
  expect(useStudioStore.getState().error).not.toBeNull();
  expect(complete()).toBe(false);
});

it("reports missing nullcline initial variables instead of silent idle", async () => {
  const fetch = vi.fn<typeof globalThis.fetch>();
  vi.stubGlobal("fetch", fetch);
  useStudioStore.setState({ odeInit: { v: 0 } });
  await run("runNullclines");
  expect(fetch).not.toHaveBeenCalled();
  expect(useStudioStore.getState().error).toContain("initial values for two variables");
  expect(useStudioStore.getState().isSimulating).toBe(false);
});

it("withdraws precision completion after a Q-format change without discarding the plot", async () => {
  const finish = deferred("runPrecision");
  const pending = run("runPrecision");
  finish();
  await pending;
  const old = useStudioStore.getState().precResult;
  expect(complete()).toBe(true);
  useStudioStore.getState().setModelQFormat("Q16.16");
  expect(complete()).toBe(false);
  expect(useStudioStore.getState().precResult).toBe(old);
  useStudioStore.getState().setModelQFormat("Q8.8");
  expect(complete()).toBe(true);
});

it.each([false, true])("drops an obsolete precision profile response, rejected=%s", async (rejected) => {
  const finish = deferred("runPrecision");
  const pending = run("runPrecision");
  useStudioStore.getState().setModelQFormat("Q16.16");
  finish(rejected);
  await pending;
  expect(useStudioStore.getState().precResult).toBeNull();
  expect(useStudioStore.getState().error).toBeNull();
  expect(useStudioStore.getState().isSimulating).toBe(false);
  expect(complete()).toBe(false);
});

it("keeps frequency analysis current across unrelated Q-format changes", async () => {
  const finish = deferred("runFreqResponse");
  const pending = run("runFreqResponse");
  useStudioStore.getState().setModelQFormat("Q16.16");
  finish();
  await pending;
  expect(complete()).toBe(true);
});

it("does not use an older frequency plot to qualify a mismatched precision profile", async () => {
  let finish = deferred("runFreqResponse");
  let pending = run("runFreqResponse");
  finish();
  await pending;
  finish = deferred("runPrecision");
  pending = run("runPrecision");
  finish();
  await pending;
  useStudioStore.getState().setModelQFormat("Q16.16");
  expect(useStudioStore.getState().freqResult).not.toBeNull();
  expect(complete()).toBe(false);
  finish = deferred("runPrecision");
  pending = run("runPrecision");
  finish();
  await pending;
  expect(complete()).toBe(true);
});

it("records the same Q-format that the production client submits", async () => {
  useStudioStore.getState().setModelQFormat("Q16.16");
  const fetch = vi.fn<typeof globalThis.fetch>().mockResolvedValue(new Response(JSON.stringify(payload("runPrecision"))));
  vi.stubGlobal("fetch", fetch);
  await run("runPrecision");
  const requestBody = fetch.mock.calls[0]?.[1]?.body;
  if (typeof requestBody !== "string") throw new Error("expected a JSON request body");
  const body: unknown = JSON.parse(requestBody);
  expect(body).toMatchObject({ q_format: "Q16.16" });
  const key: unknown = JSON.parse(useStudioStore.getState().analysisExperimentKey ?? "null");
  expect(key).toMatchObject({ qFormat: "Q16.16" });
  expect(complete()).toBe(true);
});
