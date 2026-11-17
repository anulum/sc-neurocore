// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio analysis runs: which kind each starts, several models, the network and an imported trace

import { createStore } from "zustand/vanilla";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { fetchMultiSimulate, importTrace, simulateNetwork } from "../api/client";
import type { ImportedTrace, NetworkResult, SimulateResponse } from "../api/types";
import { runStoreHeavyAnalysis } from "./studioHeavyAnalysis";
import { studioStageSet } from "./studioStageFailure";
import { studioInitialData } from "./studioInitialState";
import { createStudioEquationIrActions } from "./studioEquationIrActions";
import { createStudioStoreActions } from "./studioStoreActions";
import type { StudioState } from "./studioTypes";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return { ...actual, fetchMultiSimulate: vi.fn(), importTrace: vi.fn(), simulateNetwork: vi.fn() };
});
vi.mock("./studioHeavyAnalysis", async () => {
  const actual = await vi.importActual<typeof import("./studioHeavyAnalysis")>("./studioHeavyAnalysis");
  return { ...actual, runStoreHeavyAnalysis: vi.fn(() => Promise.resolve()) };
});

vi.mock("./studioStageFailure", async () => {
  const actual = await vi.importActual<typeof import("./studioStageFailure")>("./studioStageFailure");
  return { ...actual, studioStageSet: vi.fn() };
});

// The stage setter is recorded but does its real work, so every other action behaves as in the application.
beforeEach(async () => {
  const actual = await vi.importActual<typeof import("./studioStageFailure")>("./studioStageFailure");
  vi.mocked(studioStageSet).mockImplementation(actual.studioStageSet);
});

afterEach(() => {
  vi.resetAllMocks();
});

/** A store with every real action over the initial data. */
function runStore(overrides: Partial<StudioState> = {}) {
  const store = createStore<StudioState>((set, get) => ({
    ...studioInitialData, ...createStudioStoreActions(set, get), ...createStudioEquationIrActions(set, get),
  }));
  store.setState(overrides);
  return store;
}

/** A promise the test settles. */
function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((ok) => { resolve = ok; });
  return { promise, resolve };
}

describe("the heavy analyses", () => {
  it.each<[string, (s: StudioState) => Promise<void>]>([
    ["fi_curve", (s) => s.runFICurve()],
    ["bifurcation", (s) => s.runBifurcation()],
    ["sensitivity", (s) => s.runSensitivity()],
    ["heatmap", (s) => s.runHeatmap()],
  ])("start the %s analysis as that kind, through the analysis stage", async (kind, run) => {
    const store = runStore();
    await run(store.getState());
    expect(runStoreHeavyAnalysis).toHaveBeenCalledOnce();
    expect(vi.mocked(runStoreHeavyAnalysis).mock.calls[0]?.[0]).toBe(kind);
    expect(studioStageSet).toHaveBeenCalledOnce();
    expect(vi.mocked(studioStageSet).mock.calls[0]?.[0]).toBe("analyse");
    // The analysis receives the very setter that stage made.
    expect(vi.mocked(runStoreHeavyAnalysis).mock.calls[0]?.[2]).toBe(vi.mocked(studioStageSet).mock.results[0]?.value);
  });
});

describe("several models side by side", () => {
  it("runs at most four of the named models with the run's stimulus, and shows the results", async () => {
    const store = runStore({ duration: 300, current: 6, protocol: "pulse" });
    const results = [{ model: "LIF" }] as unknown as SimulateResponse[];
    const answered = deferred<SimulateResponse[]>();
    vi.mocked(fetchMultiSimulate).mockReturnValue(answered.promise);
    const running = store.getState().runMultiSimulate(["A", "B", "C", "D", "E"]);
    expect(store.getState()).toMatchObject({ isSimulating: true, activeTab: "multi", error: null });
    answered.resolve(results);
    await running;
    expect(vi.mocked(fetchMultiSimulate).mock.calls[0]?.[0]).toStrictEqual(["A", "B", "C", "D"].map((name) => ({
      name, params: null, dt: null, duration: 300, current: 6, protocol: "pulse",
    })));
    expect(store.getState()).toMatchObject({ isSimulating: false, multiResults: results });
  });

  it("starts nothing while another run is busy, and reports a refusal", async () => {
    const busy = runStore({ isSimulating: true });
    await busy.getState().runMultiSimulate(["A"]);
    expect(fetchMultiSimulate).not.toHaveBeenCalled();
    const store = runStore();
    vi.mocked(fetchMultiSimulate).mockRejectedValue(new Error("model library offline"));
    await store.getState().runMultiSimulate(["A"]);
    expect(store.getState()).toMatchObject({ isSimulating: false, error: "model library offline" });
  });
});

describe("the network", () => {
  it("writes one network parameter and keeps the others", () => {
    const store = runStore();
    store.getState().setNetworkParam("n_exc", 120);
    expect(store.getState().networkParams).toStrictEqual({ ...studioInitialData.networkParams, n_exc: 120 });
  });

  it("runs the network with its parameters and the run's duration, and shows the result", async () => {
    const store = runStore({ duration: 500 });
    const result = { rates: [] } as unknown as NetworkResult;
    vi.mocked(simulateNetwork).mockResolvedValue(result);
    await store.getState().runNetwork();
    expect(vi.mocked(simulateNetwork).mock.calls[0]?.[0]).toStrictEqual({ ...studioInitialData.networkParams, duration: 500 });
    expect(store.getState()).toMatchObject({ isSimulating: false, networkResult: result, activeTab: "network" });
  });

  it("starts nothing while another run is busy, and reports a refusal", async () => {
    const busy = runStore({ isSimulating: true });
    await busy.getState().runNetwork();
    expect(simulateNetwork).not.toHaveBeenCalled();
    const store = runStore();
    vi.mocked(simulateNetwork).mockRejectedValue(new Error(""));
    await store.getState().runNetwork();
    expect(store.getState().isSimulating).toBe(false);
    expect(store.getState().error).toBe("Error");
  });
});

describe("an imported trace", () => {
  const CSV = Array.from({ length: 12 }, (_, i) => `${String(i * 0.1)},${String(-65 + i)}`).join("\n");

  it("sends the trace's voltages with the run's time step and shows the imported trace", async () => {
    const store = runStore({ dt: 0.025 });
    const imported = { voltage: [] } as unknown as ImportedTrace;
    vi.mocked(importTrace).mockResolvedValue(imported);
    await store.getState().importCSV(CSV);
    expect(vi.mocked(importTrace).mock.calls[0]?.[0]).toStrictEqual({
      voltage: Array.from({ length: 12 }, (_, i) => -65 + i), dt: 0.025,
    });
    expect(store.getState()).toMatchObject({ importedTrace: imported, activeTab: "trace" });
  });

  it("refuses a trace too short to analyse before sending it, and names what the server refuses", async () => {
    const store = runStore();
    await store.getState().importCSV("1\n2\n3");
    expect(store.getState().error).toBe("Need at least 10 data points");
    expect(importTrace).not.toHaveBeenCalled();
    vi.mocked(importTrace).mockRejectedValue("unreadable trace");
    await store.getState().importCSV(CSV);
    expect(store.getState().error).toBe("unreadable trace");
  });
});
