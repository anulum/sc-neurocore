// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — A late response cannot land on the current experiment

/**
 * A run started before the reader changed the model can still be in flight
 * when they change it. These cases hold the store to dropping that response
 * rather than presenting another experiment's trace as this one's.
 */

import { afterEach, describe, expect, it, vi } from "vitest";

import { studioExperimentKey } from "../studioExperimentKey";
import { studioGuidedFlowInputs } from "../studioGuidedFlowInputs";
import { studioSimulationConfigInput } from "../studioSimulationConfigInput";
import { useStudioStore } from "./studio";

const initialState = useStudioStore.getState();

const RUN_BODY = JSON.stringify({
  current_trace: [0, 1],
  dt: 0.1,
  n_steps: 2,
  spike_count: 0,
  spikes: [],
  states: { v: [-65, -64] },
  stats: { isi_cv: null, isi_histogram: null, isi_mean_ms: null, rate_hz: 0 },
  time: [0.1, 0.2],
});

/** Read simulation completion through the same projection used by App. */
function simulationComplete(): boolean {
  return studioGuidedFlowInputs(useStudioStore.getState(), {
    evidenceExportSatisfied: false,
    trainingSkipped: false,
  }).simulationComplete;
}

afterEach(() => {
  useStudioStore.setState(initialState, true);
  vi.unstubAllGlobals();
});

/**
 * Stub `fetch` with a response the case releases when it chooses.
 *
 * @returns The release function, which resolves the pending request.
 */
function deferredRun(): () => void {
  let release = (): void => {
    throw new Error("the request was never made");
  };
  vi.stubGlobal(
    "fetch",
    vi.fn<typeof globalThis.fetch>(
      () =>
        new Promise<Response>((resolve) => {
          release = () => {
            resolve(new Response(RUN_BODY, { headers: { "Content-Type": "application/json" } }));
          };
        }),
    ),
  );
  return () => {
    release();
  };
}

describe("a simulation response that arrives late", () => {
  it("does not land on an experiment the reader has since changed", async () => {
    const release = deferredRun();
    useStudioStore.setState({ selectedModelName: "SCLapicqueLIFNeuron", sourceMode: "model" });

    const run = useStudioStore.getState().runSimulation();
    useStudioStore.setState({ selectedModelName: "SCIzhikevichNeuron" });
    release();
    await run;

    const after = useStudioStore.getState();
    expect(after.result).toBeNull();
    expect(after.resultExperimentKey).toBeNull();
    // The run it belonged to really has ended; leaving the panel spinning
    // would be a second lie on top of the first.
    expect(after.isSimulating).toBe(false);
  });

  it("lands, and records its experiment, when nothing changed under it", async () => {
    const release = deferredRun();
    useStudioStore.setState({ selectedModelName: "SCLapicqueLIFNeuron", sourceMode: "model" });
    const expected = studioExperimentKey(
      studioSimulationConfigInput(useStudioStore.getState()),
    );

    const run = useStudioStore.getState().runSimulation();
    release();
    await run;

    const after = useStudioStore.getState();
    expect(after.result).not.toBeNull();
    expect(after.resultExperimentKey).toBe(expected);
    expect(after.isSimulating).toBe(false);
  });
});

describe("a simulation rerun replaces completion evidence", () => {
  it.each(["model", "ode"] as const)("%s retains traces without completing a pending or failed rerun", async (sourceMode) => {
    useStudioStore.setState({ sourceMode, selectedModelName: "SCLapicqueLIFNeuron" });
    const first = deferredRun();
    const initialRun = useStudioStore.getState().runSimulation();
    first();
    await initialRun;
    expect(simulationComplete()).toBe(true);
    const previousTrace = useStudioStore.getState().result;

    let fail = (_reason: Error): void => { throw new Error("request not submitted"); };
    vi.stubGlobal("fetch", vi.fn<typeof globalThis.fetch>(() => new Promise<Response>(
      (_resolve, reject) => { fail = reject; },
    )));
    const rerun = useStudioStore.getState().runSimulation();
    expect(useStudioStore.getState().isSimulating).toBe(true);
    expect(useStudioStore.getState().result).toBe(previousTrace);
    const pendingComplete = simulationComplete();
    fail(new Error("simulation unavailable"));
    await rerun;
    expect(useStudioStore.getState().isSimulating).toBe(false);
    expect(useStudioStore.getState().error).toContain("simulation unavailable");
    expect(useStudioStore.getState().result).toBe(previousTrace);
    expect(pendingComplete).toBe(false);
    expect(simulationComplete()).toBe(false);

    const release = deferredRun();
    const recovery = useStudioStore.getState().runSimulation();
    release();
    await recovery;
    expect(simulationComplete()).toBe(true);
    expect(useStudioStore.getState().error).toBeNull();
  });
});
