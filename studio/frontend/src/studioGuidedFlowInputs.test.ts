// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — What the guided workflow may and may not call done

import { describe, expect, it } from "vitest";
import { studioBundleContextKey } from "./studioBundleContext";

import type { SimulateResponse, SynthResult } from "./api/client";
import {
  studioExperimentKey,
  studioTrainingKey,
} from "./studioExperimentKey";
import { studioGuidedFlowInputs } from "./studioGuidedFlowInputs";
import type { StudioGuidedFlowSource } from "./studioGuidedFlowSource";
import { studioSimulationConfigInput } from "./studioSimulationConfigInput";

const TRAINING_CONFIG = {
  batch_size: 32,
  dataset: "synthetic",
  epochs: 4,
  hidden: [128],
  learn_beta: false,
  learn_threshold: true,
  lr: 0.001,
  surrogate: "atan_surrogate",
  timesteps: 25,
};

/**
 * Build the store fields the workflow reads, overridden field by field.
 *
 * @param overrides - The fields to change.
 * @returns The source.
 */
function source(overrides: Partial<StudioGuidedFlowSource> = {}): StudioGuidedFlowSource {
  return {
    bundleContexts: {}, candidates: [], graphPopulations: [], graphProjections: [],
    synthTarget: "ice40", projectRevision: null,
    latestSynthesisJobId: null, latestMultiTargetSynthesisJobId: null,
    verilogSrc: "", svSource: "",
    analysisExperimentKey: null,
    modelQFormat: "Q8.8",
    bifResult: null,
    charResult: null,
    compareResult: null,
    compileEvidenceBundle: null,
    compileTraceability: null,
    cosimResult: null,
    current: 10,
    dt: 0.1,
    duration: 100,
    equations: ["dv/dt = -(v - E_L) / tau_m + I / C"],
    evidenceBundle: null,
    fiResult: null,
    freqResult: null,
    frequencyHz: 10,
    heatmapResult: null,
    modelParams: { tau_m: 10 },
    multiTargetResult: null,
    nullclineResult: null,
    odeInit: { v: -65 },
    odeParams: { C: 1, E_L: -65, tau_m: 10 },
    precResult: null,
    projectEvidenceBundle: null,
    protocol: "constant",
    reset: "v = -65",
    result: null,
    resultExperimentKey: null,
    seed: null,
    selectedModelName: "SCLapicqueLIFNeuron",
    sensResult: null,
    stageFailure: null,
    sourceMode: "model",
    synthesisEvidenceBundle: null,
    synthResult: null,
    threshold: "v > -50",
    trainingConfig: TRAINING_CONFIG,
    trainingExperimentKey: null,
    trainingStatus: "idle",
    trial: "replay",
    ...overrides,
  };
}

const RTL_DIGEST = "a".repeat(64);

/** A compile whose RTL a parity report can be checked against. */
const COMPILED = {
  output: { rtl_sha256: RTL_DIGEST },
} as unknown as StudioGuidedFlowSource["compileTraceability"];

/** Identifiable receipt; these tests do not inspect artifact contents. */
const BUNDLE = { bundle_id: "seb_test", job_id: "sj_test" } as StudioGuidedFlowSource["evidenceBundle"];

/**
 * Build a co-simulation report over the given RTL.
 *
 * @param bitExact - What the report says about parity.
 * @param sourceSha256 - The RTL it was run against.
 * @returns The report.
 */
function cosim(
  bitExact: boolean,
  sourceSha256: string,
): StudioGuidedFlowSource["cosimResult"] {
  return {
    bit_exact: bitExact,
    rtl: { source_sha256: sourceSha256 },
  } as unknown as StudioGuidedFlowSource["cosimResult"];
}

/** A run, of which only the presence matters to the workflow. */
const RUN = {
  current_trace: [0, 1],
  dt: 0.1,
  n_steps: 2,
  spike_count: 0,
  spikes: [],
  states: { v: [-65, -64] },
  stats: { isi_cv: null, isi_histogram: null, isi_mean_ms: null, rate_hz: 0 },
  time: [0.1, 0.2],
} as unknown as SimulateResponse;

/**
 * Build a synthesis result that either succeeded or did not.
 *
 * @param success - What the run reports about itself.
 * @returns The result.
 */
function synth(success: boolean): SynthResult {
  return { success } as unknown as SynthResult;
}

/**
 * The inputs derived from a source, with nothing decided by the reader.
 *
 * @param state - The store fields.
 * @returns The derived inputs.
 */
function derive(state: StudioGuidedFlowSource) {
  return studioGuidedFlowInputs(state, {
    evidenceExportSatisfied: false,
    trainingSkipped: false,
  });
}

describe("simulation completion", () => {
  it("completes when the run on record is of the experiment on screen", () => {
    const state = source({ result: RUN });
    state.resultExperimentKey = studioExperimentKey(studioSimulationConfigInput(state));

    expect(derive(state).simulationComplete).toBe(true);
  });

  it("stops completing once a parameter changes under it", () => {
    // Editing params invalidates descendants: the trace is still shown, and it
    // is no longer evidence that this experiment has been run.
    const ran = source({ result: RUN });
    const key = studioExperimentKey(studioSimulationConfigInput(ran));
    const edited = source({ modelParams: { tau_m: 20 }, result: RUN, resultExperimentKey: key });

    expect(derive(edited).simulationComplete).toBe(false);
  });

  it("does not complete on a run that recorded no experiment at all", () => {
    expect(derive(source({ result: RUN })).simulationComplete).toBe(false);
  });
});

describe("analysis completion", () => {
  it("completes on any analysis of the experiment on screen", () => {
    const state = source({ fiResult: {} as never });
    state.analysisExperimentKey = studioExperimentKey(studioSimulationConfigInput(state));

    expect(derive(state).analysisComplete).toBe(true);
  });

  it("stops completing once the model changes under it", () => {
    const analysed = source({ fiResult: {} as never });
    const key = studioExperimentKey(studioSimulationConfigInput(analysed));
    const switched = source({
      analysisExperimentKey: key,
      fiResult: {} as never,
      selectedModelName: "SCIzhikevichNeuron",
    });

    expect(derive(switched).analysisComplete).toBe(false);
  });
});

describe("training completion", () => {
  it("does not complete on epochs alone while the run is still going", () => {
    // The defect this replaces: one epoch marked the step complete.
    const state = source({ trainingStatus: "running" });
    state.trainingExperimentKey = studioTrainingKey(TRAINING_CONFIG);

    expect(derive(state).trainingComplete).toBe(false);
  });

  it("does not complete on a run that failed after reporting epochs", () => {
    const state = source({ trainingStatus: "failed" });
    state.trainingExperimentKey = studioTrainingKey(TRAINING_CONFIG);

    expect(derive(state).trainingComplete).toBe(false);
  });

  it("does not complete on a run the reader stopped", () => {
    // `stopped` is terminal and deliberately not complete.
    const state = source({ trainingStatus: "stopped" });
    state.trainingExperimentKey = studioTrainingKey(TRAINING_CONFIG);

    expect(derive(state).trainingComplete).toBe(false);
  });

  it("completes on a run that finished under the configuration on screen", () => {
    const state = source({ trainingStatus: "completed" });
    state.trainingExperimentKey = studioTrainingKey(TRAINING_CONFIG);

    expect(derive(state).trainingComplete).toBe(true);
  });

  it("stops completing once the training configuration changes", () => {
    const state = source({
      trainingConfig: { ...TRAINING_CONFIG, dataset: "shd" },
      trainingExperimentKey: studioTrainingKey(TRAINING_CONFIG),
      trainingStatus: "completed",
    });

    expect(derive(state).trainingComplete).toBe(false);
  });

  it("is unaffected by a change to the simulation experiment", () => {
    // Training a network on a dataset does not stop being trained because the
    // reader changed the integration step on another panel.
    const state = source({
      dt: 0.05,
      trainingExperimentKey: studioTrainingKey(TRAINING_CONFIG),
      trainingStatus: "completed",
    });

    expect(derive(state).trainingComplete).toBe(true);
  });
});

describe("synthesis completion", () => {
  it("does not complete an ODE run on a synthesis object that reports failure", () => {
    // The defect this replaces: a non-null result completed the step.
    const state = source({ sourceMode: "ode", synthResult: synth(false) });

    expect(derive(state).synthesisComplete).toBe(false);
  });

  it("completes an ODE run on a synthesis that reports success", () => {
    const state = source({ sourceMode: "ode", synthResult: synth(true) });

    expect(derive(state).synthesisComplete).toBe(true);
  });

  it("does not complete an ODE run when every target of a multi-target run failed", () => {
    const state = source({
      multiTargetResult: { targets: { ecp5: synth(false), ice40: synth(false) } } as never,
      sourceMode: "ode",
    });

    expect(derive(state).synthesisComplete).toBe(false);
  });

  it("completes an ODE run when one target of a multi-target run succeeded", () => {
    const state = source({
      multiTargetResult: { targets: { ecp5: synth(true), ice40: synth(false) } } as never,
      sourceMode: "ode",
    });

    expect(derive(state).synthesisComplete).toBe(true);
  });

  it("requires a catalogue model's synthesis to reach silicon", () => {
    const state = source({ synthResult: synth(true) });

    expect(derive(state).synthesisComplete).toBe(false);
  });
});

describe("the rules this module carries unchanged", () => {
  it("counts a catalogue model as selected only once one is named", () => {
    expect(derive(source({ selectedModelName: "" })).modelSelected).toBe(false);
    expect(derive(source()).modelSelected).toBe(true);
  });

  it("counts an ODE workspace as designed once it has equations", () => {
    expect(derive(source({ equations: [], sourceMode: "ode" })).modelSelected).toBe(false);
    expect(derive(source({ sourceMode: "ode" })).modelSelected).toBe(true);
  });

  it("applies co-simulation to catalogue models only", () => {
    expect(derive(source({ sourceMode: "ode" })).cosimApplicable).toBe(false);
    expect(derive(source()).cosimApplicable).toBe(true);
  });

  it("completes compilation on a traceability record", () => {
    expect(derive(source()).compileComplete).toBe(false);
    expect(derive(source({ compileTraceability: COMPILED })).compileComplete).toBe(true);
  });

  it("refuses parity for RTL that has since been recompiled", () => {
    // A parity report for other RTL says nothing about what would be
    // synthesised now.
    const stale = source({
      compileTraceability: COMPILED,
      cosimResult: cosim(true, "b".repeat(64)),
    });

    expect(derive(stale).cosimComplete).toBe(false);
  });

  it("completes parity when the report and the compile agree on the RTL", () => {
    const matching = source({
      compileTraceability: COMPILED,
      cosimResult: cosim(true, RTL_DIGEST),
    });

    expect(derive(matching).cosimComplete).toBe(true);
  });

  it("refuses parity from a report that is not bit-exact", () => {
    const inexact = source({
      compileTraceability: COMPILED,
      cosimResult: cosim(false, RTL_DIGEST),
    });

    expect(derive(inexact).cosimComplete).toBe(false);
  });

  it("refuses parity when nothing has been compiled to compare against", () => {
    expect(derive(source({ cosimResult: cosim(true, RTL_DIGEST) })).cosimComplete).toBe(false);
  });

  it.each([
    ["an admin bundle", "evidenceBundle"],
    ["a project bundle", "projectEvidenceBundle"],
    ["a compile bundle", "compileEvidenceBundle"],
    ["a synthesis bundle", "synthesisEvidenceBundle"],
  ] as const)("rejects an unattested %s", (_label, field) => {
    expect(derive(source({ [field]: BUNDLE })).evidenceExported).toBe(false);
  });

  it.each(["project", "compile", "synthesis"] as const)("binds %s completion to its context and receipt", (surface) => {
    const state = source({ [`${surface}EvidenceBundle`]: BUNDLE });
    state.bundleContexts[surface] = {
      key: studioBundleContextKey(surface, state), bundleId: "seb_test", jobId: "sj_test",
    };
    expect(derive(state).evidenceExported).toBe(true);
    expect(derive({ ...state, sourceMode: "ode" }).evidenceExported).toBe(false);
    state.bundleContexts[surface].jobId = "sj_other";
    expect(derive(state).evidenceExported).toBe(false);
  });

  it("completes the export step on the session's own cart", () => {
    const withCart = studioGuidedFlowInputs(source(), {
      evidenceExportSatisfied: true,
      trainingSkipped: false,
    });

    expect(withCart.evidenceExported).toBe(true);
  });

  it("does not complete the export step when nothing has been exported", () => {
    expect(derive(source()).evidenceExported).toBe(false);
  });

  it("carries the reader's decision to skip training through unchanged", () => {
    const skipped = studioGuidedFlowInputs(source(), {
      evidenceExportSatisfied: false,
      trainingSkipped: true,
    });

    expect(skipped.trainingSkipped).toBe(true);
  });
});

describe("stage failures", () => {
  it("shows a recorded failure only while its experiment is on screen", () => {
    const state = source();
    const key = studioExperimentKey(studioSimulationConfigInput(state));
    state.stageFailure = { stage: "simulate", message: "Simulation request failed", experimentKey: key };

    expect(derive(state).failures).toEqual({ simulate: "Simulation request failed" });
    expect(derive({ ...state, duration: 150 }).failures).toEqual({});
  });

  it("never shows a failure recorded under no experiment", () => {
    const state = source();
    state.stageFailure = { stage: "simulate", message: "invalid current", experimentKey: null };

    expect(derive(state).failures).toEqual({});
  });

  it.each(["failed", "interrupted"])("reads a %s training run of the configuration on screen as failed", (status) => {
    const state = source({ trainingStatus: status });
    state.trainingExperimentKey = studioTrainingKey(TRAINING_CONFIG);

    expect(derive(state).failures).toEqual({ train: `Training run ${status}` });
    expect(derive({ ...state, trainingExperimentKey: null }).failures).toEqual({});
  });

  it("does not read a stopped training run as failed", () => {
    const state = source({ trainingStatus: "stopped" });
    state.trainingExperimentKey = studioTrainingKey(TRAINING_CONFIG);

    expect(derive(state).failures).toEqual({});
  });

  it("reads a co-simulation of the current RTL that is not bit-exact as failed", () => {
    const state = source({ compileTraceability: COMPILED, cosimResult: cosim(false, RTL_DIGEST) });

    expect(derive(state).failures).toEqual({
      cosim: "RTL co-simulation of the compiled design is not bit-exact",
    });
    expect(derive({ ...state, cosimResult: cosim(false, "b".repeat(64)) }).failures).toEqual({});
    expect(derive({ ...state, compileTraceability: null }).failures).toEqual({});
  });

  it("prefers the recorded message over one read from a result for the same stage", () => {
    const state = source({ compileTraceability: COMPILED, cosimResult: cosim(false, RTL_DIGEST) });
    const key = studioExperimentKey(studioSimulationConfigInput(state));
    state.stageFailure = { stage: "cosim", message: "Parity mismatch at step 12", experimentKey: key };

    expect(derive(state).failures).toEqual({ cosim: "Parity mismatch at step 12" });
  });

  it("reads a catalogue synthesis whose terminal reported failure as failed", () => {
    const failed = {
      silicon_terminal: { success: false, place_and_route: { error: "nextpnr: placement failed" } },
    } as unknown as SynthResult;
    const unexplained = { silicon_terminal: { success: false, place_and_route: null } } as unknown as SynthResult;

    expect(derive(source({ synthResult: failed })).failures)
      .toEqual({ synthesise: "nextpnr: placement failed" });
    expect(derive(source({ synthResult: unexplained })).failures)
      .toEqual({ synthesise: "Synthesis/PnR terminal did not complete" });
    expect(derive(source({ synthResult: synth(false) })).failures).toEqual({});
  });

  it("reads an ODE synthesis that reported failure as failed", () => {
    const withError = { success: false, error: "yosys: syntax error" } as unknown as SynthResult;

    expect(derive(source({ sourceMode: "ode", synthResult: withError })).failures)
      .toEqual({ synthesise: "yosys: syntax error" });
    expect(derive(source({ sourceMode: "ode", synthResult: synth(false) })).failures)
      .toEqual({ synthesise: "Synthesis did not complete" });
    expect(derive(source({ sourceMode: "ode", synthResult: synth(true) })).failures).toEqual({});
  });

  it("reads a multi-target run in which no target succeeded as failed", () => {
    const multi = (success: boolean) => ({
      supported: ["ice40"],
      target_provenance_matrix: {},
      targets: { ice40: synth(success) },
    }) as unknown as StudioGuidedFlowSource["multiTargetResult"];

    expect(derive(source({ sourceMode: "ode", multiTargetResult: multi(false) })).failures)
      .toEqual({ synthesise: "No synthesis target succeeded" });
    expect(derive(source({ sourceMode: "ode", multiTargetResult: multi(true) })).failures).toEqual({});
    expect(derive(source({
      sourceMode: "ode",
      multiTargetResult: { ...multi(false), targets: {} } as StudioGuidedFlowSource["multiTargetResult"],
    })).failures).toEqual({});
  });
});
