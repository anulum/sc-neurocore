// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio guided default-flow state machine tests
import { describe, expect, it } from "vitest";

import {
  COSIM_NOT_APPLICABLE_REASON,
  TRAINING_SKIPPED_REASON,
  computeGuidedFlowState,
  type GuidedFlowCapabilityMap,
  type GuidedFlowInputs,
  type GuidedFlowStep,
  type GuidedFlowStepKey,
} from "./guidedFlowState";

/**
 * Build a set of accomplished-evidence facts, overridden field by field.
 *
 * @param overrides - The fields to change.
 * @returns The inputs.
 */
function inputs(overrides: Partial<GuidedFlowInputs> = {}): GuidedFlowInputs {
  return {
    modelSelected: false,
    simulationComplete: false,
    analysisComplete: false,
    trainingComplete: false,
    trainingSkipped: false,
    compileComplete: false,
    cosimApplicable: false,
    cosimComplete: false,
    synthesisComplete: false,
    evidenceExported: false,
    ...overrides,
  };
}

/** Inputs for a run that has been designed, simulated and analysed. */
const ANALYSED = { modelSelected: true, simulationComplete: true, analysisComplete: true };

/**
 * Every capability available except the ones named.
 *
 * @param unavailable - The steps the deployment cannot perform.
 * @returns The capability map.
 */
function capabilitiesWithout(...unavailable: GuidedFlowStepKey[]): GuidedFlowCapabilityMap {
  const all: GuidedFlowCapabilityMap = {
    design: true,
    simulate: true,
    analyse: true,
    train: true,
    compile: true,
    cosim: true,
    synthesise: true,
    export: true,
  };
  for (const key of unavailable) {
    all[key] = false;
  }
  return all;
}

/**
 * Read one step out of a computed flow.
 *
 * @param state - The flow.
 * @param key - The step to look up.
 * @returns The step.
 */
function stepOf(state: ReturnType<typeof computeGuidedFlowState>, key: GuidedFlowStepKey): GuidedFlowStep {
  const step = state.steps.find((candidate) => candidate.key === key);
  if (!step) {
    throw new Error(`missing guided-flow step ${key}`);
  }
  return step;
}

describe("computeGuidedFlowState", () => {
  it("starts at design with downstream steps blocked on prerequisites", () => {
    const state = computeGuidedFlowState(inputs());

    expect(state.currentStepKey).toBe("design");
    expect(state.completedCount).toBe(0);
    expect(state.skippedCount).toBe(0);
    expect(state.totalCount).toBe(7);
    expect(stepOf(state, "design").status).toBe("current");
    expect(stepOf(state, "simulate")).toMatchObject({ status: "blocked", reason: "Requires Design" });
  });

  it("advances the current step as evidence accumulates", () => {
    const state = computeGuidedFlowState(inputs({ modelSelected: true, simulationComplete: true }));

    expect(stepOf(state, "design").status).toBe("completed");
    expect(stepOf(state, "simulate").status).toBe("completed");
    expect(stepOf(state, "analyse").status).toBe("current");
    expect(state.completedCount).toBe(2);
  });

  it("treats train as the current optional step while compile stays available", () => {
    const state = computeGuidedFlowState(inputs(ANALYSED));

    expect(stepOf(state, "train")).toMatchObject({ status: "current", optional: true });
    expect(stepOf(state, "compile").status).toBe("available");
  });

  it("shows a skipped training step as skipped, not as training evidence", () => {
    const state = computeGuidedFlowState(inputs({ ...ANALYSED, trainingSkipped: true }));

    expect(stepOf(state, "train")).toMatchObject({ status: "skipped", reason: TRAINING_SKIPPED_REASON });
    expect(stepOf(state, "compile").status).toBe("current");
    expect(state.completedCount).toBe(3);
    expect(state.skippedCount).toBe(1);
  });

  it("prefers completed training evidence over an earlier decision to skip", () => {
    const state = computeGuidedFlowState(
      inputs({ ...ANALYSED, trainingComplete: true, trainingSkipped: true }),
    );

    expect(stepOf(state, "train").status).toBe("completed");
    expect(state.skippedCount).toBe(0);
  });

  it("marks a step the deployment cannot perform unsupported, with the registry's message", () => {
    const state = computeGuidedFlowState(
      inputs({ modelSelected: true }),
      capabilitiesWithout("simulate"),
      { simulate: "Simulation backend is offline" },
    );

    expect(stepOf(state, "simulate")).toMatchObject({
      status: "unsupported",
      reason: "Simulation backend is offline",
    });
    expect(stepOf(state, "analyse")).toMatchObject({ status: "blocked", reason: "Requires Simulate" });
    expect(state.currentStepKey).toBeNull();
  });

  it("names an unsupported step generically when the registry gives no message", () => {
    const state = computeGuidedFlowState(inputs(), capabilitiesWithout("design"));

    expect(stepOf(state, "design")).toMatchObject({
      status: "unsupported",
      reason: "Design capability is unavailable",
    });
  });

  it("keeps completed evidence completed even when the capability is now unavailable", () => {
    const state = computeGuidedFlowState(
      inputs({ modelSelected: true, simulationComplete: true }),
      capabilitiesWithout("simulate"),
    );

    expect(stepOf(state, "simulate").status).toBe("completed");
    expect(state.currentStepKey).toBe("analyse");
  });

  it("requires model RTL co-simulation parity between compile and synthesis", () => {
    const state = computeGuidedFlowState(inputs({
      ...ANALYSED,
      compileComplete: true,
      cosimApplicable: true,
      trainingSkipped: true,
    }));

    expect(state.totalCount).toBe(8);
    expect(stepOf(state, "cosim").status).toBe("current");
    expect(stepOf(state, "synthesise")).toMatchObject({
      status: "blocked",
      reason: "Requires Co-sim parity",
    });
  });

  it("shows co-simulation as not applicable to a hand-written equation and leaves it out of the count", () => {
    const state = computeGuidedFlowState(inputs({
      ...ANALYSED,
      compileComplete: true,
      trainingSkipped: true,
    }));

    expect(stepOf(state, "cosim")).toMatchObject({
      status: "not_applicable",
      reason: COSIM_NOT_APPLICABLE_REASON,
    });
    expect(state.totalCount).toBe(7);
    expect(state.currentStepKey).toBe("synthesise");
  });

  it("offers a failed step as the one to retry, with its reason", () => {
    const state = computeGuidedFlowState(inputs({
      ...ANALYSED,
      trainingSkipped: true,
      failures: { compile: "RTL emission failed" },
    }));

    expect(stepOf(state, "compile")).toMatchObject({ status: "failed", reason: "RTL emission failed" });
    expect(state.currentStepKey).toBe("compile");
    expect(state.completedCount).toBe(3);
  });

  it("does not count a failed step as done even when a later step is reachable", () => {
    const state = computeGuidedFlowState(inputs({
      ...ANALYSED,
      failures: { train: "Training run failed", compile: "RTL emission failed" },
    }));

    expect(stepOf(state, "train").status).toBe("failed");
    expect(state.currentStepKey).toBe("train");
    expect(stepOf(state, "compile")).toMatchObject({ status: "failed", reason: "RTL emission failed" });
    expect(state.completedCount).toBe(3);
  });

  it("lets a reachable step after a failed optional step stay available", () => {
    const state = computeGuidedFlowState(inputs({
      ...ANALYSED,
      failures: { train: "Training run interrupted" },
    }));

    expect(state.currentStepKey).toBe("train");
    expect(stepOf(state, "compile").status).toBe("available");
  });

  it("reports a step waiting on its predecessor as blocked rather than failed", () => {
    const state = computeGuidedFlowState(inputs({
      modelSelected: true,
      failures: { compile: "RTL emission failed" },
    }));

    expect(stepOf(state, "compile")).toMatchObject({ status: "blocked", reason: "Requires Analyse" });
    expect(state.currentStepKey).toBe("simulate");
  });

  it("lets evidence for the current inputs win over a recorded failure", () => {
    const state = computeGuidedFlowState(inputs({
      modelSelected: true,
      simulationComplete: true,
      failures: { simulate: "Simulation request failed" },
    }));

    expect(stepOf(state, "simulate").status).toBe("completed");
  });

  it("reports a fully completed flow with no step to act on", () => {
    const state = computeGuidedFlowState(inputs({
      ...ANALYSED,
      trainingComplete: true,
      compileComplete: true,
      synthesisComplete: true,
      evidenceExported: true,
    }));

    expect(state.completedCount).toBe(7);
    expect(state.currentStepKey).toBeNull();
    expect(state.steps.filter((step) => step.status !== "not_applicable")
      .every((step) => step.status === "completed")).toBe(true);
  });
});
