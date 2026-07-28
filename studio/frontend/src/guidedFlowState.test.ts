// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio guided default-flow state machine tests
import { describe, expect, it } from "vitest";

import {
  computeGuidedFlowState,
  type GuidedFlowCapabilityMap,
  type GuidedFlowInputs,
  type GuidedFlowStepKey,
  type GuidedFlowStepStatus,
} from "./guidedFlowState";

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

function statusOf(
  state: ReturnType<typeof computeGuidedFlowState>,
  key: GuidedFlowStepKey,
): GuidedFlowStepStatus {
  const step = state.steps.find((candidate) => candidate.key === key);
  if (!step) {
    throw new Error(`missing guided-flow step ${key}`);
  }
  return step.status;
}

describe("computeGuidedFlowState", () => {
  it("starts at design with downstream steps blocked on prerequisites", () => {
    const state = computeGuidedFlowState(inputs());

    expect(state.currentStepKey).toBe("design");
    expect(state.completedCount).toBe(0);
    expect(state.totalCount).toBe(7);
    expect(statusOf(state, "design")).toBe("current");
    expect(statusOf(state, "simulate")).toBe("blocked");
    const simulate = state.steps.find((step) => step.key === "simulate");
    expect(simulate?.blockedReason).toBe("Requires Design");
  });

  it("advances the current step as evidence accumulates", () => {
    const state = computeGuidedFlowState(
      inputs({ modelSelected: true, simulationComplete: true }),
    );

    expect(statusOf(state, "design")).toBe("completed");
    expect(statusOf(state, "simulate")).toBe("completed");
    expect(statusOf(state, "analyse")).toBe("current");
    expect(state.completedCount).toBe(2);
  });

  it("treats train as the current optional step while compile stays available", () => {
    const state = computeGuidedFlowState(
      inputs({ modelSelected: true, simulationComplete: true, analysisComplete: true }),
    );

    expect(statusOf(state, "train")).toBe("current");
    expect(state.steps.find((step) => step.key === "train")?.optional).toBe(true);
    expect(statusOf(state, "compile")).toBe("available");
  });

  it("counts a skipped training step as complete and moves to compile", () => {
    const state = computeGuidedFlowState(
      inputs({
        modelSelected: true,
        simulationComplete: true,
        analysisComplete: true,
        trainingSkipped: true,
      }),
    );

    expect(statusOf(state, "train")).toBe("completed");
    expect(statusOf(state, "compile")).toBe("current");
  });

  it("blocks a step whose capability is unavailable with a concrete reason", () => {
    const capabilities: GuidedFlowCapabilityMap = {
      design: true,
      simulate: false,
      analyse: true,
      train: true,
      compile: true,
      cosim: true,
      synthesise: true,
      export: true,
    };
    const state = computeGuidedFlowState(inputs({ modelSelected: true }), capabilities);

    const simulate = state.steps.find((step) => step.key === "simulate");
    expect(simulate?.status).toBe("blocked");
    expect(simulate?.blockedReason).toBe("Simulate capability is unavailable");
    // Design is done, simulate is capability-blocked, so nothing is current.
    expect(state.currentStepKey).toBeNull();
  });

  it("requires model RTL co-simulation parity between compile and synthesis", () => {
    const state = computeGuidedFlowState(inputs({
      analysisComplete: true,
      compileComplete: true,
      cosimApplicable: true,
      modelSelected: true,
      simulationComplete: true,
      trainingSkipped: true,
    }));

    expect(state.totalCount).toBe(8);
    expect(statusOf(state, "cosim")).toBe("current");
    expect(statusOf(state, "synthesise")).toBe("blocked");
    expect(state.steps.find((step) => step.key === "synthesise")?.blockedReason)
      .toBe("Requires Co-sim parity");
  });

  it("reports a fully completed flow with no current step", () => {
    const state = computeGuidedFlowState(
      inputs({
        modelSelected: true,
        simulationComplete: true,
        analysisComplete: true,
        trainingComplete: true,
        compileComplete: true,
        synthesisComplete: true,
        evidenceExported: true,
      }),
    );

    expect(state.completedCount).toBe(7);
    expect(state.currentStepKey).toBeNull();
    expect(state.steps.every((step) => step.status === "completed")).toBe(true);
  });
});
