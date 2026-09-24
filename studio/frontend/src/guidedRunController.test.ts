// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio guided run controller tests
import { describe, expect, it } from "vitest";

import {
  buildGuidedRunController,
  type GuidedRunActionKey,
  type GuidedRunActions,
} from "./guidedRunController";
import {
  computeGuidedFlowState,
  type GuidedFlowCapabilityMap,
  type GuidedFlowInputs,
} from "./guidedFlowState";

const allCapabilities: GuidedFlowCapabilityMap = {
  analyse: true,
  compile: true,
  cosim: true,
  design: true,
  export: true,
  simulate: true,
  synthesise: true,
  train: true,
};

/**
 * Compute a flow from overridden inputs.
 *
 * @param overrides - The evidence facts to change.
 * @param capabilities - Which steps the deployment can perform.
 * @returns The flow.
 */
function flow(overrides: Partial<GuidedFlowInputs> = {}, capabilities = allCapabilities) {
  return computeGuidedFlowState(
    {
      analysisComplete: false,
      compileComplete: false,
      cosimApplicable: false,
      cosimComplete: false,
      evidenceExported: false,
      modelSelected: true,
      simulationComplete: false,
      synthesisComplete: false,
      trainingComplete: false,
      trainingSkipped: false,
      ...overrides,
    },
    capabilities,
  );
}

/**
 * Build actions that record which of them ran.
 *
 * @param calls - The list each action appends its own key to.
 * @returns The actions.
 */
function actions(calls: GuidedRunActionKey[] = []): GuidedRunActions {
  return {
    exportEvidence: async () => {
      calls.push("export-evidence");
    },
    runAnalysis: async () => {
      calls.push("run-analysis");
    },
    runCompile: async () => {
      calls.push("run-compile");
    },
    runCosim: async () => {
      calls.push("run-cosim");
    },
    runSimulation: async () => {
      calls.push("run-simulation");
    },
    runSynthesis: async () => {
      calls.push("run-synthesis");
    },
    skipTraining: async () => {
      calls.push("skip-training");
    },
  };
}

describe("buildGuidedRunController", () => {
  it("maps the deterministic operator path to the existing Studio actions", async () => {
    const calls: GuidedRunActionKey[] = [];
    const runActions = actions(calls);

    await buildGuidedRunController({
      exportReady: false,
      flow: flow(),
      sourceMode: "ode",
    }, runActions).runNextStep();
    await buildGuidedRunController({
      exportReady: false,
      flow: flow({ simulationComplete: true }),
      sourceMode: "ode",
    }, runActions).runNextStep();
    await buildGuidedRunController({
      exportReady: false,
      flow: flow({ analysisComplete: true, simulationComplete: true }),
      sourceMode: "ode",
    }, runActions).runNextStep();
    await buildGuidedRunController({
      exportReady: false,
      flow: flow({ analysisComplete: true, simulationComplete: true, trainingSkipped: true }),
      sourceMode: "ode",
    }, runActions).runNextStep();
    await buildGuidedRunController({
      exportReady: false,
      flow: flow({
        analysisComplete: true,
        compileComplete: true,
        simulationComplete: true,
        trainingSkipped: true,
      }),
      sourceMode: "ode",
    }, runActions).runNextStep();
    await buildGuidedRunController({
      exportReady: true,
      flow: flow({
        analysisComplete: true,
        compileComplete: true,
        simulationComplete: true,
        synthesisComplete: true,
        trainingSkipped: true,
      }),
      sourceMode: "ode",
    }, runActions).runNextStep();

    expect(calls).toEqual([
      "run-simulation",
      "run-analysis",
      "skip-training",
      "run-compile",
      "run-synthesis",
      "export-evidence",
    ]);
  });

  it("blocks synthesis with the capability registry message when the tool lane is unavailable", () => {
    const controller = buildGuidedRunController({
      capabilityMessages: { synthesise: "Synthesis tools are unavailable." },
      exportReady: false,
      flow: flow(
        {
          analysisComplete: true,
          compileComplete: true,
          simulationComplete: true,
          trainingSkipped: true,
        },
        { ...allCapabilities, synthesise: false },
      ),
      sourceMode: "ode",
    }, actions());

    expect(controller.nextActionKey).toBe("blocked");
    expect(controller.nextActionLabel).toBe("Resolve blocker");
    expect(controller.blockerReason).toBe("Synthesis tools are unavailable.");
    expect(controller.exportReady).toBe(false);
  });

  it("routes catalogue model mode through the same compile action", async () => {
    const calls: GuidedRunActionKey[] = [];
    const controller = buildGuidedRunController({
      exportReady: false,
      flow: flow({ analysisComplete: true, simulationComplete: true, trainingSkipped: true }),
      compileConfigured: true,
      sourceMode: "model",
    }, actions(calls));

    expect(controller.nextActionKey).toBe("run-compile");
    expect(controller.blockerReason).toBeNull();
    await expect(controller.runNextStep()).resolves.toEqual({ ok: true });
    expect(calls).toEqual(["run-compile"]);
  });

  it("supports the catalogue model path through simulation, analysis, and compile", async () => {
    const calls: GuidedRunActionKey[] = [];
    const runActions = actions(calls);

    await buildGuidedRunController({
      exportReady: false,
      flow: flow({ modelSelected: true }),
      sourceMode: "model",
    }, runActions).runNextStep();
    await buildGuidedRunController({
      exportReady: false,
      flow: flow({ modelSelected: true, simulationComplete: true }),
      sourceMode: "model",
    }, runActions).runNextStep();

    expect(calls).toEqual(["run-simulation", "run-analysis"]);

    const afterAnalysis = buildGuidedRunController({
      exportReady: false,
      flow: flow({
        analysisComplete: true,
        modelSelected: true,
        simulationComplete: true,
        trainingSkipped: true,
      }),
      compileConfigured: true,
      sourceMode: "model",
    }, runActions);
    expect(afterAnalysis.nextActionKey).toBe("run-compile");
    expect(afterAnalysis.blockerReason).toBeNull();
    await afterAnalysis.runNextStep();
    expect(calls).toEqual(["run-simulation", "run-analysis", "run-compile"]);
    // Skipping training is a decision, not training evidence.
    expect(afterAnalysis.completedEvidence).toEqual(["Design", "Simulate", "Analyse"]);
  });

  it("runs bit-exact co-simulation after model compile and before synthesis", async () => {
    const calls: GuidedRunActionKey[] = [];
    const controller = buildGuidedRunController({
      cosimConfigured: true,
      exportReady: false,
      flow: flow({
        analysisComplete: true,
        compileComplete: true,
        cosimApplicable: true,
        modelSelected: true,
        simulationComplete: true,
        trainingSkipped: true,
      }),
      sourceMode: "model",
    }, actions(calls));

    expect(controller.nextActionKey).toBe("run-cosim");
    await expect(controller.runNextStep()).resolves.toEqual({ ok: true });
    expect(calls).toEqual(["run-cosim"]);
  });

  it("blocks a catalogue model without a canonical schema-backed RTL path", () => {
    const controller = buildGuidedRunController({
      compileConfigured: false,
      exportReady: false,
      flow: flow({ analysisComplete: true, simulationComplete: true, trainingSkipped: true }),
      sourceMode: "model",
    }, actions());

    expect(controller.nextActionKey).toBe("blocked");
    expect(controller.blockerReason).toBe(
      "Selected model has no canonical schema-backed RTL path.",
    );
  });

  it("reports failed actions without claiming progress", async () => {
    const controller = buildGuidedRunController({
      exportReady: false,
      flow: flow(),
      sourceMode: "ode",
    }, {
      ...actions(),
      runSimulation: async () => {
        throw new Error("simulation endpoint failed");
      },
    });

    await expect(controller.runNextStep()).resolves.toEqual({
      error: "simulation endpoint failed",
      ok: false,
    });
  });

  it("keeps export blocked until evidence is ready", async () => {
    const controller = buildGuidedRunController({
      exportReady: false,
      flow: flow({
        analysisComplete: true,
        compileComplete: true,
        simulationComplete: true,
        synthesisComplete: true,
        trainingSkipped: true,
      }),
      sourceMode: "ode",
    }, actions());

    expect(controller.nextActionKey).toBe("blocked");
    expect(controller.blockerReason).toBe("Evidence export is not ready yet.");
    await expect(controller.runNextStep()).resolves.toEqual({
      error: "Evidence export is not ready yet.",
      ok: false,
    });
  });

  it("offers a failed step again as a retry and records how the retry ended", async () => {
    const outcomes: [string, string | null][] = [];
    const failing: GuidedRunActions = {
      ...actions(),
      recordOutcome: (step, failure) => { outcomes.push([step, failure]); },
      runCompile: () => Promise.reject(new Error("RTL emission failed again")),
    };
    const controller = buildGuidedRunController({
      exportReady: false,
      flow: flow({
        analysisComplete: true,
        simulationComplete: true,
        trainingSkipped: true,
        failures: { compile: "RTL emission failed" },
      }),
      sourceMode: "ode",
    }, failing);

    expect(controller.nextActionKey).toBe("run-compile");
    expect(controller.nextActionLabel).toBe("Retry: Compile RTL");
    await expect(controller.runNextStep()).resolves.toEqual({
      error: "RTL emission failed again",
      ok: false,
    });
    expect(outcomes).toEqual([["compile", "RTL emission failed again"]]);
  });

  it("records a successful step so its earlier failure is withdrawn", async () => {
    const outcomes: [string, string | null][] = [];
    const controller = buildGuidedRunController({
      exportReady: false,
      flow: flow({ simulationComplete: false }),
      sourceMode: "ode",
    }, { ...actions(), recordOutcome: (step, failure) => { outcomes.push([step, failure]); } });

    await expect(controller.runNextStep()).resolves.toEqual({ ok: true });
    expect(outcomes).toEqual([["simulate", null]]);
  });

  it("records nothing when there was nothing to run", async () => {
    const outcomes: string[] = [];
    const recordOutcome = (step: string) => { outcomes.push(step); };
    const blocked = buildGuidedRunController({
      exportReady: false,
      flow: flow({ modelSelected: false }),
      sourceMode: "ode",
    }, { ...actions(), recordOutcome });
    const complete = buildGuidedRunController({
      exportReady: true,
      flow: flow({
        analysisComplete: true,
        compileComplete: true,
        evidenceExported: true,
        simulationComplete: true,
        synthesisComplete: true,
        trainingComplete: true,
      }),
      sourceMode: "ode",
    }, { ...actions(), recordOutcome });

    await blocked.runNextStep();
    await complete.runNextStep();
    expect(blocked.nextActionKey).toBe("blocked");
    expect(complete.nextActionKey).toBe("complete");
    expect(outcomes).toEqual([]);
  });

  it("names an unsupported step by its own reason when the registry gives no message", () => {
    const controller = buildGuidedRunController({
      exportReady: false,
      flow: flow({}, { ...allCapabilities, simulate: false }),
      sourceMode: "ode",
    }, actions());

    expect(controller.nextActionKey).toBe("blocked");
    expect(controller.blockerReason).toBe("Simulate capability is unavailable");
  });

  it("does not claim skipped training as completed evidence", () => {
    const controller = buildGuidedRunController({
      exportReady: false,
      flow: flow({ analysisComplete: true, simulationComplete: true, trainingSkipped: true }),
      sourceMode: "ode",
    }, actions());

    expect(controller.completedEvidence).toEqual(["Design", "Simulate", "Analyse"]);
  });
});
