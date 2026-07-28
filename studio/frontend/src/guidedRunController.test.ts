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
  design: true,
  export: true,
  simulate: true,
  synthesise: true,
  train: true,
};

function flow(overrides: Partial<GuidedFlowInputs> = {}, capabilities = allCapabilities) {
  return computeGuidedFlowState(
    {
      analysisComplete: false,
      compileComplete: false,
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
    expect(afterAnalysis.completedEvidence).toEqual(
      expect.arrayContaining(["Design", "Simulate", "Analyse", "Train"]),
    );
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
});
