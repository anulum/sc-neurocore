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

  it("blocks compile honestly when the guided run is still in model mode", () => {
    const controller = buildGuidedRunController({
      exportReady: false,
      flow: flow({ analysisComplete: true, simulationComplete: true, trainingSkipped: true }),
      sourceMode: "model",
    }, actions());

    expect(controller.nextActionKey).toBe("blocked");
    expect(controller.blockerReason).toBe("Guided compile requires ODE source mode.");
  });

  it("supports the catalogue model path through sim and analysis before the ODE compile gate", async () => {
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
      sourceMode: "model",
    }, runActions);
    expect(afterAnalysis.nextActionKey).toBe("blocked");
    expect(afterAnalysis.blockerReason).toBe("Guided compile requires ODE source mode.");
    expect(afterAnalysis.completedEvidence).toEqual(
      expect.arrayContaining(["Design", "Simulate", "Analyse", "Train"]),
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
