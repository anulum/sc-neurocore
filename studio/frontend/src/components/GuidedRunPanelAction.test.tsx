// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio guided run panel action tests
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { buildGuidedRunController } from "../guidedRunController";
import { computeGuidedFlowState } from "../guidedFlowState";
import GuidedFlowPanel from "./GuidedFlowPanel";

describe("GuidedFlowPanel guided run actions", () => {
  it("renders next action, completed evidence, and export readiness", () => {
    const state = computeGuidedFlowState({
      analysisComplete: true,
      compileComplete: true,
      evidenceExported: false,
      modelSelected: true,
      simulationComplete: true,
      synthesisComplete: true,
      trainingComplete: false,
      trainingSkipped: true,
    });
    const controller = buildGuidedRunController({
      exportReady: true,
      flow: state,
      sourceMode: "ode",
    }, {
      exportEvidence: async () => undefined,
      runAnalysis: async () => undefined,
      runCompile: async () => undefined,
      runSimulation: async () => undefined,
      runSynthesis: async () => undefined,
      skipTraining: async () => undefined,
    });

    const html = renderToStaticMarkup(<GuidedFlowPanel controller={controller} state={state} />);

    expect(html).toContain("Run next step");
    expect(html).toContain("Export evidence");
    expect(html).toContain("Evidence ready");
    expect(html).toContain("Completed evidence");
    expect(html).toContain("Design, Simulate, Analyse, Train, Compile, Synthesise");
  });

  it("renders blocker reasons from the controller", () => {
    const state = computeGuidedFlowState({
      analysisComplete: true,
      compileComplete: true,
      evidenceExported: false,
      modelSelected: true,
      simulationComplete: true,
      synthesisComplete: false,
      trainingComplete: false,
      trainingSkipped: true,
    }, {
      analyse: true,
      compile: true,
      design: true,
      export: true,
      simulate: true,
      synthesise: false,
      train: true,
    });
    const controller = buildGuidedRunController({
      capabilityMessages: { synthesise: "Synthesis tools are unavailable." },
      exportReady: false,
      flow: state,
      sourceMode: "ode",
    }, {
      exportEvidence: async () => undefined,
      runAnalysis: async () => undefined,
      runCompile: async () => undefined,
      runSimulation: async () => undefined,
      runSynthesis: async () => undefined,
      skipTraining: async () => undefined,
    });

    const html = renderToStaticMarkup(<GuidedFlowPanel controller={controller} state={state} />);

    expect(html).toContain("Synthesis tools are unavailable.");
    expect(html).toContain("Evidence pending");
    expect(html).toContain("disabled");
  });
});
