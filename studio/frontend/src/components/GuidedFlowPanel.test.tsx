// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio guided default-flow panel tests
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { computeGuidedFlowState } from "../guidedFlowState";
import GuidedFlowPanel from "./GuidedFlowPanel";

describe("GuidedFlowPanel", () => {
  it("renders step titles, progress, current step, and blocked reasons", () => {
    const state = computeGuidedFlowState({
      modelSelected: true,
      simulationComplete: true,
      analysisComplete: false,
      trainingComplete: false,
      trainingSkipped: false,
      compileComplete: false,
      cosimApplicable: false,
      cosimComplete: false,
      synthesisComplete: false,
      evidenceExported: false,
    });

    const html = renderToStaticMarkup(<GuidedFlowPanel state={state} />);

    expect(html).toContain("Guided flow");
    expect(html).toContain("2/7");
    expect(html).toContain("Design");
    expect(html).toContain("Synthesise");
    expect(html).toContain("Export evidence");
    expect(html).toContain('data-step="analyse"');
    expect(html).toContain('aria-current="step"');
    // Train is downstream of analyse and renders its blocked reason.
    expect(html).toContain("Requires Analyse");
    expect(html).toContain("(optional)");
  });

  it("marks every step done for a completed flow", () => {
    const state = computeGuidedFlowState({
      modelSelected: true,
      simulationComplete: true,
      analysisComplete: true,
      trainingComplete: true,
      trainingSkipped: false,
      compileComplete: true,
      cosimApplicable: false,
      cosimComplete: false,
      synthesisComplete: true,
      evidenceExported: true,
    });

    const html = renderToStaticMarkup(<GuidedFlowPanel state={state} />);

    expect(html).toContain("7/7");
    expect(html).not.toContain('aria-current="step"');
    expect(html).not.toContain("blocked");
  });
});
