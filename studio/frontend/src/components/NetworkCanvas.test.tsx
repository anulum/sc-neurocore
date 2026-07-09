// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { PipelineEvidenceModel } from "../pipelineEvidence";
import { PipelineEvidenceStrip } from "./NetworkCanvas";

const evidence: PipelineEvidenceModel = {
  actionKind: "studio.pipeline.run",
  classification: "compile",
  evidenceArtifact: "pipeline/evidence.json",
  pipeline: "graph -> simulate -> compile -> synthesise",
  replayRoute: "POST /api/pipeline/run",
  resultArtifact: "pipeline/result.json",
  status: "completed",
  step: "complete",
  target: "ICE40",
};

describe("NetworkCanvas", () => {
  it("renders path-free pipeline action evidence metadata", () => {
    const html = renderToStaticMarkup(<PipelineEvidenceStrip evidence={evidence} />);

    expect(html).toContain("class");
    expect(html).toContain("compile");
    expect(html).toContain("studio.pipeline.run");
    expect(html).toContain("completed");
    expect(html).toContain("ICE40");
    expect(html).toContain("POST /api/pipeline/run");
    expect(html).toContain("pipeline/result.json / pipeline/evidence.json");
  });
});
