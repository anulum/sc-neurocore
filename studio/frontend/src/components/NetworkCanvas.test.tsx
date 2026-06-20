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
