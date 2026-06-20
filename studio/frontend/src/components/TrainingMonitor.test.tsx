import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { TrainingEvidenceStrip } from "./TrainingMonitor";

describe("TrainingMonitor", () => {
  it("renders path-free training evidence metadata for submitted jobs", () => {
    const html = renderToStaticMarkup(
      <TrainingEvidenceStrip
        evidence={{
          actionKind: "studio.training.run",
          classification: "training",
          configSummary: "synthetic, 4 epochs, superspike, 16 steps",
          evidenceArtifact: "training/evidence.json",
          jobId: "sj_training",
          latestEpoch: "2",
          replayRoute: "POST /api/training/start",
          status: "completed",
          statusArtifact: "training/status.json",
        }}
      />,
    );

    expect(html).toContain("Evidence");
    expect(html).toContain("training");
    expect(html).toContain("studio.training.run");
    expect(html).toContain("sj_training");
    expect(html).toContain("POST /api/training/start");
    expect(html).toContain("training/status.json / training/evidence.json");
    expect(html).toContain("synthetic, 4 epochs, superspike, 16 steps");
    expect(html).toContain("Epoch");
    expect(html).toContain(">2<");
  });
});
