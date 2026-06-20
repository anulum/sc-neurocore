import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import {
  TrainingCheckpointControls,
  TrainingEvidenceStrip,
  TrainingWeightRestorePlanStrip,
} from "./TrainingMonitor";

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

  it("renders checkpoint controls with export disabled until a job exists", () => {
    const html = renderToStaticMarkup(
      <TrainingCheckpointControls
        canExport={false}
        onExport={() => undefined}
        onImportText={() => undefined}
      />,
    );

    expect(html).toContain("Export checkpoint");
    expect(html).toContain("Import checkpoint");
    expect(html).toContain("disabled");
    expect(html).toContain("Import training checkpoint file");
  });

  it("renders checkpoint weight restore plan metadata", () => {
    const html = renderToStaticMarkup(
      <TrainingWeightRestorePlanStrip
        restorePlan={{
          architecture: "64->128->10",
          artifact_route_template: "/api/studio/jobs/{job_id}/artifacts/{artifact_path}",
          config_sha256: "2".repeat(64),
          format: "torch_state_dict",
          framework: "pytorch",
          loader_policy: "download_from_authenticated_artifact_route_and_verify_sha256",
          metadata_artifact: {
            relative_path: "training/model_state.json",
            sha256: "b".repeat(64),
            size_bytes: 512,
          },
          parameter_count: 9610,
          restore_ready: true,
          schema_version: "studio.training.weight-restore-plan.v1",
          source_job_id: "sj_training",
          source_status: "completed",
          weights_artifact: {
            relative_path: "training/model_state.pt",
            sha256: "a".repeat(64),
            size_bytes: 4096,
          },
        }}
      />,
    );

    expect(html).toContain("studio.training.weight-restore-plan.v1");
    expect(html).toContain("sj_training");
    expect(html).toContain("completed");
    expect(html).toContain("download_from_authenticated_artifact_route_and_verify_sha256");
    expect(html).toContain("/api/studio/jobs/{job_id}/artifacts/{artifact_path}");
    expect(html).toContain("training/model_state.pt #aaaaaaaaaaaa");
    expect(html).toContain("training/model_state.json #bbbbbbbbbbbb");
    expect(html).toContain(">9610<");
  });
});
