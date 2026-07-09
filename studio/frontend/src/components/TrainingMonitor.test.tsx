// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import {
  TrainingCheckpointControls,
  TrainingEvidenceStrip,
  TrainingWeightAttachStrip,
  TrainingWeightLiveAttachStrip,
  TrainingWeightMaterializationStrip,
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
        onExportVerification={() => undefined}
        onVerify={() => undefined}
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
        verification={{
          actual_sha256: "c".repeat(64),
          expected_sha256: "a".repeat(64),
          relative_path: "training/model_state.pt",
          size_bytes: 4096,
          source_job_id: "sj_training",
          status: "verified",
          verified_at_utc: "2026-06-20T12:00:00.000Z",
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
    expect(html).toContain("Verified");
    expect(html).toContain("cccccccccccc");
    expect(html).toContain("Verify weights");
    expect(html).toContain("Export verification");
    expect(html).toContain(">9610<");
  });

  it("renders verification export disabled before artifact verification", () => {
    const html = renderToStaticMarkup(
      <TrainingWeightRestorePlanStrip
        onExportVerification={() => undefined}
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

    expect(html).toContain("pending");
    expect(html).toContain("Export verification");
    expect(html).toContain("disabled");
  });

  it("renders nothing when no weight materialization is present", () => {
    const html = renderToStaticMarkup(
      <TrainingWeightMaterializationStrip materialization={null} />,
    );

    expect(html).toBe("");
  });

  it("renders path-free server weight materialization evidence", () => {
    const html = renderToStaticMarkup(
      <TrainingWeightMaterializationStrip
        materialization={{
          artifacts: [
            { relative_path: "training/weight-restore.json", sha256: "f".repeat(64), size_bytes: 256 },
          ],
          evidence_classification: "training",
          job_id: "sj_restore",
          materialization: {
            architecture: "64->128->10",
            config_sha256: "7".repeat(64),
            format: "torch_state_dict",
            framework: "pytorch",
            loaded_key_count: 6,
            metadata_sha256: "8".repeat(64),
            parameter_count: 9610,
            schema_version: "studio.training.weight-materialization.v1",
            source_job_id: "sj_training",
            weights_sha256: "9".repeat(64),
          },
          schema_version: "studio.training.weight-restore.v1",
          source_job_id: "sj_training",
          source_status: "completed",
          status: "completed",
        }}
      />,
    );

    expect(html).toContain("studio.training.weight-restore.v1");
    expect(html).toContain("training");
    expect(html).toContain("sj_restore");
    expect(html).toContain("sj_training");
    expect(html).toContain("Loaded keys");
    expect(html).toContain(">6<");
    expect(html).toContain("999999999999");
    expect(html).toContain("888888888888");
  });

  it("renders nothing when no weight attach result is present", () => {
    const html = renderToStaticMarkup(<TrainingWeightAttachStrip attach={null} />);

    expect(html).toBe("");
  });

  it("renders path-free warm-start attach metadata", () => {
    const html = renderToStaticMarkup(
      <TrainingWeightAttachStrip
        attach={{
          architecture_fingerprint: "c".repeat(64),
          job_id: "sj_attach",
          source_job_id: "sj_training",
          status: "running",
        }}
      />,
    );

    expect(html).toContain("warm_start");
    expect(html).toContain("sj_attach");
    expect(html).toContain("sj_training");
    expect(html).toContain("running");
    expect(html).toContain("cccccccccccc");
  });

  it("renders nothing when no live attach result is present", () => {
    const html = renderToStaticMarkup(<TrainingWeightLiveAttachStrip liveAttach={null} />);

    expect(html).toBe("");
  });

  it("renders path-free live attach request metadata", () => {
    const html = renderToStaticMarkup(
      <TrainingWeightLiveAttachStrip
        liveAttach={{
          architecture_fingerprint: "d".repeat(64),
          source_job_id: "sj_source",
          status: "attach_requested",
          target_job_id: "sj_running",
        }}
      />,
    );

    expect(html).toContain("Live attach");
    expect(html).toContain("attach_requested");
    expect(html).toContain("sj_running");
    expect(html).toContain("sj_source");
    expect(html).toContain("dddddddddddd");
  });
});
