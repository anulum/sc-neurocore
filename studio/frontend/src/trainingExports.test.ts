// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio training export helper tests

import { describe, expect, it } from "vitest";

import type {
  TrainingCheckpointPayload,
  TrainingWeightRestorePlan,
} from "./api/client";
import type { TrainingWeightRestoreVerification } from "./trainingRestore";
import {
  trainingCheckpointExport,
  trainingCheckpointFilename,
  trainingWeightRestoreVerificationExport,
} from "./trainingExports";

const digest = "a".repeat(64);

function checkpoint(jobId = "job/with spaces"): TrainingCheckpointPayload {
  return {
    checkpoint_sha256: digest,
    config: { dataset: "synthetic" },
    config_sha256: "b".repeat(64),
    evidence_summary: null,
    final_metrics: null,
    generated_at_utc: "2026-06-21T00:00:00Z",
    job_id: jobId,
    schema_version: "studio.training.checkpoint.v1",
    status: "completed",
    weight_checkpoint: null,
  };
}

function restorePlan(): TrainingWeightRestorePlan {
  return {
    architecture: "64->128->10",
    artifact_route_template: "/api/studio/jobs/{job_id}/artifacts/{artifact_path}",
    config_sha256: "b".repeat(64),
    format: "torch_state_dict",
    framework: "pytorch",
    loader_policy: "download_from_authenticated_artifact_route_and_verify_sha256",
    metadata_artifact: {
      relative_path: "training/model_state.json",
      sha256: "c".repeat(64),
      size_bytes: 512,
    },
    parameter_count: 9610,
    restore_ready: true,
    schema_version: "studio.training.weight-restore-plan.v1",
    source_job_id: "job/with spaces",
    source_status: "completed",
    weights_artifact: {
      relative_path: "training/model_state.pt",
      sha256: digest,
      size_bytes: 4096,
    },
  };
}

function verification(): TrainingWeightRestoreVerification {
  return {
    actual_sha256: digest,
    expected_sha256: digest,
    relative_path: "training/model_state.pt",
    size_bytes: 4096,
    source_job_id: "job/with spaces",
    status: "verified",
    verified_at_utc: "2026-06-21T00:00:00.000Z",
  };
}

describe("training export helpers", () => {
  it("builds safe checkpoint export filenames", () => {
    expect(trainingCheckpointFilename(checkpoint())).toBe("training_checkpoint_job_with_spaces.json");
    expect(trainingCheckpointFilename(checkpoint("  ///  "))).toBe("training_checkpoint_training.json");
  });

  it("exports checkpoint JSON with stable indentation", async () => {
    const payload = checkpoint("job-1");
    const exported = trainingCheckpointExport(payload);

    expect(exported.filename).toBe("training_checkpoint_job-1.json");
    await expect(exported.blob.text()).resolves.toBe(JSON.stringify(payload, null, 2));
  });

  it("exports weight-restore verification manifests with safe filenames", async () => {
    const exported = trainingWeightRestoreVerificationExport(restorePlan(), verification());

    expect(exported.filename).toBe("training_weight_restore_job_with_spaces.json");
    await expect(exported.blob.text()).resolves.toContain(
      "\"schema_version\": \"studio.training.weight-restore-verification.v1\"",
    );
  });

  it("rejects inconsistent weight-restore verification exports", () => {
    const forgedVerification = { ...verification(), actual_sha256: "d".repeat(64) };

    expect(() =>
      trainingWeightRestoreVerificationExport(restorePlan(), forgedVerification),
    ).toThrow("digest is not confirmed");
  });
});
