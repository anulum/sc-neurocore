// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

import { describe, expect, it } from "vitest";

import type { TrainingWeightRestorePlan } from "./api/client";
import {
  buildTrainingWeightRestoreVerificationManifest,
  sha256Blob,
  verifyTrainingWeightArtifactBlob,
} from "./trainingRestore";

async function restorePlanForPayload(payload: string): Promise<TrainingWeightRestorePlan> {
  const blob = new Blob([payload]);
  return {
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
      sha256: await sha256Blob(blob),
      size_bytes: blob.size,
    },
  };
}

describe("trainingRestore", () => {
  it("verifies downloaded weight artifact size and digest", async () => {
    const blob = new Blob(["weights"]);
    const plan = await restorePlanForPayload("weights");

    const verification = await verifyTrainingWeightArtifactBlob(
      plan,
      blob,
      () => new Date("2026-06-20T12:00:00Z"),
    );

    expect(verification).toEqual({
      actual_sha256: plan.weights_artifact.sha256,
      expected_sha256: plan.weights_artifact.sha256,
      relative_path: "training/model_state.pt",
      size_bytes: blob.size,
      source_job_id: "sj_training",
      status: "verified",
      verified_at_utc: "2026-06-20T12:00:00.000Z",
    });
  });

  it("rejects mismatched downloaded weight artifact size", async () => {
    const plan = await restorePlanForPayload("weights");

    await expect(
      verifyTrainingWeightArtifactBlob(plan, new Blob(["other weights"])),
    ).rejects.toThrow("size mismatch");
  });

  it("rejects mismatched downloaded weight artifact digest", async () => {
    const plan = await restorePlanForPayload("weights");
    const forgedPlan = {
      ...plan,
      weights_artifact: { ...plan.weights_artifact, size_bytes: new Blob(["forged"]).size },
    };

    await expect(
      verifyTrainingWeightArtifactBlob(forgedPlan, new Blob(["forged"])),
    ).rejects.toThrow("SHA-256 mismatch");
  });

  it("builds a path-free restore verification manifest", async () => {
    const plan = await restorePlanForPayload("weights");
    const verification = await verifyTrainingWeightArtifactBlob(
      plan,
      new Blob(["weights"]),
      () => new Date("2026-06-20T12:00:00Z"),
    );

    expect(buildTrainingWeightRestoreVerificationManifest(plan, verification)).toEqual({
      artifact_route_template: "/api/studio/jobs/{job_id}/artifacts/{artifact_path}",
      loader_policy: "download_from_authenticated_artifact_route_and_verify_sha256",
      metadata_artifact_path: "training/model_state.json",
      metadata_artifact_sha256: "b".repeat(64),
      metadata_artifact_size_bytes: 512,
      schema_version: "studio.training.weight-restore-verification.v1",
      source_job_id: "sj_training",
      source_status: "completed",
      verification,
      weights_artifact_path: "training/model_state.pt",
      weights_artifact_sha256: plan.weights_artifact.sha256,
      weights_artifact_size_bytes: new Blob(["weights"]).size,
    });
  });

  it("rejects restore verification manifests for inconsistent hashes", async () => {
    const plan = await restorePlanForPayload("weights");
    const verification = {
      actual_sha256: "c".repeat(64),
      expected_sha256: plan.weights_artifact.sha256,
      relative_path: "training/model_state.pt",
      size_bytes: new Blob(["weights"]).size,
      source_job_id: "sj_training",
      status: "verified" as const,
      verified_at_utc: "2026-06-20T12:00:00.000Z",
    };

    expect(() => buildTrainingWeightRestoreVerificationManifest(plan, verification)).toThrow(
      "digest is not confirmed",
    );
  });
});
