import type { TrainingWeightRestorePlan } from "./api/client";

export interface TrainingWeightRestoreVerification {
  actual_sha256: string;
  expected_sha256: string;
  relative_path: string;
  size_bytes: number;
  source_job_id: string;
  status: "verified";
  verified_at_utc: string;
}

export interface TrainingWeightRestoreVerificationManifest {
  artifact_route_template: string;
  loader_policy: string;
  metadata_artifact_sha256: string;
  metadata_artifact_size_bytes: number;
  metadata_artifact_path: string;
  schema_version: "studio.training.weight-restore-verification.v1";
  source_job_id: string;
  source_status: string;
  verification: TrainingWeightRestoreVerification;
  weights_artifact_sha256: string;
  weights_artifact_size_bytes: number;
  weights_artifact_path: string;
}

export async function sha256Blob(blob: Blob): Promise<string> {
  const bytes = await blob.arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

export async function verifyTrainingWeightArtifactBlob(
  restorePlan: TrainingWeightRestorePlan,
  blob: Blob,
  clock: () => Date = () => new Date(),
): Promise<TrainingWeightRestoreVerification> {
  const actualSize = blob.size;
  const expected = restorePlan.weights_artifact;
  if (actualSize !== expected.size_bytes) {
    throw new Error(
      `Training weight artifact size mismatch: expected ${expected.size_bytes}, got ${actualSize}`,
    );
  }
  const actualSha256 = await sha256Blob(blob);
  if (actualSha256 !== expected.sha256) {
    throw new Error("Training weight artifact SHA-256 mismatch.");
  }
  return {
    actual_sha256: actualSha256,
    expected_sha256: expected.sha256,
    relative_path: expected.relative_path,
    size_bytes: actualSize,
    source_job_id: restorePlan.source_job_id,
    status: "verified",
    verified_at_utc: clock().toISOString(),
  };
}

export function buildTrainingWeightRestoreVerificationManifest(
  restorePlan: TrainingWeightRestorePlan,
  verification: TrainingWeightRestoreVerification,
): TrainingWeightRestoreVerificationManifest {
  if (restorePlan.source_job_id !== verification.source_job_id) {
    throw new Error("Training weight restore verification source job mismatch.");
  }
  if (restorePlan.weights_artifact.relative_path !== verification.relative_path) {
    throw new Error("Training weight restore verification artifact path mismatch.");
  }
  if (restorePlan.weights_artifact.sha256 !== verification.expected_sha256) {
    throw new Error("Training weight restore verification expected digest mismatch.");
  }
  if (verification.actual_sha256 !== verification.expected_sha256) {
    throw new Error("Training weight restore verification digest is not confirmed.");
  }
  if (restorePlan.weights_artifact.size_bytes !== verification.size_bytes) {
    throw new Error("Training weight restore verification size mismatch.");
  }
  return {
    artifact_route_template: restorePlan.artifact_route_template,
    loader_policy: restorePlan.loader_policy,
    metadata_artifact_path: restorePlan.metadata_artifact.relative_path,
    metadata_artifact_sha256: restorePlan.metadata_artifact.sha256,
    metadata_artifact_size_bytes: restorePlan.metadata_artifact.size_bytes,
    schema_version: "studio.training.weight-restore-verification.v1",
    source_job_id: restorePlan.source_job_id,
    source_status: restorePlan.source_status,
    verification,
    weights_artifact_path: restorePlan.weights_artifact.relative_path,
    weights_artifact_sha256: restorePlan.weights_artifact.sha256,
    weights_artifact_size_bytes: restorePlan.weights_artifact.size_bytes,
  };
}
