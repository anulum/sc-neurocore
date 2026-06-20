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
