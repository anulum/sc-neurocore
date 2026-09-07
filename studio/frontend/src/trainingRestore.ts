// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Source/config provenance header

/**
 * Checking that downloaded training weights are the weights that were trained.
 *
 * Two things are compared before any weight is loaded: the artefact's size and
 * its SHA-256, both against what the restore plan says the server wrote. A
 * mismatch throws rather than returning a flag, because the caller's next step
 * is to load the file into a network, and a verification whose failure can be
 * ignored is not a verification.
 *
 * The manifest builder checks the plan and the verification against each other
 * as well -- same job, same path, same expected digest, and a digest that
 * actually matched. A manifest is the record someone else reads to believe the
 * restore happened; assembling one from two documents that disagree would make
 * that record false.
 */

import type { TrainingWeightRestorePlan } from "./api/client";

/** What a verification found: both digests, the size, and when it ran. */
export interface TrainingWeightRestoreVerification {
  actual_sha256: string;
  expected_sha256: string;
  relative_path: string;
  size_bytes: number;
  source_job_id: string;
  status: "verified";
  verified_at_utc: string;
}

/**
 * The record of a verified restore: which job, which artefacts, which digests,
 * and how a loader is expected to fetch them.
 */
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

/**
 * Digest a blob with SHA-256.
 *
 * @param blob - The bytes.
 * @returns The digest, as lowercase hexadecimal.
 */
export async function sha256Blob(blob: Blob): Promise<string> {
  const bytes = await blob.arrayBuffer();
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");
}

/**
 * Check a downloaded weights artefact against its restore plan.
 *
 * The size is checked before the digest, because it is free and it catches
 * the common failure -- a truncated download -- without hashing megabytes
 * to find out.
 *
 * @param restorePlan - What the server says the artefact should be.
 * @param blob - The bytes that arrived.
 * @param clock - Reads the time the verification happened.
 * @returns The verification.
 * @throws {Error} When the size or the digest disagrees. It throws rather
 *   than reporting, because the caller's next step is to load these
 *   weights into a network.
 */
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

/**
 * Build the record of a verified restore.
 *
 * @param restorePlan - The plan the verification was made against.
 * @param verification - What the verification found.
 * @returns The manifest.
 * @throws {Error} When the plan and the verification disagree about the
 *   job, the path, the expected digest or the size, or when the
 *   verification's own digests do not match. A manifest assembled from
 *   documents that disagree is a false record.
 */
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
