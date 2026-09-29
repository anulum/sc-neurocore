// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — portable event training input custody

/** Portable event declarations; the server validates their full scientific content. */
export interface EventTrainingData {
  schema: "sc-neurocore.studio.event-training-data.v1";
  training_cpu_threads: 1;
  manifest: Record<string, unknown>;
  split: Record<string, unknown>;
  encoder: Record<string, unknown>;
  train_split: string;
  evaluation_split: string;
  digests: { manifest: string; split: string; encoder: string };
}

/**
 * Read an object without coercing a malformed document.
 *
 * @param value - JSON field to inspect.
 * @returns Its fields or null.
 */
function record(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown> : null;
}

/**
 * Recognise the digest spelling used by the event-data producer.
 *
 * @param value - Digest field.
 * @returns Whether it is a SHA-256 declaration, without attesting its contents.
 */
function digest(value: unknown): value is string {
  return typeof value === "string" && /^sha256:[0-9a-f]{64}$/.test(value);
}

/**
 * Preserve portable input at workspace and retained-job consumer boundaries.
 *
 * The server remains responsible for file checksums, canonical digests, group
 * partitions and encoder semantics before admitting a job. This synchronous
 * reader checks the envelope and its association with the selected dataset
 * and window; it does not label a declaration as scientifically verified.
 *
 * @param value - Candidate event_data field.
 * @param dataset - Selected training dataset.
 * @param timesteps - Selected training window.
 * @returns The unchanged portable envelope, or no event input for a static dataset.
 * @throws {Error} If an event declaration is missing, malformed or belongs to another input.
 */
export function readEventTrainingData(
  value: unknown, dataset: string, timesteps: number,
): EventTrainingData | undefined {
  if (dataset === "synthetic" || dataset === "mnist") {
    if (value !== undefined && value !== null) {
      throw new Error("Static training datasets cannot carry event input");
    }
    return undefined;
  }
  const data = record(value);
  const manifest = record(data?.manifest);
  const description = record(manifest?.dataset);
  const split = record(data?.split);
  const encoder = record(data?.encoder);
  const digests = record(data?.digests);
  const fields = ["schema", "training_cpu_threads", "manifest", "split", "encoder",
    "train_split", "evaluation_split", "digests"];
  if (!["nmnist", "shd", "dvs_cifar10"].includes(dataset)
    || data === null || Object.keys(data).length !== fields.length
    || !fields.every((key) => Object.hasOwn(data, key))
    || data.schema !== "sc-neurocore.studio.event-training-data.v1"
    || data.training_cpu_threads !== 1
    || manifest?.schema !== "sc-neurocore.event-dataset.v1"
    || description?.name !== dataset
    || split?.schema !== "sc-neurocore.event-dataset-split.v1"
    || encoder?.schema !== "sc-neurocore.input-encoder.v1"
    || encoder.encoder !== "event-binning"
    || encoder.n_steps !== timesteps
    || typeof data.train_split !== "string" || data.train_split.length === 0
    || typeof data.evaluation_split !== "string" || data.evaluation_split.length === 0
    || data.train_split === data.evaluation_split
    || !digest(digests?.manifest) || !digest(digests.split) || !digest(digests.encoder)
    || Object.keys(digests).length !== 3
    || split.manifest_digest !== digests.manifest) {
    throw new Error("Event training input is missing or incompatible with this configuration");
  }
  return {
    schema: data.schema,
    training_cpu_threads: data.training_cpu_threads,
    manifest,
    split,
    encoder,
    train_split: data.train_split,
    evaluation_split: data.evaluation_split,
    digests: { manifest: digests.manifest, split: digests.split, encoder: digests.encoder },
  };
}
