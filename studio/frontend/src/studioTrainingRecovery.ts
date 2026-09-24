// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — retained Training Monitor job decoding

import type { ResolvedTrainingConfig, TrainingJobSummary } from "./api/client";
import type { StudioProjectTrainingConfig } from "./studioProjectState";

const TRAINING_STATUSES = new Set([
  "running", "completed", "stopped", "failed", "interrupted", "unknown",
]);
const STOP_STATUSES = new Set([
  "stopping", "completed", "stopped", "failed", "interrupted", "unknown",
]);
const TERMINAL_STATUSES = new Set(["completed", "stopped", "failed", "interrupted"]);

/** Statuses the Stop API can report after a cooperative cancel request. */
export type TrainingStopStatus =
  "stopping" | "completed" | "stopped" | "failed" | "interrupted" | "unknown";

/**
 * Decide whether a selected training run already has a final outcome.
 *
 * @param status - Current public Training Monitor status.
 * @returns Whether a pending transport reply must leave it unchanged.
 */
export function isTrainingTerminalStatus(status: string): boolean {
  return TERMINAL_STATUSES.has(status);
}

/**
 * Read a plain object.
 *
 * @param value - Candidate object.
 * @returns Its fields, or null.
 */
function record(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown> : null;
}

/**
 * Check a positive finite number.
 *
 * @param value - Candidate number.
 * @returns Whether it is finite and positive.
 */
function positiveFinite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

/**
 * Check a nonnegative finite number.
 *
 * @param value - Candidate number.
 * @returns Whether it is finite and nonnegative.
 */
function nonnegativeFinite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

/**
 * Check a positive integer.
 *
 * @param value - Candidate integer.
 * @returns Whether it is positive.
 */
function positiveInteger(value: unknown): value is number {
  return positiveFinite(value) && Number.isInteger(value);
}

/**
 * Check a nonempty string.
 *
 * @param value - Candidate text.
 * @returns Whether it contains text.
 */
function nonemptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

/**
 * Validate the complete resolved configuration before displaying provenance.
 *
 * @param value - Candidate configuration.
 * @returns Verified configuration or null.
 */
function resolvedConfig(value: unknown): ResolvedTrainingConfig | null {
  const data = record(value);
  if (data?.schema_version !== "studio.training-config.v1"
    || !nonemptyString(data.dataset)
    || !positiveInteger(data.epochs)
    || !positiveInteger(data.batch_size)
    || !positiveFinite(data.lr)
    || !Array.isArray(data.hidden)
    || !data.hidden.every(positiveInteger)
    || !positiveInteger(data.timesteps)
    || !nonemptyString(data.surrogate)
    || typeof data.learn_beta !== "boolean"
    || typeof data.learn_threshold !== "boolean"
    || !nonnegativeFinite(data.max_grad_norm)
    || typeof data.seed !== "number"
    || !Number.isSafeInteger(data.seed)
    || data.seed < 0
    || data.seed >= 2 ** 32) {
    return null;
  }
  return data as unknown as ResolvedTrainingConfig;
}

/**
 * Decode the durable training list before it becomes UI evidence.
 *
 * Old ledger rows legitimately have no configuration. A malformed non-null
 * configuration is refused instead of being shown as an old row.
 *
 * @param value - Candidate list response.
 * @returns Validated summaries in server order.
 */
export function decodeTrainingJobSummaries(value: unknown): TrainingJobSummary[] {
  if (!Array.isArray(value)) throw new Error("Training job list is invalid");
  return value.map((item: unknown) => {
    const data = record(item);
    if (data === null
      || !nonemptyString(data.job_id)
      || data.job_id.includes("/")
      || !TRAINING_STATUSES.has(String(data.status))) {
      throw new Error("Training job list contains an invalid record");
    }
    const config = data.config === null ? null : resolvedConfig(data.config);
    if (data.config !== null && config === null) {
      throw new Error("Training job list contains an invalid configuration");
    }
    return { job_id: data.job_id, status: data.status as string, config };
  });
}

/**
 * Confirm the fresh status belongs to the selected retained training job.
 *
 * @param value - Candidate status response.
 * @param jobId - Selected job ID.
 * @returns The verified current status.
 */
export function decodeTrainingRecoveryStatus(value: unknown, jobId: string): string {
  const data = record(value);
  if (data?.job_id !== jobId || !TRAINING_STATUSES.has(String(data.status))) {
    throw new Error("Selected training job status is invalid");
  }
  return data.status as string;
}

/**
 * Read a Stop outcome only for the selected job and the public training vocabulary.
 *
 * @param value - Candidate Stop API response.
 * @param jobId - Job the operator asked to stop.
 * @returns Its verified current or pending outcome.
 */
export function decodeTrainingStopResult(value: unknown, jobId: string): TrainingStopStatus {
  const data = record(value);
  if (data?.job_id !== jobId || !STOP_STATUSES.has(String(data.status))) {
    throw new Error("Training stop outcome is invalid");
  }
  return data.status as TrainingStopStatus;
}

/**
 * Carry only project settings from a verified historical job configuration.
 *
 * @param config - Validated retained configuration, or absent legacy data.
 * @returns Settings for provenance display, or null.
 */
export function observedTrainingConfig(
  config: ResolvedTrainingConfig | null,
): StudioProjectTrainingConfig | null {
  if (config === null) return null;
  return {
    dataset: config.dataset,
    epochs: config.epochs,
    batch_size: config.batch_size,
    lr: config.lr,
    hidden: [...config.hidden],
    timesteps: config.timesteps,
    surrogate: config.surrogate,
    learn_beta: config.learn_beta,
    learn_threshold: config.learn_threshold,
  };
}
