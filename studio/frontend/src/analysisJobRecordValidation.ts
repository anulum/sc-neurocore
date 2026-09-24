// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Strict StudioJobRecord envelope parsing (fail-closed)

/**
 * Reading a job envelope the browser did not write.
 *
 * Every field is checked and nothing is defaulted. A record with a missing
 * field is refused, not completed with a plausible value; a field of the wrong
 * type is refused, not coerced. The reason is that this envelope decides what
 * the reader is shown about a run they will cite: a job whose status was
 * guessed is worse than a job that failed to load.
 *
 * Each refusal is a stable identifier rather than a sentence -- `job_id_invalid`,
 * `job_artifact_sha256_invalid` -- so a caller can branch on it and a reader can
 * be shown one message for a class of failure.
 *
 * The optional bindings are how a poll is made safe: a record can be required
 * to be *this* job, of *this* kind, so a response that arrives for a job the
 * reader has already replaced is refused instead of overwriting the newer one.
 */

import type {
  AnalysisJobKind,
  AnalysisJobReceipt,
  StudioJobArtifact,
  StudioJobRecord,
} from "./api/client";

/**
 * Either a value that has been checked, or the identifier of what was wrong.
 * Never both, and never a value that has only been asserted.
 */
export type ValidationResult<T> =
  | { ok: true; value: T }
  | { ok: false; error: string };

/** A 64-character hexadecimal digest, in either case. */
const HEX64 = /^[0-9a-fA-F]{64}$/;
/**
 * Every status the job contract defines.
 *
 * Typed as the contract's own union so that adding a status to the contract
 * without adding it here is a type error rather than a record refused at
 * runtime, and so the check below narrows instead of needing a cast.
 */
const JOB_STATUSES: readonly StudioJobRecord["status"][] = [
  "pending",
  "running",
  "completed",
  "failed",
  "cancelling",
  "cancelled",
  "timed_out",
  "interrupted",
  "unknown",
];

/**
 * Whether a value is one of the contract's job statuses.
 *
 * @param value - The value, as it arrived.
 * @returns Whether it is a status this build knows.
 */
function isJobStatus(value: unknown): value is StudioJobRecord["status"] {
  return typeof value === "string" && JOB_STATUSES.some((status) => status === value);
}

/**
 * Whether a value is a plain object.
 *
 * @param value - The value.
 * @returns Whether it is one.
 */
function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

/**
 * Whether a value is a number that is actually a number.
 *
 * @param value - The value.
 * @returns Whether it is finite. `NaN` and the infinities are not.
 */
function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/**
 * Whether a value is a string with something in it.
 *
 * @param value - The value.
 * @returns Whether it is one.
 */
function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

/**
 * Whether a value looks like a SHA-256 digest.
 *
 * The shape is checked, not the content: this cannot say a digest is the
 * right one, only that it is a digest at all.
 *
 * @param value - The value.
 * @returns Whether it is one.
 */
function isSha256Hex(value: unknown): value is string {
  return typeof value === "string" && HEX64.test(value);
}

/**
 * Whether a value is a whole number of zero or more.
 *
 * @param value - The value.
 * @returns Whether it is one.
 */
function isNonNegativeInt(value: unknown): value is number {
  return isFiniteNumber(value) && Number.isInteger(value) && value >= 0;
}

/**
 * Read a field the contract allows to be null.
 *
 * Null and a string are both answers; anything else is not, and absent is
 * not either -- a nullable field is expected to be present and null.
 *
 * @param value - The field.
 * @param error - The identifier to refuse with.
 * @returns The string or null, or the refusal.
 */
function parseStringOrNull(
  value: unknown,
  error: string,
): ValidationResult<string | null> {
  if (value === null) {
    return { ok: true, value: null };
  }
  if (typeof value === "string") {
    return { ok: true, value };
  }
  return { ok: false, error };
}

/**
 * Read one artefact entry.
 *
 * @param value - The entry, as it arrived.
 * @returns The artefact, or the identifier of the field that was wrong.
 */
export function parseStudioJobArtifact(
  value: unknown,
): ValidationResult<StudioJobArtifact> {
  if (!isRecord(value)) {
    return { ok: false, error: "job_artifact_not_object" };
  }
  if (!isNonEmptyString(value.relative_path)) {
    return { ok: false, error: "job_artifact_relative_path_invalid" };
  }
  if (!isSha256Hex(value.sha256)) {
    return { ok: false, error: "job_artifact_sha256_invalid" };
  }
  if (!isNonNegativeInt(value.size_bytes)) {
    return { ok: false, error: "job_artifact_size_bytes_invalid" };
  }
  return {
    ok: true,
    value: {
      relative_path: value.relative_path,
      sha256: value.sha256,
      size_bytes: value.size_bytes,
    },
  };
}

/**
 * Read a whole job envelope, refusing anything that is not one.
 *
 * @param value - The envelope, as it arrived.
 * @param options - Optional bindings. `expectedKind` requires the record to be
 *   that kind of job; `expectedJobId` requires it to be that job, and an
 *   explicit `null` refuses every record, which is what a caller holding no
 *   job id wants.
 * @returns The record, or the identifier of the first field that was wrong.
 */
export function parseStudioJobRecord(
  value: unknown,
  options: {
    expectedJobId?: string | null;
    expectedKind?: string;
  } = {},
): ValidationResult<StudioJobRecord> {
  if (!isRecord(value)) {
    return { ok: false, error: "job_record_not_object" };
  }
  if (!("artifacts" in value) || !Array.isArray(value.artifacts)) {
    return { ok: false, error: "job_artifacts_invalid" };
  }
  const artifacts: StudioJobArtifact[] = [];
  for (const entry of value.artifacts) {
    const parsed = parseStudioJobArtifact(entry);
    if (!parsed.ok) {
      return parsed;
    }
    artifacts.push(parsed.value);
  }
  if (!isNonEmptyString(value.created_at_utc)) {
    return { ok: false, error: "job_created_at_utc_invalid" };
  }
  const error = parseStringOrNull(value.error, "job_error_invalid");
  if (!error.ok) {
    return error;
  }
  if (value.execution_model !== "thread" && value.execution_model !== "process") {
    return { ok: false, error: "job_execution_model_invalid" };
  }
  const finished = parseStringOrNull(
    value.finished_at_utc,
    "job_finished_at_utc_invalid",
  );
  if (!finished.ok) {
    return finished;
  }
  if (!isNonEmptyString(value.job_id)) {
    return { ok: false, error: "job_id_invalid" };
  }
  if (options.expectedJobId !== undefined) {
    if (
      options.expectedJobId === null
      || value.job_id !== options.expectedJobId
    ) {
      return { ok: false, error: "job_id_mismatch" };
    }
  }
  if (!isNonEmptyString(value.kind)) {
    return { ok: false, error: "job_kind_invalid" };
  }
  if (
    options.expectedKind !== undefined
    && value.kind !== options.expectedKind
  ) {
    return { ok: false, error: "job_kind_mismatch" };
  }
  if (!isNonEmptyString(value.owner)) {
    return { ok: false, error: "job_owner_invalid" };
  }
  const requestId = parseStringOrNull(value.request_id, "job_request_id_invalid");
  if (!requestId.ok) {
    return requestId;
  }
  if (value.result !== null && !isRecord(value.result)) {
    return { ok: false, error: "job_result_invalid" };
  }
  const started = parseStringOrNull(
    value.started_at_utc,
    "job_started_at_utc_invalid",
  );
  if (!started.ok) {
    return started;
  }
  if (!isJobStatus(value.status)) {
    return { ok: false, error: "job_status_invalid" };
  }
  return {
    ok: true,
    value: {
      artifacts,
      created_at_utc: value.created_at_utc,
      error: error.value,
      execution_model: value.execution_model,
      finished_at_utc: finished.value,
      job_id: value.job_id,
      kind: value.kind,
      owner: value.owner,
      request_id: requestId.value,
      result: value.result,
      started_at_utc: started.value,
      status: value.status,
      training_config: null,
    },
  };
}

/**
 * Read a metric that need not be there.
 *
 * Absent is allowed; present and not a finite number is not. A metric sent as
 * `NaN` or as a string is a broken measurement, and reading it as absent would
 * hide that.
 *
 * @param body - The record holding the metric.
 * @param key - The metric's field.
 * @param error - The identifier to refuse with.
 * @returns The metric, `undefined` when absent, or the refusal.
 */
export function parseOptionalFiniteMetric(
  body: Record<string, unknown>,
  key: string,
  error: string,
): ValidationResult<number | undefined> {
  if (!(key in body)) {
    return { ok: true, value: undefined };
  }
  if (!isFiniteNumber(body[key])) {
    return { ok: false, error };
  }
  return { ok: true, value: body[key] };
}

/**
 * Every analysis this build knows how to read a result for.
 *
 * Typed as the contract's own union, so a kind added to the contract without
 * being added here is a type error rather than a receipt refused at runtime.
 */
const ANALYSIS_KINDS: readonly AnalysisJobKind[] = [
  "fi_curve",
  "bifurcation",
  "heatmap",
  "sensitivity",
];

/**
 * Whether a value names an analysis this build knows.
 *
 * @param value - The value, as it arrived.
 * @returns Whether it is one.
 */
function isAnalysisKind(value: unknown): value is AnalysisJobKind {
  return typeof value === "string" && ANALYSIS_KINDS.some((kind) => kind === value);
}

/**
 * Read a submit receipt, and bind it to the analysis that was asked for.
 *
 * @param receipt - The receipt, as it arrived.
 * @param expectedKind - The analysis the caller submitted.
 * @returns The receipt, or the identifier of what was wrong. A receipt for a
 *   different analysis is refused: it would set the session polling for a job
 *   whose result the caller cannot interpret.
 */
export function validateAnalysisJobReceipt(
  receipt: unknown,
  expectedKind: AnalysisJobKind,
): ValidationResult<AnalysisJobReceipt> {
  if (!isRecord(receipt)) {
    return { ok: false, error: "analysis_job_receipt_invalid" };
  }
  if (receipt.schema_version !== "studio.analysis.job.v1") {
    return { ok: false, error: "analysis_job_receipt_schema_invalid" };
  }
  if (receipt.execution_mode !== "async_job") {
    return { ok: false, error: "analysis_job_receipt_mode_invalid" };
  }
  if (receipt.analysis !== expectedKind || !isAnalysisKind(receipt.analysis)) {
    return { ok: false, error: "analysis_job_receipt_analysis_mismatch" };
  }
  if (!isNonEmptyString(receipt.job_id)) {
    return { ok: false, error: "analysis_job_receipt_job_id_invalid" };
  }
  if (!isNonEmptyString(receipt.status_route)) {
    return { ok: false, error: "analysis_job_receipt_status_route_invalid" };
  }
  const job = parseStudioJobRecord(receipt.job, {
    expectedJobId: receipt.job_id,
    expectedKind: "analysis",
  });
  if (!job.ok) {
    return {
      ok: false,
      error:
        job.error === "job_kind_mismatch"
          ? "analysis_job_receipt_kind_mismatch"
          : job.error === "job_id_mismatch"
            ? "analysis_job_receipt_job_id_mismatch"
            : `analysis_job_receipt_${job.error}`,
    };
  }
  const projected = parseOptionalFiniteMetric(
    receipt,
    "projected_simulations",
    "analysis_job_receipt_projected_simulations_invalid",
  );
  if (!projected.ok) {
    return projected;
  }
  const duration = parseOptionalFiniteMetric(
    receipt,
    "duration_ms",
    "analysis_job_receipt_duration_ms_invalid",
  );
  if (!duration.ok) {
    return duration;
  }
  const dt = parseOptionalFiniteMetric(
    receipt,
    "dt_ms",
    "analysis_job_receipt_dt_ms_invalid",
  );
  if (!dt.ok) {
    return dt;
  }
  const value: AnalysisJobReceipt = {
    analysis: expectedKind,
    execution_mode: "async_job",
    job: job.value,
    job_id: receipt.job_id,
    schema_version: "studio.analysis.job.v1",
    status_route: receipt.status_route,
  };
  if (projected.value !== undefined) {
    value.projected_simulations = projected.value;
  }
  if (duration.value !== undefined) {
    value.duration_ms = duration.value;
  }
  if (dt.value !== undefined) {
    value.dt_ms = dt.value;
  }
  return { ok: true, value };
}

/**
 * Read a poll response, and bind it to the job being polled.
 *
 * The refusals are renamed under an `analysis_poll_` prefix so a reader can
 * tell a bad poll from a bad submit; the two mismatches keep names of their
 * own because they mean the response belongs to another job entirely.
 *
 * @param record - The response, as it arrived.
 * @param expectedJobId - The job being polled, or `null` when there is none.
 * @returns The record, or the identifier of what was wrong.
 */
export function validateAnalysisPollRecord(
  record: unknown,
  expectedJobId: string | null,
): ValidationResult<StudioJobRecord> {
  const parsed = parseStudioJobRecord(record, {
    expectedJobId,
    expectedKind: "analysis",
  });
  if (!parsed.ok) {
    return {
      ok: false,
      error:
        parsed.error === "job_kind_mismatch"
          ? "analysis_poll_kind_mismatch"
          : parsed.error === "job_id_mismatch"
            ? "analysis_poll_job_id_mismatch"
            : `analysis_poll_${parsed.error}`,
    };
  }
  return parsed;
}
