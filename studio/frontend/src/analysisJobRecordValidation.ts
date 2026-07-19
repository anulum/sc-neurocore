// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Strict StudioJobRecord envelope parsing (fail-closed)

/**
 * Parse untrusted job envelopes without defaults, silent drops, or unchecked
 * casts. Malformed required, nullable, optional, or artifact fields reject.
 */

import type {
  AnalysisJobKind,
  AnalysisJobReceipt,
  StudioJobArtifact,
  StudioJobRecord,
} from "./api/client";

export type ValidationResult<T> =
  | { ok: true; value: T }
  | { ok: false; error: string };

const HEX64 = /^[0-9a-fA-F]{64}$/;
const JOB_STATUSES = new Set([
  "pending",
  "running",
  "completed",
  "failed",
  "cancelling",
  "cancelled",
  "timed_out",
]);

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === "string" && value.length > 0;
}

function isSha256Hex(value: unknown): value is string {
  return typeof value === "string" && HEX64.test(value);
}

function isNonNegativeInt(value: unknown): value is number {
  return isFiniteNumber(value) && Number.isInteger(value) && value >= 0;
}

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
 * Guard one job artifact entry.
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
 * Parse a full StudioJobRecord envelope fail-closed.
 *
 * @param expectedKind - When set, require exact job kind (e.g. ``analysis``).
 * @param expectedJobId - When set, require exact job id binding.
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
  if (typeof value.status !== "string" || !JOB_STATUSES.has(value.status)) {
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
      result: value.result === null ? null : value.result,
      started_at_utc: started.value,
      status: value.status as StudioJobRecord["status"],
    },
  };
}

/**
 * Parse optional finite metrics: absent ok; present must be finite numbers.
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

const ANALYSIS_KINDS = new Set<AnalysisJobKind>([
  "fi_curve",
  "bifurcation",
  "heatmap",
  "sensitivity",
]);

/**
 * Validate an analysis job submit receipt and bind kind/id with a strict job envelope.
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
  if (
    receipt.analysis !== expectedKind
    || !ANALYSIS_KINDS.has(receipt.analysis as AnalysisJobKind)
  ) {
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
 * Bind a polled job record to the retained analysis job id and kind.
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
