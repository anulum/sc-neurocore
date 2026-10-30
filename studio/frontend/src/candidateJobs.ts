// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Candidate job identity and result admission

import { parseStudioJobRecord } from "./analysisJobRecordValidation";
import type { CandidateReviewPacket, CandidateRun } from "./api/candidatesApi";
import type { StudioJobRecord } from "./api/types";

/** The two supported scientific operations. */
export type CandidateOperation = "simulate" | "review";

/** Immutable scientific input identity retained by the ledger. */
export interface CandidateJobAdmission {
  candidate_operation: CandidateOperation;
  candidate_name: string;
  candidate_sha256: string;
  request_sha256: string;
  projected_steps: number;
}

/** An admitted owner-scoped record, including a checked scientific result. */
export interface CandidateJob extends Omit<StudioJobRecord, "result"> {
  admission: CandidateJobAdmission;
  experiment_sha256: string;
  result: CandidateRun | CandidateReviewPacket | null;
}

/** A submission receipt bound to its embedded job and status route. */
export interface CandidateJobReceipt {
  job: CandidateJob;
  status_route: string;
}

/**
 * Determine whether an untrusted value is a JSON object.
 *
 * @param value - Untrusted JSON field.
 * @returns Whether the complete value satisfies this field contract.
 */
function record(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Check a finite scalar.
 *
 * @param value - Untrusted JSON field.
 * @returns Whether the complete value satisfies this field contract.
 */
function finite(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

/**
 * Check a bounded-count representation.
 *
 * @param value - Untrusted JSON field.
 * @returns Whether the complete value satisfies this field contract.
 */
function count(value: unknown): value is number {
  return finite(value) && Number.isSafeInteger(value) && value >= 0;
}

/**
 * Check a lower-case SHA-256 representation.
 *
 * @param value - Untrusted JSON field.
 * @returns Whether the complete value satisfies this field contract.
 */
function digest(value: unknown): value is string {
  return typeof value === "string" && /^[0-9a-f]{64}$/.test(value);
}

/**
 * Admit every simulation field displayed or exported as numerical evidence.
 *
 * @param value - Untrusted JSON field.
 * @returns Whether the complete value satisfies this field contract.
 */
function run(value: Record<string, unknown>): boolean {
  return typeof value.candidate === "string" && digest(value.candidate_sha256)
    && record(value.units) && typeof value.units.current === "string" && typeof value.units.time === "string"
    && count(value.steps) && value.steps > 0 && finite(value.current)
    && record(value.profile) && typeof value.profile.method === "string"
    && finite(value.profile.dt) && value.profile.dt > 0 && typeof value.profile.time_unit === "string"
    && count(value.spike_count) && Array.isArray(value.spike_steps) && value.spike_steps.every(count)
    && value.spike_count === value.spike_steps.length
    && (value.diverged_at_step === null
      ? value.divergence === null && record(value.final_state) && Object.values(value.final_state).every(finite)
      : count(value.diverged_at_step) && typeof value.divergence === "string" && value.final_state === null)
    && count(value.sample_every) && value.sample_every > 0
    && record(value.trace) && Object.values(value.trace).every((points) => Array.isArray(points) && points.every(finite));
}

/**
 * Admit each reference check and the packet's explicit limits.
 *
 * @param value - Untrusted JSON field.
 * @returns Whether the complete value satisfies this field contract.
 */
function review(value: Record<string, unknown>): boolean {
  return value.schema_version === "sc-neurocore.studio.candidate-review.v1"
    && digest(value.candidate_sha256) && digest(value.packet_sha256)
    && typeof value.reference_tests_passed === "boolean"
    && Array.isArray(value.not_established) && value.not_established.every((item) => typeof item === "string")
    && Array.isArray(value.reference_tests) && value.reference_tests.every((test: unknown) =>
      record(test) && typeof test.name === "string" && typeof test.passed === "boolean"
      && (test.diverged_at_step === null || count(test.diverged_at_step))
      && Array.isArray(test.checks) && test.checks.every((check: unknown) =>
        record(check) && typeof check.quantity === "string" && (check.observed === null || finite(check.observed))
        && typeof check.held === "boolean"));
}

/**
 * Select the complete result grammar associated with the admitted operation.
 *
 * @param value - Untrusted scientific result.
 * @param operation - Operation retained by the checked admission.
 * @returns Whether the result matches that operation's supported shape.
 */
function scientificResult(value: Record<string, unknown>, operation: CandidateOperation): boolean {
  return operation === "simulate" ? run(value) : review(value);
}

/**
 * Validate a ledger response before using its identity or numerical result.
 *
 * @param value - Untrusted HTTP response.
 * @param expectedId - Job requested by the caller, when observing one job.
 * @param summary - History intentionally omits potentially large results.
 * @returns The checked record.
 * @throws {Error} When identity, operation, state or scientific payload is inconsistent.
 */
export function parseCandidateJob(value: unknown, expectedId?: string, summary = false): CandidateJob {
  const base = parseStudioJobRecord(value, { expectedJobId: expectedId, expectedKind: "analysis" });
  if (!base.ok || !record(value) || !/^sj_[0-9a-f]{16}$/.test(base.value.job_id)) {
    throw new Error("Invalid candidate job record.");
  }
  const admission = value.admission;
  if (!record(admission) || (admission.candidate_operation !== "simulate" && admission.candidate_operation !== "review")
    || typeof admission.candidate_name !== "string" || admission.candidate_name.length === 0
    || !digest(admission.candidate_sha256) || !digest(admission.request_sha256)
    || !count(admission.projected_steps) || value.experiment_sha256 !== admission.request_sha256
    || base.value.execution_model !== "process") {
    throw new Error("Invalid candidate job input identity.");
  }
  const result = base.value.result;
  if (result !== null && (!scientificResult(result, admission.candidate_operation)
    || result.candidate_sha256 !== admission.candidate_sha256)) {
    throw new Error("Invalid candidate job scientific result.");
  }
  if (!summary && base.value.status === "completed" && result === null) {
    throw new Error("Completed candidate job has no result.");
  }
  return {
    ...base.value,
    admission: {
      candidate_operation: admission.candidate_operation,
      candidate_name: admission.candidate_name,
      candidate_sha256: admission.candidate_sha256,
      request_sha256: admission.request_sha256,
      projected_steps: admission.projected_steps,
    },
    experiment_sha256: admission.request_sha256,
    result: result as CandidateJob["result"],
  };
}

/**
 * Check a submission receipt's operation, hashes and embedded record together.
 *
 * @param value - Untrusted submission response.
 * @param operation - Operation submitted by the caller.
 * @returns A receipt safe to observe without resubmitting the calculation.
 */
export function parseCandidateJobReceipt(value: unknown, operation: CandidateOperation): CandidateJobReceipt {
  if (!record(value) || value.schema_version !== "studio.candidate.job.v1") throw new Error("Invalid candidate job receipt.");
  const job = parseCandidateJob(value.job);
  const identity = job.admission;
  const route = `/api/candidates/jobs/${job.job_id}`;
  if (value.operation !== operation || identity.candidate_operation !== operation || value.job_id !== job.job_id
    || value.candidate_sha256 !== identity.candidate_sha256 || value.request_sha256 !== identity.request_sha256
    || value.projected_steps !== identity.projected_steps || value.status_route !== route) {
    throw new Error("Candidate job receipt does not match its input identity.");
  }
  return { job, status_route: route };
}

/**
 * Validate a bounded history response, retaining the server's newest-first order.
 *
 * @param value - Untrusted history response.
 * @returns Checked records without full result arrays.
 */
export function parseCandidateJobHistory(value: unknown): CandidateJob[] {
  if (!record(value) || value.schema_version !== "studio.candidate.jobs.v1" || !Array.isArray(value.jobs)
    || value.jobs.length > 100) throw new Error("Invalid candidate job history.");
  return value.jobs.map((job: unknown) => parseCandidateJob(job, undefined, true));
}

/**
 * Report whether polling can still observe a live transition.
 *
 * @param job - Checked ledger observation.
 * @returns Whether the supervisor can still change the live execution state.
 */
export function candidateJobActive(job: CandidateJob): boolean {
  return ["pending", "running", "cancelling"].includes(job.status);
}

/**
 * Keep successive observations bound to the input admitted for this job.
 *
 * @param expected - Original receipt or stored history identity.
 * @param observed - Complete independently parsed status or cancellation response.
 * @returns The response after comparing every immutable identity field.
 * @throws {Error} When a response would relabel the selected calculation.
 */
export function bindCandidateJobObservation(expected: CandidateJob, observed: CandidateJob): CandidateJob {
  const before = expected.admission;
  const after = observed.admission;
  if (expected.job_id !== observed.job_id || expected.owner !== observed.owner
    || expected.created_at_utc !== observed.created_at_utc || expected.request_id !== observed.request_id
    || before.candidate_operation !== after.candidate_operation || before.candidate_name !== after.candidate_name
    || before.candidate_sha256 !== after.candidate_sha256 || before.request_sha256 !== after.request_sha256
    || before.projected_steps !== after.projected_steps) {
    throw new Error("Candidate job observation changed its admitted input identity.");
  }
  return observed;
}
