// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Analysis job session policy (outside UI)

/**
 * What an analysis job looks like to the reader, as a state machine.
 *
 * A job is submitted, then polled until it stops. This module owns only the
 * question "given where we were and what just happened, what should be on
 * screen"; timers, requests and cancellation live in `analysisJobSession.ts`,
 * and the answer's shape is checked in `analysisJobValidation.ts`.
 *
 * The phases separate two kinds of ending that a single `failed` would blur.
 * `failed`, `cancelled` and `timed_out` are the server's verdicts on the job.
 * `malformed` is this build's verdict on the *answer*: the job succeeded and
 * returned something that is not the analysis it claimed to be. An operator
 * needs to know which of those happened, because one is a job to rerun and the
 * other is a server to fix.
 *
 * Every error message the reducer stores passes through `publicErrorMessage`,
 * so a server error quoting a filesystem path does not put that path on
 * screen.
 */

import type {
  AnalysisJobKind,
  AnalysisJobReceipt,
  AnalysisJobResult,
  StudioJobRecord,
} from "./api/client";
import {
  validateAnalysisJobReceipt,
  validateAnalysisJobResult,
  validateAnalysisPollRecord,
} from "./analysisJobValidation";

export { validateAnalysisJobResult } from "./analysisJobValidation";
export {
  createAnalysisJobSession,
  type AnalysisJobApi,
  type AnalysisJobSession,
  type AnalysisJobSessionOptions,
} from "./analysisJobSession";

/**
 * Where a job is. The three terminal failures are distinct on purpose:
 * `failed`, `cancelled` and `timed_out` are the server's verdicts on the job,
 * and `malformed` is this build's verdict on the answer it returned.
 */
export type AnalysisJobPhase =
  | "idle"
  | "submitting"
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled"
  | "timed_out"
  | "interrupted"
  | "unknown"
  | "malformed";

/**
 * Everything the view needs about a job: which analysis it is, where it is,
 * what it returned, and where to ask again.
 */
export interface AnalysisJobViewState {
  analysis: AnalysisJobKind | null;
  error: string | null;
  jobId: string | null;
  phase: AnalysisJobPhase;
  result: AnalysisJobResult | null;
  statusRoute: string | null;
}

/** Everything that can happen to a job while the reader watches it. */
export type AnalysisJobEvent =
  | { type: "submit_started"; analysis: AnalysisJobKind }
  | { type: "submit_succeeded"; receipt: AnalysisJobReceipt }
  | { type: "submit_failed"; message: string }
  | { type: "poll"; record: StudioJobRecord }
  | { type: "poll_failed"; message: string };

/** The phases in which a job is still going. */
const BUSY: ReadonlySet<AnalysisJobPhase> = new Set([
  "submitting",
  "pending",
  "running",
  "unknown",
]);

/**
 * The state before anything has been submitted.
 *
 * @returns The idle state.
 */
export function initialAnalysisJobState(): AnalysisJobViewState {
  return {
    analysis: null,
    error: null,
    jobId: null,
    phase: "idle",
    result: null,
    statusRoute: null,
  };
}

/**
 * Whether the job is still going.
 *
 * @param phase - The phase.
 * @returns Whether a submit or a poll is in flight.
 */
export function isAnalysisJobBusy(phase: AnalysisJobPhase): boolean {
  return BUSY.has(phase);
}

/**
 * Whether a new job may be started.
 *
 * A finished job -- however it finished -- may be replaced. Only a job still
 * running blocks another, which is what stops a double-click submitting twice.
 *
 * @param state - The current state.
 * @returns Whether a submit is allowed.
 */
export function canSubmitAnalysisJob(state: AnalysisJobViewState): boolean {
  return !isAnalysisJobBusy(state.phase);
}

/**
 * Name a phase for the reader.
 *
 * `malformed` is shown as `invalid`, because "malformed" describes the
 * response and the reader is being told about their run.
 *
 * @param phase - The phase.
 * @returns Its label.
 */
export function analysisJobPhaseLabel(phase: AnalysisJobPhase): string {
  switch (phase) {
    case "idle":
      return "idle";
    case "submitting":
      return "submitting";
    case "pending":
      return "pending";
    case "running":
      return "running";
    case "completed":
      return "completed";
    case "failed":
      return "failed";
    case "cancelled":
      return "cancelled";
    case "timed_out":
      return "timed_out";
    case "interrupted":
      return "interrupted";
    case "unknown":
      return "unknown";
    case "malformed":
      return "invalid";
    default: {
      const _exhaustive: never = phase;
      return _exhaustive;
    }
  }
}

/**
 * Reduce a server error to something safe to show.
 *
 * Absolute paths under the deployment's own directories are replaced with
 * `[path]`: an error message is written for an operator reading a log, and the
 * browser shows it to whoever opened the panel.
 *
 * @param raw - The server's message.
 * @returns The message to show, or a generic identifier when it was blank.
 */
function publicErrorMessage(raw: string): string {
  const trimmed = raw.trim();
  if (trimmed.length === 0) {
    return "analysis_job_failed";
  }
  return trimmed.replace(/\/(?:home|media|tmp|var)\/[^\s"']+/g, "[path]");
}

/**
 * Apply one event to the state.
 *
 * @param state - Where the job was.
 * @param event - What happened.
 * @returns Where the job is now. Every returned state is complete: a
 *   transition that clears the result says so rather than leaving the previous
 *   analysis's numbers beside a new phase.
 */
export function reduceAnalysisJob(
  state: AnalysisJobViewState,
  event: AnalysisJobEvent,
): AnalysisJobViewState {
  switch (event.type) {
    case "submit_started":
      if (!canSubmitAnalysisJob(state)) {
        return state;
      }
      return {
        analysis: event.analysis,
        error: null,
        jobId: null,
        phase: "submitting",
        result: null,
        statusRoute: null,
      };
    case "submit_failed":
      return {
        ...state,
        error: publicErrorMessage(event.message),
        jobId: null,
        phase: "failed",
        result: null,
        statusRoute: null,
      };
    case "submit_succeeded": {
      const expected = state.analysis;
      if (expected === null) {
        return {
          analysis: null,
          error: "analysis_job_session_kind_missing",
          jobId: null,
          phase: "malformed",
          result: null,
          statusRoute: null,
        };
      }
      const validated = validateAnalysisJobReceipt(event.receipt, expected);
      if (!validated.ok) {
        return {
          analysis: expected,
          error: validated.error,
          jobId: null,
          phase: "malformed",
          result: null,
          statusRoute: null,
        };
      }
      const receipt = validated.value;
      const phase: AnalysisJobPhase =
        receipt.job.status === "running" ? "running" : "pending";
      return {
        analysis: expected,
        error: null,
        jobId: receipt.job_id,
        phase,
        result: null,
        statusRoute: receipt.status_route,
      };
    }
    case "poll_failed":
      return {
        ...state,
        error: publicErrorMessage(event.message),
        phase: "failed",
        result: null,
      };
    case "poll": {
      const bound = validateAnalysisPollRecord(event.record, state.jobId);
      if (!bound.ok) {
        return {
          ...state,
          error: bound.error,
          phase: "malformed",
          result: null,
        };
      }
      const record = bound.value;
      if (record.status === "pending") {
        return { ...state, error: null, phase: "pending" };
      }
      if (record.status === "running" || record.status === "cancelling") {
        return { ...state, error: null, phase: "running" };
      }
      if (record.status === "failed") {
        return {
          ...state,
          error: publicErrorMessage(record.error ?? "analysis_job_failed"),
          phase: "failed",
          result: null,
        };
      }
      if (record.status === "cancelled") {
        return {
          ...state,
          error: "analysis_job_cancelled",
          phase: "cancelled",
          result: null,
        };
      }
      if (record.status === "timed_out") {
        return {
          ...state,
          error: "analysis_job_timed_out",
          phase: "timed_out",
          result: null,
        };
      }
      if (record.status === "interrupted" || record.status === "unknown") {
        return {
          ...state,
          error: `analysis_job_${record.status}`,
          phase: record.status,
          result: null,
        };
      }
      // Every other status has returned by now, so the record is completed.
      // There used to be a further `status === "completed"` test and an
      // `analysis_job_status_unknown` fallback beneath it. The fallback was
      // unreachable: the poll record is validated against the contract's own
      // status list before it gets here, so an unknown status is refused there
      // as `job_status_invalid` and never reaches this reducer.
      if (state.analysis === null) {
        return {
          ...state,
          error: "analysis_job_session_kind_missing",
          phase: "malformed",
          result: null,
        };
      }
      const validated = validateAnalysisJobResult(record.result, state.analysis);
      if (!validated.ok) {
        return {
          ...state,
          error: validated.error,
          phase: "malformed",
          result: null,
        };
      }
      return {
        ...state,
        error: null,
        phase: "completed",
        result: validated.value,
      };
    }
    default: {
      const _exhaustive: never = event;
      return _exhaustive;
    }
  }
}
