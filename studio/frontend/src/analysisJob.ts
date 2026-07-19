// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Analysis job session policy (outside UI)

/**
 * Pure state machine and session lifecycle for POST /api/analysis/jobs.
 * No invented progress percentages; polling stops on dispose/terminal.
 */

import type {
  AnalysisJobKind,
  AnalysisJobReceipt,
  AnalysisJobRequestBody,
  AnalysisJobResult,
  StudioJobRecord,
} from "./api/client";
import {
  validateAnalysisJobReceipt,
  validateAnalysisJobResult,
  validateAnalysisPollRecord,
} from "./analysisJobValidation";

export { validateAnalysisJobResult } from "./analysisJobValidation";

export type AnalysisJobPhase =
  | "idle"
  | "submitting"
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled"
  | "timed_out"
  | "malformed";

export interface AnalysisJobViewState {
  analysis: AnalysisJobKind | null;
  error: string | null;
  jobId: string | null;
  phase: AnalysisJobPhase;
  result: AnalysisJobResult | null;
  statusRoute: string | null;
}

export type AnalysisJobEvent =
  | { type: "submit_started"; analysis: AnalysisJobKind }
  | { type: "submit_succeeded"; receipt: AnalysisJobReceipt }
  | { type: "submit_failed"; message: string }
  | { type: "poll"; record: StudioJobRecord }
  | { type: "poll_failed"; message: string };

const BUSY: ReadonlySet<AnalysisJobPhase> = new Set([
  "submitting",
  "pending",
  "running",
]);

/**
 * Initial idle analysis-job view state.
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
 * True while submit or poll is in flight.
 */
export function isAnalysisJobBusy(phase: AnalysisJobPhase): boolean {
  return BUSY.has(phase);
}

/**
 * True when a new analysis job may be submitted.
 */
export function canSubmitAnalysisJob(state: AnalysisJobViewState): boolean {
  return !isAnalysisJobBusy(state.phase);
}

/**
 * Path-free operator label for the current real phase.
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
    case "malformed":
      return "invalid";
    default: {
      const _exhaustive: never = phase;
      return _exhaustive;
    }
  }
}

function publicErrorMessage(raw: string): string {
  const trimmed = raw.trim();
  if (trimmed.length === 0) {
    return "analysis_job_failed";
  }
  return trimmed.replace(/\/(?:home|media|tmp|var)\/[^\s"']+/g, "[path]");
}

/**
 * Reduce one analysis-job event into the next view state.
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
      if (record.status === "completed") {
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
      return {
        ...state,
        error: "analysis_job_status_unknown",
        phase: "failed",
        result: null,
      };
    }
    default: {
      const _exhaustive: never = event;
      return _exhaustive;
    }
  }
}

export interface AnalysisJobApi {
  fetchJob: (statusRoute: string) => Promise<StudioJobRecord>;
  submit: (request: AnalysisJobRequestBody) => Promise<AnalysisJobReceipt>;
}

export interface AnalysisJobSessionOptions {
  api: AnalysisJobApi;
  clearTimeoutFn?: typeof clearTimeout;
  onChange?: (state: AnalysisJobViewState) => void;
  pollIntervalMs?: number;
  setTimeoutFn?: typeof setTimeout;
}

export interface AnalysisJobSession {
  dispose: () => void;
  getState: () => AnalysisJobViewState;
  startJob: (request: AnalysisJobRequestBody) => Promise<void>;
}

/**
 * Create a non-React analysis-job session (submit + poll to terminal).
 */
export function createAnalysisJobSession(
  options: AnalysisJobSessionOptions,
): AnalysisJobSession {
  const pollIntervalMs = options.pollIntervalMs ?? 500;
  const setTimeoutFn = options.setTimeoutFn ?? setTimeout;
  const clearTimeoutFn = options.clearTimeoutFn ?? clearTimeout;
  let state = initialAnalysisJobState();
  let disposed = false;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let generation = 0;

  const publish = (next: AnalysisJobViewState) => {
    state = next;
    options.onChange?.(state);
  };

  const apply = (event: AnalysisJobEvent) => {
    if (disposed) {
      return;
    }
    publish(reduceAnalysisJob(state, event));
  };

  const stopPolling = () => {
    if (timer !== null) {
      clearTimeoutFn(timer);
      timer = null;
    }
  };

  const schedulePoll = (gen: number, statusRoute: string) => {
    stopPolling();
    timer = setTimeoutFn(() => {
      void (async () => {
        if (disposed || gen !== generation) {
          return;
        }
        try {
          const record = await options.api.fetchJob(statusRoute);
          if (disposed || gen !== generation) {
            return;
          }
          apply({ type: "poll", record });
          if (state.phase === "pending" || state.phase === "running") {
            schedulePoll(gen, statusRoute);
          } else {
            stopPolling();
          }
        } catch (error: unknown) {
          if (disposed || gen !== generation) {
            return;
          }
          const message =
            error instanceof Error ? error.message : "analysis_poll_failed";
          apply({ type: "poll_failed", message });
          stopPolling();
        }
      })();
    }, pollIntervalMs);
  };

  return {
    dispose: () => {
      disposed = true;
      generation += 1;
      stopPolling();
    },
    getState: () => state,
    startJob: async (request) => {
      if (disposed || !canSubmitAnalysisJob(state)) {
        return;
      }
      generation += 1;
      const gen = generation;
      stopPolling();
      apply({ type: "submit_started", analysis: request.analysis });
      try {
        const receipt = await options.api.submit(request);
        if (disposed || gen !== generation) {
          return;
        }
        apply({ type: "submit_succeeded", receipt });
        if (state.phase === "pending" || state.phase === "running") {
          const route = state.statusRoute;
          if (route !== null) {
            try {
              const record = await options.api.fetchJob(route);
              if (disposed || gen !== generation) {
                return;
              }
              apply({ type: "poll", record });
              if (state.phase === "pending" || state.phase === "running") {
                schedulePoll(gen, route);
              }
            } catch (error: unknown) {
              if (disposed || gen !== generation) {
                return;
              }
              const message =
                error instanceof Error ? error.message : "analysis_poll_failed";
              apply({ type: "poll_failed", message });
            }
          }
        }
      } catch (error: unknown) {
        if (disposed || gen !== generation) {
          return;
        }
        const message =
          error instanceof Error ? error.message : "analysis_submit_failed";
        apply({ type: "submit_failed", message });
      }
    },
  };
}
