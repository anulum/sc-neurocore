// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Analysis job timer/API session lifecycle

/**
 * Submit + poll lifecycle for analysis jobs. Reducer/view policy stays in
 * analysisJob.ts; this module only owns timers, generation, and dispose.
 */

import type {
  AnalysisJobReceipt,
  AnalysisJobRequestBody,
  StudioJobRecord,
} from "./api/client";
import {
  canSubmitAnalysisJob,
  initialAnalysisJobState,
  reduceAnalysisJob,
  type AnalysisJobViewState,
} from "./analysisJob";

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

  const apply = (
    event: Parameters<typeof reduceAnalysisJob>[1],
  ) => {
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
