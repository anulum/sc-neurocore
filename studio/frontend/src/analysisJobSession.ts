// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Analysis job timer/API session lifecycle

/**
 * The submit-and-poll loop for one analysis job.
 *
 * Two things make this more than a loop. **Generations**: every start bumps a
 * counter, and a response that comes back under an old generation is dropped,
 * so a reader who starts a second job never sees the first one's result
 * arrive. **Disposal**: the session can be thrown away mid-flight, and every
 * point after an `await` checks that before publishing anything.
 *
 * Polling stops as soon as the job stops. There is no maximum attempt count
 * and no backoff: the server's own timeout is what ends a job that never
 * finishes, and inventing a second, shorter deadline here would report a
 * running job as failed.
 *
 * What to show is decided in `analysisJob.ts`; this module owns only timers,
 * generations and disposal.
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

/** The two routes a session needs: submit one job, then ask about it. */
export interface AnalysisJobApi {
  fetchJob: (statusRoute: string) => Promise<StudioJobRecord>;
  submit: (request: AnalysisJobRequestBody) => Promise<AnalysisJobReceipt>;
}

/**
 * What a session may be given instead of its defaults. The timer functions are
 * injectable so a test can run a poll loop without waiting for it.
 */
export interface AnalysisJobSessionOptions {
  api: AnalysisJobApi;
  clearTimeoutFn?: typeof clearTimeout;
  onChange?: (state: AnalysisJobViewState) => void;
  pollIntervalMs?: number;
  setTimeoutFn?: typeof setTimeout;
}

/**
 * A running session: read its state, start a job, or throw it away. Disposing
 * stops the timers and silences every response still in flight.
 */
export interface AnalysisJobSession {
  dispose: () => void;
  getState: () => AnalysisJobViewState;
  startJob: (request: AnalysisJobRequestBody) => Promise<void>;
}

/**
 * Start a session that can run one analysis job at a time.
 *
 * @param options - The API, the poll interval, where to report state, and the
 *   timer functions -- replaceable so a test can drive the clock.
 * @returns The session. `startJob` resolves when the job has been submitted
 *   and its first poll answered, not when the job finishes; the rest arrives
 *   through `onChange`.
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

  // `disposed` and `state` are read back through these two functions rather
  // than directly. Both are reassigned from outside the awaiting code -- by
  // `dispose`, and by every `apply` -- so a direct read is narrowed by the
  // checker to whatever it was before the `await`, and the guards that follow
  // an await look dead when they are the only thing keeping a disposed session
  // from publishing. A call returning a declared type is not narrowed, which
  // is what makes those guards readable as the live code they are.
  const isDisposed = (): boolean => disposed;
  const readState = (): AnalysisJobViewState => state;

  const publish = (next: AnalysisJobViewState) => {
    state = next;
    options.onChange?.(state);
  };

  const apply = (
    event: Parameters<typeof reduceAnalysisJob>[1],
  ) => {
    if (isDisposed()) {
      return;
    }
    publish(reduceAnalysisJob(readState(), event));
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
        if (isDisposed() || gen !== generation) {
          return;
        }
        try {
          const record = await options.api.fetchJob(statusRoute);
          if (isDisposed() || gen !== generation) {
            return;
          }
          apply({ type: "poll", record });
          const polled = readState();
          if (polled.phase === "pending" || polled.phase === "running") {
            schedulePoll(gen, statusRoute);
          } else {
            stopPolling();
          }
        } catch (error: unknown) {
          if (isDisposed() || gen !== generation) {
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
    getState: readState,
    startJob: async (request) => {
      if (isDisposed() || !canSubmitAnalysisJob(readState())) {
        return;
      }
      generation += 1;
      const gen = generation;
      stopPolling();
      apply({ type: "submit_started", analysis: request.analysis });
      try {
        const receipt = await options.api.submit(request);
        if (isDisposed() || gen !== generation) {
          return;
        }
        apply({ type: "submit_succeeded", receipt });
        const submitted = readState();
        if (submitted.phase === "pending" || submitted.phase === "running") {
          const route = submitted.statusRoute;
          if (route !== null) {
            try {
              const record = await options.api.fetchJob(route);
              if (isDisposed() || gen !== generation) {
                return;
              }
              apply({ type: "poll", record });
              const afterFirstPoll = readState();
              if (
                afterFirstPoll.phase === "pending"
                || afterFirstPoll.phase === "running"
              ) {
                schedulePoll(gen, route);
              }
            } catch (error: unknown) {
              if (isDisposed() || gen !== generation) {
                return;
              }
              const message =
                error instanceof Error ? error.message : "analysis_poll_failed";
              apply({ type: "poll_failed", message });
            }
          }
        }
      } catch (error: unknown) {
        if (isDisposed() || gen !== generation) {
          return;
        }
        const message =
          error instanceof Error ? error.message : "analysis_submit_failed";
        apply({ type: "submit_failed", message });
      }
    },
  };
}
