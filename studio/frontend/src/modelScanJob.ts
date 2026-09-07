// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Model Browser async model-scan job policy (outside UI)

/**
 * Scanning the whole model catalogue, as a state machine.
 *
 * A scan runs every model in the catalogue and reports how each behaves. It
 * takes long enough to be a job rather than a request, so the shape here is
 * the same as the analysis job's: submit, poll, stop when it stops.
 *
 * The phases separate the server's verdict on the job -- `failed`,
 * `cancelled`, `timed_out` -- from this build's verdict on the answer,
 * `malformed`. A scan that completes and returns something that is not a scan
 * result is not a failed scan; it is a server to fix, and the reader should be
 * told which they have.
 *
 * A scan's own failures are not job failures. A model that could not be run is
 * reported inside the result as a failed model with its reason, and the job
 * still completes: one broken model in a catalogue of hundreds should not cost
 * the reader the other results.
 */

import type {
  ModelBehavior,
  ModelScanJobReceipt,
  ModelScanMetadata,
  ModelScanResponse,
  StudioJobRecord,
} from "./api/client";
import {
  validateModelScanJobReceipt,
  validateModelScanJobResult,
  validateModelScanPollRecord,
} from "./modelScanJobValidation";

export { validateModelScanJobResult } from "./modelScanJobValidation";

/** Terminal and in-flight phases shown to the operator (no invented progress %). */
/**
 * Where a scan is. `failed`, `cancelled` and `timed_out` are the server's
 * verdicts on the job; `malformed` is this build's verdict on the answer.
 */
export type ModelScanJobPhase =
  | "idle"
  | "submitting"
  | "pending"
  | "running"
  | "completed"
  | "failed"
  | "cancelled"
  | "timed_out"
  | "malformed";

/** Everything the control needs: where the scan is, and what it found. */
export interface ModelScanJobViewState {
  behaviors: Record<string, ModelBehavior>;
  error: string | null;
  jobId: string | null;
  phase: ModelScanJobPhase;
  scanMetadata: ModelScanMetadata | null;
  statusRoute: string | null;
}

/** Everything that can happen to a scan while the reader watches it. */
export type ModelScanJobEvent =
  | { type: "submit_started" }
  | { type: "submit_succeeded"; receipt: ModelScanJobReceipt }
  | { type: "submit_failed"; message: string }
  | { type: "poll"; record: StudioJobRecord }
  | { type: "poll_failed"; message: string };

/** The phases in which a scan is still going. */
const BUSY_PHASES: ReadonlySet<ModelScanJobPhase> = new Set([
  "submitting",
  "pending",
  "running",
]);

/**
 * The state before any scan has been started.
 *
 * @returns The idle state.
 */
export function initialModelScanJobState(): ModelScanJobViewState {
  return {
    behaviors: {},
    error: null,
    jobId: null,
    phase: "idle",
    scanMetadata: null,
    statusRoute: null,
  };
}

/**
 * Whether a scan is still going.
 *
 * @param phase - The phase.
 * @returns Whether a submit or a poll is in flight. This is what stops a
 *   second click starting a second scan of the whole catalogue.
 */
export function isModelScanJobBusy(phase: ModelScanJobPhase): boolean {
  return BUSY_PHASES.has(phase);
}

/**
 * Whether a new scan may be started.
 *
 * @param state - The current state.
 * @returns Whether the control is free. A finished scan -- however it finished
 *   -- may be replaced.
 */
export function canSubmitModelScanJob(state: ModelScanJobViewState): boolean {
  return !isModelScanJobBusy(state.phase);
}

/**
 * Name a phase for the control's button.
 *
 * The idle label is `Scan` rather than `idle`, because the control is a button
 * and its label is what it will do, not what it is doing.
 *
 * @param phase - The phase.
 * @returns Its label.
 */
export function modelScanJobPhaseLabel(phase: ModelScanJobPhase): string {
  switch (phase) {
    case "idle":
      return "Scan";
    case "submitting":
      return "submitting";
    case "pending":
      return "pending";
    case "running":
      return "running";
    case "completed":
      return "Scanned";
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

/**
 * Reduce a server error to something safe to show.
 *
 * Absolute paths under the deployment's own directories are replaced with
 * `[path]`: the message was written for an operator reading a log, and the
 * browser shows it to whoever opened the panel.
 *
 * @param raw - The server's message.
 * @returns The message to show, or a generic identifier when it was blank.
 */
function publicErrorMessage(raw: string): string {
  const trimmed = raw.trim();
  if (trimmed.length === 0) {
    return "model_scan_failed";
  }
  // Strip absolute/home paths if a backend ever leaks them.
  return trimmed.replace(/\/(?:home|media|tmp|var)\/[^\s"']+/g, "[path]");
}

/**
 * Index a scan's models by name, for the browser to look them up.
 *
 * @param payload - The validated scan result.
 * @returns The models, keyed by name.
 */
function behaviorsFromPayload(payload: ModelScanResponse): Record<string, ModelBehavior> {
  const map: Record<string, ModelBehavior> = {};
  // No guard on the entries: `validateModelScanJobResult` has already refused
  // a payload whose models are not models, so a check here would be a second
  // opinion on a question already settled.
  for (const model of payload.models) {
    map[model.name] = model;
  }
  return map;
}

/**
 * Apply one event to the state.
 *
 * @param state - Where the scan was.
 * @param event - What happened.
 * @returns Where the scan is now. A transition that clears the results
 *   says so, rather than leaving the previous scan's models beside a new
 *   phase.
 */
export function reduceModelScanJob(
  state: ModelScanJobViewState,
  event: ModelScanJobEvent,
): ModelScanJobViewState {
  switch (event.type) {
    case "submit_started":
      if (!canSubmitModelScanJob(state)) {
        return state;
      }
      return {
        behaviors: {},
        error: null,
        jobId: null,
        phase: "submitting",
        scanMetadata: null,
        statusRoute: null,
      };
    case "submit_failed":
      return {
        ...state,
        behaviors: {},
        error: publicErrorMessage(event.message),
        jobId: null,
        phase: "failed",
        scanMetadata: null,
        statusRoute: null,
      };
    case "submit_succeeded": {
      const validatedReceipt = validateModelScanJobReceipt(event.receipt);
      if (!validatedReceipt.ok) {
        return {
          behaviors: {},
          error: validatedReceipt.error,
          jobId: null,
          phase: "malformed",
          scanMetadata: null,
          statusRoute: null,
        };
      }
      const receipt = validatedReceipt.value;
      const initialStatus = receipt.job.status;
      const phase: ModelScanJobPhase =
        initialStatus === "running" ? "running" : "pending";
      return {
        behaviors: {},
        error: null,
        jobId: receipt.job_id,
        phase,
        scanMetadata: null,
        statusRoute: receipt.status_route,
      };
    }
    case "poll_failed":
      return {
        ...state,
        behaviors: {},
        error: publicErrorMessage(event.message),
        phase: "failed",
        scanMetadata: null,
      };
    case "poll": {
      const bound = validateModelScanPollRecord(event.record, state.jobId);
      if (!bound.ok) {
        return {
          ...state,
          behaviors: {},
          error: bound.error,
          phase: "malformed",
          scanMetadata: null,
        };
      }
      const record = bound.value;
      const status = record.status;
      if (status === "pending") {
        return { ...state, phase: "pending", error: null };
      }
      if (status === "running" || status === "cancelling") {
        return { ...state, phase: "running", error: null };
      }
      if (status === "failed") {
        return {
          ...state,
          behaviors: {},
          error: publicErrorMessage(record.error ?? "model_scan_job_failed"),
          phase: "failed",
          scanMetadata: null,
        };
      }
      if (status === "cancelled") {
        return {
          ...state,
          behaviors: {},
          error: "model_scan_job_cancelled",
          phase: "cancelled",
          scanMetadata: null,
        };
      }
      if (status === "timed_out") {
        return {
          ...state,
          behaviors: {},
          error: "model_scan_job_timed_out",
          phase: "timed_out",
          scanMetadata: null,
        };
      }
      // Every other status has returned by now, so the record is completed.
      // There used to be a further `status === "completed"` test and a
      // `model_scan_job_status_unknown` fallback beneath it. The fallback was
      // unreachable: the poll record is validated against the contract's own
      // status list before it gets here, so an unknown status is refused there
      // and never reaches this reducer.
      const validated = validateModelScanJobResult(record.result);
      if (!validated.ok) {
        return {
          ...state,
          behaviors: {},
          error: validated.error,
          phase: "malformed",
          scanMetadata: null,
        };
      }
      return {
        ...state,
        behaviors: behaviorsFromPayload(validated.value),
        error: null,
        phase: "completed",
        scanMetadata: validated.value.scan_metadata,
      };
    }
    default: {
      const _exhaustive: never = event;
      return _exhaustive;
    }
  }
}

/** The two routes a scan needs: submit one, then ask about it. */
export interface ModelScanJobApi {
  fetchJob: (statusRoute: string) => Promise<StudioJobRecord>;
  submit: () => Promise<ModelScanJobReceipt>;
}

/**
 * What a session may be given instead of its defaults. The timer functions are
 * injectable so a test can run a poll loop without waiting for it.
 */
export interface ModelScanJobSessionOptions {
  api: ModelScanJobApi;
  clearTimeoutFn?: typeof clearTimeout;
  onChange?: (state: ModelScanJobViewState) => void;
  pollIntervalMs?: number;
  setTimeoutFn?: typeof setTimeout;
}

/**
 * A running session: read its state, start a scan, or throw it away. Disposing
 * stops the timers and silences every response still in flight.
 */
export interface ModelScanJobSession {
  dispose: () => void;
  getState: () => ModelScanJobViewState;
  startScan: () => Promise<void>;
}

/**
 * Start a session that can run one catalogue scan at a time.
 *
 * Generations are how a stale answer is dropped: every start bumps a
 * counter, and a response returning under an old one is discarded, so a
 * reader who starts a second scan never sees the first one's results
 * arrive.
 *
 * @param options - The API, the poll interval, where to report state, and
 *   the timer functions.
 * @returns The session.
 */
export function createModelScanJobSession(
  options: ModelScanJobSessionOptions,
): ModelScanJobSession {
  const pollIntervalMs = options.pollIntervalMs ?? 500;
  const setTimeoutFn = options.setTimeoutFn ?? setTimeout;
  const clearTimeoutFn = options.clearTimeoutFn ?? clearTimeout;
  let state = initialModelScanJobState();
  let disposed = false;
  let timer: ReturnType<typeof setTimeout> | null = null;
  let generation = 0;

  // `disposed` and `state` are read back through these two functions rather
  // than directly. Both are reassigned from outside the awaiting code -- by
  // `dispose`, and by every `apply` -- so a direct read is narrowed by the
  // checker to whatever it was before the `await`, and the guards that follow
  // an await look dead when they are the only thing keeping a disposed session
  // from publishing. A call returning a declared type is not narrowed, which is
  // what makes those guards readable as the live code they are.
  const isDisposed = (): boolean => disposed;
  const readState = (): ModelScanJobViewState => state;

  const publish = (next: ModelScanJobViewState) => {
    state = next;
    options.onChange?.(state);
  };

  const apply = (event: ModelScanJobEvent) => {
    if (isDisposed()) {
      return;
    }
    publish(reduceModelScanJob(readState(), event));
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
          const phase = readState().phase;
          if (phase === "pending" || phase === "running") {
            schedulePoll(gen, statusRoute);
          } else {
            stopPolling();
          }
        } catch (error: unknown) {
          if (isDisposed() || gen !== generation) {
            return;
          }
          const message = error instanceof Error ? error.message : "model_scan_poll_failed";
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
    startScan: async () => {
      if (isDisposed() || !canSubmitModelScanJob(readState())) {
        return;
      }
      generation += 1;
      const gen = generation;
      stopPolling();
      apply({ type: "submit_started" });
      try {
        const receipt = await options.api.submit();
        if (isDisposed() || gen !== generation) {
          return;
        }
        apply({ type: "submit_succeeded", receipt });
        const submitted = readState();
        if (submitted.phase === "pending" || submitted.phase === "running") {
          const route = submitted.statusRoute;
          if (route !== null) {
            // Immediate first poll, then interval.
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
              const message = error instanceof Error ? error.message : "model_scan_poll_failed";
              apply({ type: "poll_failed", message });
            }
          }
        }
      } catch (error: unknown) {
        if (isDisposed() || gen !== generation) {
          return;
        }
        const message = error instanceof Error ? error.message : "model_scan_submit_failed";
        apply({ type: "submit_failed", message });
      }
    },
  };
}
