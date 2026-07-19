// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Model Browser async model-scan job policy (outside UI)

/**
 * Pure state machine and result validation for job-backed catalogue scans.
 * Polling policy lives here; React components only wire presentation.
 */

import type {
  ModelBehavior,
  ModelScanJobReceipt,
  ModelScanMetadata,
  ModelScanResponse,
  StudioJobRecord,
} from "./api/client";

/** Terminal and in-flight phases shown to the operator (no invented progress %). */
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

export interface ModelScanJobViewState {
  behaviors: Record<string, ModelBehavior>;
  error: string | null;
  jobId: string | null;
  phase: ModelScanJobPhase;
  scanMetadata: ModelScanMetadata | null;
  statusRoute: string | null;
}

export type ModelScanJobEvent =
  | { type: "submit_started" }
  | { type: "submit_succeeded"; receipt: ModelScanJobReceipt }
  | { type: "submit_failed"; message: string }
  | { type: "poll"; record: StudioJobRecord }
  | { type: "poll_failed"; message: string };

const BUSY_PHASES: ReadonlySet<ModelScanJobPhase> = new Set([
  "submitting",
  "pending",
  "running",
]);

/**
 * Initial idle view state for a model-scan session.
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
 * True while a submit or poll cycle is in flight (duplicate Scan is forbidden).
 */
export function isModelScanJobBusy(phase: ModelScanJobPhase): boolean {
  return BUSY_PHASES.has(phase);
}

/**
 * True when the Scan control may start a new job.
 */
export function canSubmitModelScanJob(state: ModelScanJobViewState): boolean {
  return !isModelScanJobBusy(state.phase);
}

/**
 * Path-free operator label for the current real job phase.
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
 * Validate a completed job ``result`` as ``studio.model-scan.v1`` evidence.
 */
export function validateModelScanJobResult(
  result: unknown,
):
  | { ok: true; payload: ModelScanResponse }
  | { ok: false; error: string } {
  if (result === null || typeof result !== "object" || Array.isArray(result)) {
    return { ok: false, error: "model_scan_result_not_object" };
  }
  const body = result as Record<string, unknown>;
  if (body.schema_version !== "studio.model-scan.v1") {
    return { ok: false, error: "model_scan_schema_mismatch" };
  }
  if (!Array.isArray(body.models)) {
    return { ok: false, error: "model_scan_models_missing" };
  }
  const meta = body.scan_metadata;
  if (meta === null || typeof meta !== "object" || Array.isArray(meta)) {
    return { ok: false, error: "model_scan_metadata_missing" };
  }
  const metadata = meta as Record<string, unknown>;
  if (metadata.schema_version !== "studio.model-scan.v1") {
    return { ok: false, error: "model_scan_metadata_schema_mismatch" };
  }
  if (metadata.evidence_classification !== "analysis") {
    return { ok: false, error: "model_scan_evidence_class_invalid" };
  }
  if (metadata.status !== "completed") {
    return { ok: false, error: "model_scan_metadata_status_invalid" };
  }
  return {
    ok: true,
    payload: {
      models: body.models as ModelBehavior[],
      scan_metadata: metadata as unknown as ModelScanMetadata,
      schema_version: "studio.model-scan.v1",
    },
  };
}

function publicErrorMessage(raw: string): string {
  const trimmed = raw.trim();
  if (trimmed.length === 0) {
    return "model_scan_failed";
  }
  // Strip absolute/home paths if a backend ever leaks them.
  return trimmed.replace(/\/(?:home|media|tmp|var)\/[^\s"']+/g, "[path]");
}

function behaviorsFromPayload(payload: ModelScanResponse): Record<string, ModelBehavior> {
  const map: Record<string, ModelBehavior> = {};
  for (const model of payload.models) {
    if (model && typeof model.name === "string") {
      map[model.name] = model;
    }
  }
  return map;
}

/**
 * Reduce one model-scan job event into the next view state.
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
      const receipt = event.receipt;
      if (
        receipt.schema_version !== "studio.model-scan.job.v1"
        || receipt.execution_mode !== "async_job"
        || typeof receipt.job_id !== "string"
        || receipt.job_id.length === 0
        || typeof receipt.status_route !== "string"
        || receipt.status_route.length === 0
      ) {
        return {
          behaviors: {},
          error: "model_scan_job_receipt_invalid",
          jobId: null,
          phase: "malformed",
          scanMetadata: null,
          statusRoute: null,
        };
      }
      const initialStatus = receipt.job?.status;
      const phase: ModelScanJobPhase =
        initialStatus === "running"
          ? "running"
          : initialStatus === "pending"
            ? "pending"
            : "pending";
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
      const status = event.record.status;
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
          error: publicErrorMessage(event.record.error ?? "model_scan_job_failed"),
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
      if (status === "completed") {
        const validated = validateModelScanJobResult(event.record.result);
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
          behaviors: behaviorsFromPayload(validated.payload),
          error: null,
          phase: "completed",
          scanMetadata: validated.payload.scan_metadata,
        };
      }
      return {
        ...state,
        behaviors: {},
        error: "model_scan_job_status_unknown",
        phase: "failed",
        scanMetadata: null,
      };
    }
    default: {
      const _exhaustive: never = event;
      return _exhaustive;
    }
  }
}

export interface ModelScanJobApi {
  fetchJob: (statusRoute: string) => Promise<StudioJobRecord>;
  submit: () => Promise<ModelScanJobReceipt>;
}

export interface ModelScanJobSessionOptions {
  api: ModelScanJobApi;
  clearTimeoutFn?: typeof clearTimeout;
  onChange?: (state: ModelScanJobViewState) => void;
  pollIntervalMs?: number;
  setTimeoutFn?: typeof setTimeout;
}

export interface ModelScanJobSession {
  dispose: () => void;
  getState: () => ModelScanJobViewState;
  startScan: () => Promise<void>;
}

/**
 * Create a non-React session that submits one scan job and polls to terminal.
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

  const publish = (next: ModelScanJobViewState) => {
    state = next;
    options.onChange?.(state);
  };

  const apply = (event: ModelScanJobEvent) => {
    if (disposed) {
      return;
    }
    publish(reduceModelScanJob(state, event));
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
          const phase = state.phase;
          if (phase === "pending" || phase === "running") {
            schedulePoll(gen, statusRoute);
          } else {
            stopPolling();
          }
        } catch (error: unknown) {
          if (disposed || gen !== generation) {
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
    getState: () => state,
    startScan: async () => {
      if (disposed || !canSubmitModelScanJob(state)) {
        return;
      }
      generation += 1;
      const gen = generation;
      stopPolling();
      apply({ type: "submit_started" });
      try {
        const receipt = await options.api.submit();
        if (disposed || gen !== generation) {
          return;
        }
        apply({ type: "submit_succeeded", receipt });
        if (state.phase === "pending" || state.phase === "running") {
          const route = state.statusRoute;
          if (route !== null) {
            // Immediate first poll, then interval.
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
              const message = error instanceof Error ? error.message : "model_scan_poll_failed";
              apply({ type: "poll_failed", message });
            }
          }
        }
      } catch (error: unknown) {
        if (disposed || gen !== generation) {
          return;
        }
        const message = error instanceof Error ? error.message : "model_scan_submit_failed";
        apply({ type: "submit_failed", message });
      }
    },
  };
}
