// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio frontend API
// Studio API: parameter fitting.
import { get, post } from "./http";

/** One fitted parameter's search domain. */
export interface FitDomain {
  name: string;
  low: number;
  high: number;
  scale: "linear" | "log";
}

/** One stimulus and the observed response. */
export interface FitRecording {
  name: string;
  current: number[];
  observed: number[];
  group?: string;
}

/** What `POST /api/fits` takes. */
export interface FitRequestBody {
  catalogue_model?: string;
  schema?: Record<string, unknown>;
  observable: string;
  domains: FitDomain[];
  fixed: Record<string, number>;
  constraints?: FitConstraint[];
  train: FitRecording[];
  holdout: FitRecording[];
  seed: number;
  generations: number;
  population: number;
}

/** One recording's error under the fitted parameters. */
export interface FitRecordingError {
  recording: string;
  rmse: number | null;
  diverged: boolean;
}

/** A direction in parameter space the data do not constrain. */
export interface FitUnconstrainedDirection {
  relative_eigenvalue: number;
  direction: Record<string, number>;
}

/** What `POST /api/fits` answers. */
export interface FitResult {
  schema_version: string;
  problem: { domains: FitDomain[]; constraints?: FitConstraint[] };
  fitted: Record<string, number>;
  training_loss: number | null;
  training: FitRecordingError[];
  holdout: FitRecordingError[];
  optimiser: {
    method: string;
    generations_run: number;
    evaluations: number;
    failed_trials: number;
    converged: boolean;
    message: string;
    history: { generation: number; loss: number | null }[];
  };
  identifiability: {
    identifiable: boolean;
    condition_number?: number | null;
    unconstrained_directions?: FitUnconstrainedDirection[];
    reason?: string;
  };
  uncertainty: {
    method: string;
    standard_errors: Record<string, number> | null;
    correlation: Record<string, Record<string, number>> | null;
    reason?: string;
  };
  result_sha256: string;
}

/** What `POST /api/fits/replay` answers. */
export interface FitReplay {
  reproduced: boolean;
  exported_sha256: string;
  replayed_sha256: string;
}

/**
 * Fit a model's parameters to a split cohort.
 *
 * @param body - The problem.
 * @returns The result, bound under one digest.
 */
export const runFit = (body: FitRequestBody) => post<FitResult>("/fits", body);

/**
 * Run an exported fit again.
 *
 * @param result - The exported result.
 * @returns Whether it reproduced, with both digests.
 */
export const replayFit = (result: unknown) => post<FitReplay>("/fits/replay", { result });

/** A receipt and its owner-bound polling route. */
export interface LaboratorySubmission { job_id: string; status_route: string }
/** Existing process supervisor status, scoped to the signed-in caller. */
export interface LaboratoryJob<T> {
  job_id: string;
  status: "pending" | "running" | "cancelling" | "completed" | "failed" | "cancelled" | "timed_out";
  error: string | null;
  result: T | null;
  admission: { laboratory_task?: string };
}
/** A constraint expressed in parameter value space. */
export interface FitConstraint { name: string; coefficients: Record<string, number>; low: number; high: number }
/**
 * Submit a fit to the bounded background process supervisor.
 *
 * @param body - Complete fitting request.
 * @returns The server response with custody preserved.
 */
export const submitFit = (body: FitRequestBody) => post<LaboratorySubmission>("/fits/jobs", body);
/**
 * Submit deterministic replay with the same admission as a new fit.
 *
 * @param result - Exported fitting result.
 * @returns The server response with custody preserved.
 */
export const submitFitReplay = (result: unknown) => post<LaboratorySubmission>("/fits/replay/jobs", { result });
/**
 * Poll a laboratory task belonging to the current caller.
 *
 * @param id - Submitted job identity.
 * @returns The server response with custody preserved.
 */
export const laboratoryJob = <T>(id: string) => get<LaboratoryJob<T>>(`/fits/jobs/${encodeURIComponent(id)}`);
/**
 * Cancel through the existing supervisor and report its actual status.
 *
 * @param id - Submitted job identity.
 * @returns The server response with custody preserved.
 */
export const cancelLaboratoryJob = (id: string) => post<LaboratoryJob<unknown>>(`/fits/jobs/${encodeURIComponent(id)}/cancel`, {});
/** A complete sweep result including every rejected or divergent trial. */
export interface CohortResult {
  schema_version: string;
  cohort: Record<string, unknown>;
  trials: {
    model: string; parameters: Record<string, number>; status: string; rejected_constraints: string[];
    metric: { kind: string; unit: string; observable: string };
    samples: { sample: string; split: string; value: number | null; failed: boolean }[];
    trial_sha256: string;
  }[];
  selection: { model: string; trial_sha256: string | null; training_metric?: number; reason?: string }[];
  measurement_status: string;
  result_sha256: string;
}
/** Comparison based solely on compatible externally supplied measurement receipts. */
export interface MeasuredPareto {
  comparable: boolean; custody: string; reason: string | null;
  contract?: { resource_unit: string; target: string };
  metric?: { unit: string; kind: string };
  rows: { model: string; trial_sha256: string; holdout_error: number; latency_ms: number; resources: number; energy_j: number; nondominated: boolean }[];
}
/**
 * Admit a full cohort with all shared inputs, noise and parameter values.
 *
 * @param cohort - Full versioned scientific protocol.
 * @returns The server response with custody preserved.
 */
export const submitCohort = (cohort: unknown) => post<LaboratorySubmission>("/cohorts/jobs", { cohort });
/**
 * Replay the entire exported sweep, including rejected trials.
 *
 * @param result - Full exported cohort result.
 * @returns The server response with custody preserved.
 */
export const replayCohort = (result: unknown) => post<FitReplay>("/cohorts/replay", { result });
/**
 * Request a receipt-bound measurement comparison; incompatible data yield no frontier.
 *
 * @param result - Complete cohort result.
 * @param receipts - Operator-supplied measurement receipts.
 * @returns The server response with custody preserved.
 */
export const compareMeasurements = (result: unknown, receipts: unknown) => post<MeasuredPareto>("/cohorts/measurements", { result, receipts });
