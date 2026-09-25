// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio frontend API
// Studio API: parameter fitting.
import { post } from "./http";

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
}

/** What `POST /api/fits` takes. */
export interface FitRequestBody {
  catalogue_model?: string;
  schema?: Record<string, unknown>;
  observable: string;
  domains: FitDomain[];
  fixed: Record<string, number>;
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
  fitted: Record<string, number>;
  training_loss: number;
  training: FitRecordingError[];
  holdout: FitRecordingError[];
  optimiser: {
    method: string;
    generations_run: number;
    evaluations: number;
    failed_trials: number;
    converged: boolean;
    message: string;
    history: { generation: number; loss: number }[];
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
