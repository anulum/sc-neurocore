// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio frontend API
// Studio API: analysis endpoints.
import { post } from "./http";
import type {
  PrecisionResponse,
  NullclineResponse,
  CompareResponse,
  FreqResponse,
  AnalysisJobRequestBody,
  AnalysisJobReceipt,
  CodegenResponse,
  ReplayPack,
} from "./types";

/**
 * Queue an analysis that is too slow to hold a request open for.
 *
 * The receipt identifies the job; progress arrives on the progress socket
 * rather than on this call.
 *
 * @param request - The analysis to run and what to run it on.
 * @returns The receipt the job is followed by.
 */
export const submitAnalysisJob = (request: AnalysisJobRequestBody) =>
  post<AnalysisJobReceipt>("/analysis/jobs", request);

/**
 * Compute the nullclines of a two-variable system, for the phase portrait.
 *
 * @param req - The system, its parameters, and the ranges to solve over.
 * @returns The two nullclines as point sets.
 */
export const fetchNullclines = (req: Record<string, unknown>) => post<NullclineResponse>("/nullclines", req);

/**
 * Run the same experiment in float64 and in fixed point and compare them.
 *
 * @param req - The experiment and the fixed-point format to compare against.
 * @returns Both traces, their difference, and the arithmetic that produced it.
 */
export const fetchPrecision = (req: Record<string, unknown>) => post<PrecisionResponse>("/precision", req);

/**
 * Run two configurations and report where they differ.
 *
 * @param a - The first configuration.
 * @param b - The second configuration.
 * @returns Both results and the comparison the server made.
 */
export const fetchCompare = (a: Record<string, unknown>, b: Record<string, unknown>) => post<CompareResponse>("/compare", { config_a: a, config_b: b });

/**
 * Sweep input frequency and report the firing rate at each one.
 *
 * @param req - The model, the amplitude, and the frequencies to sweep.
 * @returns The rate at each frequency.
 */
export const fetchFreqResponse = (req: Record<string, unknown>) => post<FreqResponse>("/freq-response", req);

/**
 * Ask for a script that runs the experiment the server resolved.
 *
 * The script carries the pinned request and the digest of the experiment it
 * was generated from, so running it later either reproduces the same
 * experiment or says that it cannot.
 *
 * @param req - The experiment to generate a script for.
 * @returns The script, its one-line form, and the experiment digest.
 */
export const fetchCodegen = (req: Record<string, unknown>) =>
  post<CodegenResponse>("/codegen", req);

/**
 * Ask for a sealed pack that reproduces this experiment elsewhere.
 *
 * The pack carries the request, the resolved experiment, the expectation and
 * the environment, so a clean interpreter can replay it and say whether it
 * matched.
 *
 * @param req - The experiment to seal.
 * @returns The sealed `studio.replay-pack.v2` document.
 */
export const fetchReplayPack = (req: Record<string, unknown>) =>
  post<ReplayPack>("/export/replay-pack", req);

/**
 * Ask for a Jupyter notebook that cites the model and replays a sealed pack.
 *
 * @param req - The experiment to seal.
 * @returns The notebook document (nbformat 4), the pack inline.
 */
export const fetchReplayNotebook = (req: Record<string, unknown>) =>
  post<Record<string, unknown>>("/export/replay-notebook", req);
