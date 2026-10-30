// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Studio frontend API
// Studio API: candidate model packages.
import { encodeArtifactPath, get, getBlob, post } from "./http";
import { parseCandidateJob, parseCandidateJobHistory, parseCandidateJobReceipt } from "../candidateJobs";
import { parseCatalogueCandidate } from "../candidateCatalogue";

/**
 * Prepare a complete canonical model as an explicitly unfinished workspace proposal.
 *
 * @param modelName - Selected registered catalogue class.
 * @returns Checked provenance and unchanged candidate JSON text.
 */
export const candidateFromCatalogue = (modelName: string) =>
  get<unknown>(`/candidates/from-catalogue/${encodeURIComponent(modelName)}`)
    .then((value) => parseCatalogueCandidate(value, modelName));

/** One problem with a candidate, located by a JSON pointer. */
export interface CandidateDiagnostic {
  location: string;
  message: string;
}

/** What `POST /api/candidates/validate` answers. */
export type CandidateValidation = {
  schema_version: string;
  diagnostics: CandidateDiagnostic[];
} & ({ valid: true; candidate_sha256: string } | { valid: false; candidate_sha256: null });

/** One value (state or parameter) compared with the parent's. */
export interface CandidateValueRow {
  name: string;
  status: "added" | "removed" | "changed" | "unchanged";
  parent?: number;
  candidate?: number;
}

/** One equation compared with the parent's, as text and as mathematics. */
export interface CandidateEquationRow {
  variable: string;
  status: "added" | "removed" | "unchanged" | "equivalent" | "changed" | "undecided" | "not_comparable";
  parent?: string;
  candidate?: string;
  difference?: string;
  reason?: string;
}

/** What `POST /api/candidates/diff` answers. */
export interface CandidateDiff {
  schema_version: string;
  parent: string | null;
  status: "compared" | "no_parent" | "parent_has_no_schema";
  parent_schema?: string;
  state?: CandidateValueRow[];
  parameters?: CandidateValueRow[];
  dynamics?: CandidateEquationRow[];
  reset?: CandidateEquationRow[];
  threshold?: { status: string; difference?: string; reason?: string };
  integration?: { status: string; changed_fields: string[] };
}

/** Stored result of a completed candidate simulation. */
interface CandidateRunFields {
  candidate: string;
  candidate_sha256: string;
  units: { current: string; time: string };
  steps: number;
  current: number;
  profile: { method: string; dt: number; time_unit: string };
  spike_count: number;
  spike_steps: number[];
  sample_every: number;
  trace: Record<string, number[]>;
}

/** A completed trace retains a final state, or a located divergence and no final state. */
export type CandidateRun = CandidateRunFields & (
  | { final_state: Record<string, number>; diverged_at_step: null; divergence: null }
  | { final_state: null; diverged_at_step: number; divergence: string }
);

/** One reference test's outcome in a review packet. */
export interface CandidateTestResult {
  name: string;
  passed: boolean;
  diverged_at_step: number | null;
  checks: { quantity: string; observed: number | null; held: boolean }[];
}

/** Stored result of a completed candidate reference review. */
export interface CandidateReviewPacket {
  schema_version: string;
  candidate_sha256: string;
  packet_sha256: string;
  reference_tests: CandidateTestResult[];
  reference_tests_passed: boolean;
  not_established: string[];
}

/**
 * Report every problem with a candidate package.
 *
 * @param candidate - The parsed package.
 * @returns The located diagnostics; an invalid candidate is still HTTP 200.
 */
export const validateCandidate = (candidate: unknown) =>
  post<CandidateValidation>("/candidates/validate", { candidate });

/**
 * Diff a valid candidate against the catalogue model it names as parent.
 *
 * @param candidate - The parsed package.
 * @returns The diff; an invalid candidate is refused with HTTP 422.
 */
export const diffCandidate = (candidate: unknown) =>
  post<CandidateDiff>("/candidates/diff", { candidate });

/**
 * Simulate a valid candidate under its own numerical profile.
 *
 * @param candidate - The parsed package.
 * @param current - Constant injected current, in the candidate's current unit.
 * @param steps - Integration steps to take.
 * @returns The admitted durable job receipt.
 */
export const simulateCandidate = (candidate: unknown, current: number, steps: number) =>
  post<unknown>("/candidates/simulate", { candidate, current, steps })
    .then((value) => parseCandidateJobReceipt(value, "simulate"));

/**
 * Run a valid candidate's reference tests and assemble its review packet.
 *
 * @param candidate - The parsed package.
 * @returns The admitted durable job receipt.
 */
export const candidateReviewPacket = (candidate: unknown) =>
  post<unknown>("/candidates/review-packet", { candidate })
    .then((value) => parseCandidateJobReceipt(value, "review"));

/**
 * Retrieve the current actor's latest candidate jobs without large result arrays.
 *
 * @returns Checked newest-first ledger summaries.
 */
export const listCandidateJobs = () =>
  get<unknown>("/candidates/jobs").then(parseCandidateJobHistory);

/**
 * Recover one candidate job, checking the response against the requested identifier.
 *
 * @param jobId - Exact identifier returned by candidate job admission.
 * @returns Checked state and any retained scientific result.
 */
export const getCandidateJob = (jobId: string) =>
  get<unknown>(`/candidates/jobs/${encodeURIComponent(jobId)}`)
    .then((value) => parseCandidateJob(value, jobId));

/**
 * Request cancellation and retain the actual supervisor state.
 *
 * @param jobId - Exact identifier returned by candidate job admission.
 * @returns Checked actual supervisor state after requesting cancellation.
 */
export const cancelCandidateJob = (jobId: string) =>
  post<unknown>(`/candidates/jobs/${encodeURIComponent(jobId)}/cancel`, {})
    .then((value) => parseCandidateJob(value, jobId));

/**
 * Download an artifact only through the actor-scoped custody route.
 *
 * @param jobId - Exact identifier returned by candidate job admission.
 * @param path - Relative path declared by this job’s artifact manifest.
 * @returns The retained bytes verified by server artifact custody.
 */
export const candidateJobArtifact = (jobId: string, path: string) =>
  getBlob(`/candidates/jobs/${encodeURIComponent(jobId)}/artifacts/${encodeArtifactPath(path)}`);
